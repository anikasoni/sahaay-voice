"""Fusion and escalation.

Per TRD §4.7: rule-based decision combining NLU intent, speech features,
baseline deviation, recent wearable event, and dialogue state.

Every decision carries a human-readable `reason` (FR-34) suitable for the
caregiver panel.
"""
from __future__ import annotations

from typing import Optional

from core.config import thresholds
from core.logger import log
from core.schemas import (
    BaselineDeviation,
    FusionDecision,
    NluResult,
    SpeechFeatures,
    WearableEvent,
)


def decide(
    nlu: NluResult,
    features: SpeechFeatures,
    deviation: BaselineDeviation,
    wearable: Optional[WearableEvent],
    *,
    safety_pending: bool = False,
    reminder_unack_count: int = 0,
    is_safety_reply: bool = False,
    reminder_active: bool = False,
    no_reply: bool = False,
) -> FusionDecision:
    """Run the fusion rules. Order = priority."""
    cfg_t = thresholds()
    cfg_b = cfg_t["baseline"]
    cfg_s = cfg_t["safety_mode"]
    cfg_m = cfg_t["memory_mode"]

    # -- Rule 0: severe self-harm language -----------------------------------
    if nlu.intent == "self_harm_risk":
        return FusionDecision(
            outcome="caregiver_alert",
            reason="self-harm or severe distress language detected",
            severity="urgent",
            notify_caregiver=True,
        )

    # -- Rule 1: hard fall ---------------------------------------------------
    if wearable and wearable.type == "fall_detected" and wearable.severity in ("medium", "high"):
        # If we're already in Safety Mode waiting for confirmation and the
        # reply was missing or unclear → straight to caregiver alert.
        if safety_pending and is_safety_reply:
            if (features.repetition_score < cfg_s["repetition_threshold"]
                    or features.clarity_score < cfg_s["clarity_threshold"]):
                return FusionDecision(
                    outcome="caregiver_alert",
                    reason="fall reported and no clear confirmation",
                    severity="urgent" if wearable.severity == "high" else "warning",
                    notify_caregiver=True,
                )
        # Otherwise enter Safety Mode to ask
        return FusionDecision(
            outcome="normal_response",     # dialogue manager will switch to safety mode
            reason="fall reported — initiating safety check",
            severity="warning",
            notify_caregiver=False,
        )

    # -- Rule 1b: inactivity/abnormal motion → soft check-in ------------------
    if wearable and wearable.type in ("inactivity_long", "abnormal_motion") and wearable.severity in ("medium", "high"):
        return FusionDecision(
            outcome="soft_check_in",
            reason=f"{wearable.type} reported by simulated wearable",
            severity="warning" if wearable.severity == "high" else "info",
            notify_caregiver=False,
        )

    # -- Rule 2: explicit clinical question → safe deflection ----------------
    if nlu.intent == "clinical_question":
        return FusionDecision(
            outcome="safe_deflection",
            reason="clinical question deflected and caregiver notified",
            severity="warning",
            notify_caregiver=True,
        )

    # -- Rule 3: explicit emergency intent -----------------------------------
    if nlu.intent == "emergency_help":
        return FusionDecision(
            outcome="caregiver_alert",
            reason="user requested help",
            severity="urgent",
            notify_caregiver=True,
        )

    # -- Rule 4: Safety-Mode reply scoring (without a wearable event) --------
    if safety_pending and is_safety_reply:
        if (features.repetition_score < cfg_s["repetition_threshold"]
                or features.clarity_score < cfg_s["clarity_threshold"]):
            return FusionDecision(
                outcome="caregiver_alert",
                reason="safety check — reply unclear or off-target",
                severity="urgent",
                notify_caregiver=True,
            )
        return FusionDecision(
            outcome="normal_response",
            reason="safety check confirmed",
            severity="info",
        )

    # -- Rule 5: missed reminders / no reply ---------------------------------
    if reminder_active and no_reply:
        max_repeats = cfg_m["max_repeats_default"]
        if reminder_unack_count >= max_repeats:
            return FusionDecision(
                outcome="caregiver_alert",
                reason=f"reminder not acknowledged after {reminder_unack_count} repeats",
                severity="warning",
                notify_caregiver=True,
            )
        return FusionDecision(
            outcome="reminder_repeat",
            reason=f"reminder not acknowledged; repeating ({reminder_unack_count + 1}/{max_repeats})",
            severity="info",
            notify_caregiver=False,
        )

    if reminder_unack_count >= cfg_m["max_repeats_default"] + 1:
        return FusionDecision(
            outcome="caregiver_alert",
            reason=f"reminder not acknowledged after {reminder_unack_count} attempts",
            severity="warning",
            notify_caregiver=True,
        )

    # -- Rule 6: caregiver request -------------------------------------------
    if nlu.intent == "caregiver_request":
        return FusionDecision(
            outcome="caregiver_alert",
            reason="user requested caregiver contact",
            severity="info",
            notify_caregiver=True,
        )

    # -- Rule 7: strong baseline deviation → soft check-in -------------------
    if (deviation.sufficient_history
            and deviation.max_z >= cfg_b["z_high"]
            and deviation.exceed_count >= cfg_b["exceed_count_for_soft_check_in"]):
        return FusionDecision(
            outcome="soft_check_in",
            reason=(f"speech pattern deviates from baseline "
                    f"(max_z={deviation.max_z:.1f}, "
                    f"features_off={deviation.exceed_count})"),
            severity="info",
        )

    # -- Default: normal response in active mode -----------------------------
    decision = FusionDecision(
        outcome="normal_response",
        reason="ok",
        severity="info",
    )
    log("fusion", outcome=decision.outcome, reason=decision.reason)
    return decision
