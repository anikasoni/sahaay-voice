"""Dialogue manager.

Per TRD §4.8: mode state machine (Safety / Companion / Memory / idle).
Owns the per-turn flow: ASR → features → NLU → baseline → fusion → response
and routes outcomes to the correct response generator and TTS engine.

Safety Mode has priority and is hard-template-only (no LLM, see §9).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core import alerts as alerts_mod
from core import asr as asr_mod
from core import baseline as baseline_mod
from core import features as features_mod
from core import fusion as fusion_mod
from core import nlu as nlu_mod
from core import response as response_mod
from core import tts as tts_mod
from core import wearable as wearable_mod
from core.audio_capture import Utterance
from core.config import thresholds
from core.logger import log, log_turn, new_turn_id, now_iso
from core import evidence as evidence_mod
from core.schemas import (
    AsrResult,
    BaselineDeviation,
    FusionDecision,
    Language,
    Mode,
    NluResult,
    SpeechFeatures,
)


@dataclass
class TurnResult:
    """One full pipeline turn — what UI shows and what we log."""
    turn_id: str
    mode_before: Mode
    mode_after: Mode
    asr: AsrResult
    features: SpeechFeatures
    nlu: NluResult
    deviation: BaselineDeviation
    decision: FusionDecision
    response_text: str
    response_lang: Language
    response_audio_path: Optional[Path]
    audio_path: Optional[Path] = None
    timestamp: str = field(default_factory=now_iso)


@dataclass
class DialogueState:
    mode: Mode = "idle"
    default_lang: Language = "en"
    # Safety
    safety_pending: bool = False
    safety_lang: Language = "en"
    safety_target_phrase: str = ""
    safety_prompt_end_ms: int = 0
    safety_retries_used: int = 0
    # Memory
    active_reminder_id: Optional[str] = None
    active_reminder_label: str = ""
    active_reminder_lang: Language = "en"
    reminder_unack_count: int = 0
    # Last system prompt end-time (for latency measurement)
    last_prompt_end_ms: int = 0
    # Rolling transcript for caregiver context (last 6 turns)
    recent_transcript: list[str] = field(default_factory=list)
    recent_turn_ids: list[str] = field(default_factory=list)

    def remember(self, who: str, text: str, turn_id: str) -> None:
        if text.strip():
            self.recent_transcript.append(f"{who}: {text.strip()}")
            self.recent_turn_ids.append(turn_id)
            self.recent_transcript = self.recent_transcript[-12:]
            self.recent_turn_ids = self.recent_turn_ids[-12:]


class DialogueManager:
    def __init__(self) -> None:
        self.state = DialogueState()

    # --- public ops --------------------------------------------------------

    def reset(self) -> None:
        self.state = DialogueState()
        log("dialogue state reset")

    def enter_safety_mode(self, lang: Optional[Language] = None) -> tuple[str, Optional[Path]]:
        """Forced entry — called by Operator or after a wearable fall event."""
        self.state.mode = "safety"
        self.state.safety_pending = True
        self.state.safety_lang = lang or self.state.default_lang
        prompt_text = response_mod.safety_response("ask", self.state.safety_lang)
        # The Safety target phrase is the *expected* confirmation.
        cfg_p = thresholds()
        # Take the confirmation prefix from prompts; allow flexible matches
        self.state.safety_target_phrase = {
            "en": "i am okay",
            "hi": "main theek hoon",
            "pa": "main theek haan",
        }[self.state.safety_lang]
        audio_path = tts_mod.synthesize(prompt_text, self.state.safety_lang)
        self.state.safety_prompt_end_ms = int(time.time() * 1000)
        self.state.last_prompt_end_ms = self.state.safety_prompt_end_ms
        log("entered Safety Mode", lang=self.state.safety_lang)
        return prompt_text, audio_path

    def deliver_reminder(self, reminder_id: str, label: str, lang: Language) -> tuple[str, Optional[Path]]:
        """Operator/scheduler-driven."""
        self.state.mode = "memory"
        self.state.active_reminder_id = reminder_id
        self.state.active_reminder_label = label
        self.state.active_reminder_lang = lang
        self.state.reminder_unack_count = 0
        text = response_mod.memory_response("deliver", lang, label)
        audio = tts_mod.synthesize(text, lang)
        self.state.last_prompt_end_ms = int(time.time() * 1000)
        log("reminder delivered", reminder_id=reminder_id, label=label)
        return text, audio

    def handle_no_reply(self, reason: str = "no reply") -> TurnResult:
        """Process a missing response/timeout without requiring audio.

        This closes two PRD loops that a Streamlit push-to-talk UI cannot infer
        automatically: Safety Mode no-confirmation and Memory Mode missed
        acknowledgement. Operator can trigger it during the demo, and kiosk wrappers
        can call the same method on a real timer.
        """
        turn_id = new_turn_id()
        mode_before: Mode = self.state.mode
        lang: Language = self.state.safety_lang if self.state.safety_pending else self.state.active_reminder_lang if self.state.mode == "memory" else self.state.default_lang

        asr = AsrResult(text="", language=lang, avg_logprob=-5.0)
        latency = 0
        if self.state.last_prompt_end_ms:
            latency = max(0, int(time.time() * 1000) - self.state.last_prompt_end_ms)
        feats = SpeechFeatures(
            latency_ms=latency,
            latency_source="timeout_elapsed_after_prompt",
            latency_status="prompt_reply_timeout" if latency else "not_measured",
            speech_duration_ms=0,
            pause_total_ms=0,
            pause_count=0,
            audio_duration_ms=0,
            voiced_duration_ms=0,
            unvoiced_total_ms=0,
            leading_silence_ms=0,
            trailing_silence_ms=0,
            internal_pause_total_ms=0,
            internal_pause_count=0,
            mean_internal_pause_ms=0.0,
            voice_activity_ratio=0.0,
            speech_rate_cps=0.0,
            clarity_score=0.0,
            repetition_score=0.0,
            asr_avg_logprob=-5.0,
        )
        nlu = NluResult(intent="no_reply", intent_confidence=1.0, emotion="confused")
        dev = BaselineDeviation(max_z=0.0, exceed_count=0, sufficient_history=False)
        wearable_ev = wearable_mod.latest()
        decision = fusion_mod.decide(
            nlu, feats, dev, wearable_ev,
            safety_pending=self.state.safety_pending,
            reminder_unack_count=self.state.reminder_unack_count,
            is_safety_reply=self.state.safety_pending,
            reminder_active=self.state.mode == "memory" and self.state.active_reminder_id is not None,
            no_reply=True,
        )
        response_text, response_audio, mode_after = self._route(
            asr, nlu, feats, decision, wearable_ev, lang, turn_id,
        )
        self.state.remember("elder", f"[{reason}]", turn_id)
        if response_text:
            self.state.remember("sahaay", response_text, turn_id)
        result = TurnResult(
            turn_id=turn_id,
            mode_before=mode_before,
            mode_after=mode_after,
            asr=asr,
            features=feats,
            nlu=nlu,
            deviation=dev,
            decision=decision,
            response_text=response_text,
            response_lang=lang,
            response_audio_path=response_audio,
            audio_path=None,
        )
        self._persist_log(result)
        self.state.last_prompt_end_ms = int(time.time() * 1000)
        return result

    # --- the main per-utterance pipeline -----------------------------------

    def handle_utterance(self, utt: Utterance, audio_path: Optional[Path] = None) -> TurnResult:
        turn_id = new_turn_id()
        mode_before: Mode = self.state.mode

        # --- 1) ASR ---------------------------------------------------------
        asr = asr_mod.transcribe(utt)
        resp_lang: Language = "en"
        if asr.language in ("en", "hi", "pa"):
            resp_lang = asr.language  # type: ignore[assignment]
            self.state.default_lang = resp_lang
        else:
            resp_lang = self.state.default_lang

        # --- 2) Speech features --------------------------------------------
        # In Streamlit push-to-talk mode we do not receive the exact first-voice
        # wall-clock timestamp. We estimate latency as:
        #   submit_time - prompt_end_time - recording_duration + leading_silence
        # The feature extractor adds leading_silence_ms to this value.
        prompt_to_recording_start_ms: Optional[int] = None
        if self.state.last_prompt_end_ms > 0:
            prompt_to_recording_start_ms = max(
                0,
                int(time.time() * 1000) - self.state.last_prompt_end_ms - int(utt.duration_ms),
            )
        expected = self.state.safety_target_phrase if self.state.safety_pending else None
        feats = features_mod.extract(
            utt, asr,
            prompt_end_ts_ms=prompt_to_recording_start_ms,
            expected_repetition=expected,
        )

        # --- 3) NLU --------------------------------------------------------
        if asr.is_empty():
            nlu = NluResult(intent="casual_chat", intent_confidence=0.0, emotion="calm")
        else:
            nlu = nlu_mod.analyze(asr.text)
            # Hard safety override: self-harm language must notify caregiver
            # even if the statistical NLI model labels it as loneliness/sadness.
            try:
                if response_mod.contains_self_harm(asr.text, resp_lang):
                    nlu.intent = "self_harm_risk"
                    nlu.intent_confidence = max(nlu.intent_confidence, 0.99)
            except Exception:
                pass

        # --- 4) ASR/NLU quality gate --------------------------------------
        # Phase 1.6: A weak transcript must not be treated as understood.
        # If the ASR confidence/clarity and NLU confidence are poor in ordinary
        # Companion/idle mode, we ask the elder to repeat and do NOT update the
        # baseline. Critical intents still pass through the safety guardrails.
        asr_quality = thresholds().get("asr", {})
        try:
            asr_conf = max(0.0, min(1.0, math.exp(float(asr.avg_logprob))))
        except Exception:
            asr_conf = 0.0
        critical_intents = {
            "self_harm_risk", "clinical_question", "emergency_help",
            "caregiver_request", "safety_confirmation",
        }
        nlu_low = nlu.intent_confidence < float(asr_quality.get("nlu_low_confidence_threshold", 0.40))
        acoustic_or_asr_weak = (
            asr_conf < float(asr_quality.get("confidence_usable_threshold", 0.60))
            or feats.clarity_score < float(asr_quality.get("clarity_weak_threshold", 0.45))
            or feats.voiced_duration_ms < int(asr_quality.get("min_voiced_ms", 700))
        )
        very_weak_asr = asr_conf < 0.30
        weak_transcript = (nlu_low and acoustic_or_asr_weak) or very_weak_asr
        ordinary_mode = (not self.state.safety_pending and self.state.mode != "memory")
        if weak_transcript and ordinary_mode and nlu.intent not in critical_intents and not asr.is_empty():
            nlu = NluResult(
                intent="unclear_speech",
                intent_confidence=1.0,
                emotion="confused",
                emotion_confidence=1.0,
            )
            dev = BaselineDeviation(max_z=0.0, exceed_count=0, sufficient_history=False)
            decision = FusionDecision(
                outcome="normal_response",
                reason=(
                    "ASR/NLU confidence weak; asking elder to repeat "
                    f"(asr_conf={asr_conf:.2f}, clarity={feats.clarity_score:.2f})"
                ),
                severity="info",
                notify_caregiver=False,
            )
            response_text = response_mod.asr_error(resp_lang)
            response_audio = tts_mod.synthesize(response_text, resp_lang)
            mode_after = self.state.mode

            self.state.remember("elder", asr.text, turn_id)
            self.state.remember("sahaay", response_text, turn_id)
            result = TurnResult(
                turn_id=turn_id,
                mode_before=mode_before,
                mode_after=mode_after,
                asr=asr,
                features=feats,
                nlu=nlu,
                deviation=dev,
                decision=decision,
                response_text=response_text,
                response_lang=resp_lang,
                response_audio_path=response_audio,
                audio_path=audio_path,
            )
            self._persist_log(result)
            self.state.last_prompt_end_ms = int(time.time() * 1000)
            return result

        # --- 5) Baseline deviation -----------------------------------------
        dev = baseline_mod.compare(feats)

        # --- 6) Fusion -----------------------------------------------------
        wearable_ev = wearable_mod.latest()
        decision = fusion_mod.decide(
            nlu, feats, dev, wearable_ev,
            safety_pending=self.state.safety_pending,
            reminder_unack_count=self.state.reminder_unack_count,
            is_safety_reply=self.state.safety_pending,
            reminder_active=self.state.mode == "memory" and self.state.active_reminder_id is not None,
            no_reply=asr.is_empty(),
        )

        # --- 7) Route to response ------------------------------------------
        response_text, response_audio, mode_after = self._route(
            asr, nlu, feats, decision, wearable_ev, resp_lang, turn_id,
        )

        # --- 8) Update baseline (excluding abnormal and weak transcripts) ---
        abnormal = decision.outcome in ("soft_check_in", "caregiver_alert", "safe_deflection")
        baseline_mod.update(feats, abnormal=abnormal)

        # --- 9) Log turn ---------------------------------------------------
        self.state.remember("elder", asr.text, turn_id)
        if response_text:
            self.state.remember("sahaay", response_text, turn_id)

        result = TurnResult(
            turn_id=turn_id,
            mode_before=mode_before,
            mode_after=mode_after,
            asr=asr,
            features=feats,
            nlu=nlu,
            deviation=dev,
            decision=decision,
            response_text=response_text,
            response_lang=resp_lang,
            response_audio_path=response_audio,
            audio_path=audio_path,
        )
        self._persist_log(result)
        # Update prompt-end-ms for next-turn latency calc
        self.state.last_prompt_end_ms = int(time.time() * 1000)
        return result

    # --- routing logic -----------------------------------------------------

    def _route(
        self,
        asr: AsrResult,
        nlu: NluResult,
        feats: SpeechFeatures,
        decision: FusionDecision,
        wearable_ev,
        lang: Language,
        turn_id: str,
    ) -> tuple[str, Optional[Path], Mode]:
        cfg_s = thresholds()["safety_mode"]

        # If a strong wearable fall just landed and we're not already in safety,
        # the fusion decision says normal_response but the *reason* tells us
        # to enter safety. Detect and switch.
        if (wearable_ev and wearable_ev.type == "fall_detected"
                and wearable_ev.severity in ("medium", "high")
                and not self.state.safety_pending
                and decision.outcome == "normal_response"):
            text, audio = self.enter_safety_mode(lang)
            return text, audio, "safety"

        # --- Safe deflection (clinical_question) ---------------------------
        if decision.outcome == "safe_deflection":
            text = response_mod.clinical_deflection(lang)
            alerts_mod.raise_alert(
                reason=decision.reason,
                severity=decision.severity,
                recent_transcript=list(self.state.recent_transcript),
                related_turn_ids=list(self.state.recent_turn_ids) + [turn_id],
            )
            return text, tts_mod.synthesize(text, lang), self.state.mode

        # --- Caregiver alert -----------------------------------------------
        if decision.outcome == "caregiver_alert":
            alerts_mod.raise_alert(
                reason=decision.reason,
                severity=decision.severity,
                recent_transcript=list(self.state.recent_transcript),
                related_turn_ids=list(self.state.recent_turn_ids) + [turn_id],
            )
            if nlu.intent == "self_harm_risk":
                text = response_mod.companion_response(nlu, asr.text, lang)
                self.state.mode = "companion"
            elif nlu.intent == "caregiver_request":
                from core.config import get_prompt
                text = get_prompt("companion", "acknowledge", lang)
                self.state.mode = "companion"
            elif self.state.safety_pending:
                text = response_mod.safety_response("escalated", lang)
                self.state.safety_pending = False
                self.state.mode = "idle"
            elif self.state.mode == "memory":
                text = response_mod.safety_response("escalated", lang)
                self.state.active_reminder_id = None
                self.state.mode = "idle"
            else:
                # generic ack
                from core.config import get_prompt
                text = get_prompt("safety", "escalated", lang)
            return text, tts_mod.synthesize(text, lang), self.state.mode

        # --- Reminder repeat -----------------------------------------------
        if decision.outcome == "reminder_repeat":
            self.state.reminder_unack_count += 1
            text = response_mod.memory_response(
                "repeat", self.state.active_reminder_lang,
                label=self.state.active_reminder_label,
            )
            return text, tts_mod.synthesize(text, self.state.active_reminder_lang), "memory"

        # --- Soft check-in -------------------------------------------------
        if decision.outcome == "soft_check_in":
            # Phase 1.5: distinguish a gentle wellness check from hard Safety
            # Mode. Loneliness/sadness + speech deviation should sound
            # companion-like first, while fall/manual safety still uses the
            # strict confirmation loop.
            if nlu.intent == "loneliness_expression" or nlu.emotion == "sad":
                text = response_mod.companion_soft_checkin(lang)
                self.state.mode = "companion"
                self.state.last_prompt_end_ms = int(time.time() * 1000)
                return text, tts_mod.synthesize(text, lang), "companion"
            text = response_mod.soft_checkin_response(lang)
            self.state.mode = "safety"
            self.state.safety_pending = True
            self.state.safety_lang = lang
            self.state.safety_target_phrase = {
                "en": "i am okay",
                "hi": "main theek hoon",
                "pa": "main theek haan",
            }[lang]
            self.state.last_prompt_end_ms = int(time.time() * 1000)
            self.state.safety_prompt_end_ms = self.state.last_prompt_end_ms
            return text, tts_mod.synthesize(text, lang), "safety"

        # --- Normal response in active mode --------------------------------
        # Safety Mode: if we were pending and the reply passed muster, close it.
        if self.state.safety_pending:
            # decision.outcome is normal_response here ⇒ reply was good
            text = response_mod.safety_response("confirmed", lang)
            self.state.safety_pending = False
            self.state.safety_retries_used = 0
            self.state.mode = "idle"
            return text, tts_mod.synthesize(text, lang), "idle"

        # Memory Mode acknowledgement: acknowledge only clear completion words.
        # Unrelated replies cause a repeat; repeated misses eventually escalate.
        if self.state.mode == "memory":
            is_ack = nlu.intent in ("task_acknowledgement", "safety_confirmation")
            try:
                is_ack = is_ack or nlu_mod.is_task_acknowledgement(asr.text)
            except Exception:
                pass
            if is_ack:
                text = response_mod.memory_response("ack", self.state.active_reminder_lang)
                self.state.active_reminder_id = None
                self.state.reminder_unack_count = 0
                self.state.mode = "idle"
                return text, tts_mod.synthesize(text, self.state.active_reminder_lang), "idle"
            self.state.reminder_unack_count += 1
            cfg_m = thresholds()["memory_mode"]
            if self.state.reminder_unack_count > cfg_m["max_repeats_default"]:
                alerts_mod.raise_alert(
                    reason=f"reminder not acknowledged after {self.state.reminder_unack_count - 1} repeats",
                    severity="warning",
                    recent_transcript=list(self.state.recent_transcript),
                    related_turn_ids=list(self.state.recent_turn_ids) + [turn_id],
                )
                text = response_mod.safety_response("escalated", self.state.active_reminder_lang)
                self.state.active_reminder_id = None
                self.state.mode = "idle"
                return text, tts_mod.synthesize(text, self.state.active_reminder_lang), "idle"
            text = response_mod.memory_response("repeat", self.state.active_reminder_lang, self.state.active_reminder_label)
            return text, tts_mod.synthesize(text, self.state.active_reminder_lang), "memory"

        # Companion-style default
        text = response_mod.companion_response(nlu, asr.text, lang)
        # Mode follows intent
        if nlu.intent in ("loneliness_expression", "casual_chat", "caregiver_request"):
            self.state.mode = "companion"
        return text, tts_mod.synthesize(text, lang), self.state.mode

    # --- persistence -------------------------------------------------------

    def _persist_log(self, r: TurnResult) -> None:
        evidence = evidence_mod.turn_to_evidence(r)
        log_turn({
            "turn_id": r.turn_id,
            "timestamp": r.timestamp,
            "mode_before": r.mode_before,
            "mode_after": r.mode_after,
            "audio_path": str(r.audio_path) if r.audio_path else None,
            "asr": r.asr,
            "features": r.features,
            "nlu": r.nlu,
            "baseline_deviation": {
                "max_z": r.deviation.max_z,
                "exceed_count": r.deviation.exceed_count,
                "per_feature_z": r.deviation.per_feature_z,
                "sufficient_history": r.deviation.sufficient_history,
            },
            "fusion_outcome": r.decision.outcome,
            "fusion_reason": r.decision.reason,
            "response_text": r.response_text,
            "response_language": r.response_lang,
            "pipeline_evidence": evidence,
            "model_metadata": {
                "asr": asr_mod.runtime_info(),
                "nlu": nlu_mod.runtime_info(),
                "baseline_config": thresholds().get("baseline", {}),
                "safety_config": thresholds().get("safety_mode", {}),
            },
        })
