"""Operator view — developer panel for demoing scenarios."""
from __future__ import annotations

import json
import streamlit as st

from app.state import get_manager, set_last_turn, push_turn
from core import baseline as baseline_mod
from core import wearable as wearable_mod
from core import alerts as alerts_mod
from core.config import reminders, reload_all
from core.logger import purge_all
from app.ui_evidence import render_ai_evidence_panel


def render_operator() -> None:
    mgr = get_manager()
    st.markdown("### Wearable simulator")
    st.caption("Hardware is out of scope — trigger simulated events here "
               "(per TRD §4.6).")

    cols = st.columns(4)
    triggers = [
        ("Fall (high)", "fall_detected", "high"),
        ("Fall (medium)", "fall_detected", "medium"),
        ("Inactivity", "inactivity_long", "medium"),
        ("Normal", "normal", "low"),
    ]
    for col, (label, etype, sev) in zip(cols, triggers):
        with col:
            if st.button(label, use_container_width=True):
                ev = wearable_mod.publish(etype, sev)
                # A fall should proactively ask for confirmation, not wait for
                # the elder to speak first. Inactivity/abnormal motion are handled
                # as soft check-ins by the fusion layer on the next interaction.
                if etype == "fall_detected" and sev in ("medium", "high"):
                    text, audio = mgr.enter_safety_mode(lang=mgr.state.default_lang)
                    from core.dialogue import TurnResult
                    from core.schemas import AsrResult, SpeechFeatures, NluResult, BaselineDeviation, FusionDecision
                    from core.logger import new_turn_id, now_iso
                    stub = TurnResult(
                        turn_id=new_turn_id(),
                        mode_before="idle",
                        mode_after="safety",
                        asr=AsrResult(text="", language=mgr.state.default_lang),
                        features=SpeechFeatures(),
                        nlu=NluResult(intent="wearable_event", intent_confidence=1.0, emotion="calm"),
                        deviation=BaselineDeviation(max_z=0, exceed_count=0),
                        decision=FusionDecision(outcome="normal_response", reason=f"{etype} ({sev}) — safety check started"),
                        response_text=text,
                        response_lang=mgr.state.default_lang,
                        response_audio_path=audio,
                        timestamp=now_iso(),
                    )
                    set_last_turn(stub)
                    push_turn(stub)
                    st.toast("Fall event triggered Safety Mode. Switch to Elder tab to respond.")
                else:
                    st.toast(f"Wearable event: {etype} ({sev})")
                st.rerun()

    if st.button("Clear wearable event"):
        wearable_mod.clear()
        st.toast("Wearable event cleared.")
        st.rerun()

    st.divider()
    st.markdown("### Simulate missing response / timeout")
    st.caption("Use this after a Safety prompt or reminder when the elder does not answer. This exercises PRD A2 and A4.")
    if st.button("No reply / timeout now", type="primary"):
        turn = mgr.handle_no_reply("operator simulated timeout")
        set_last_turn(turn)
        push_turn(turn)
        st.toast(f"Processed timeout: {turn.decision.outcome}")
        st.rerun()

    # Show the latest event
    ev = wearable_mod.latest()
    if ev:
        st.markdown(
            f'<div class="sv-card sv-card-warning">'
            f'<b>Active wearable event:</b> {ev.type} · severity {ev.severity}'
            f' · age {ev.age_ms} ms</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("No active wearable event.")

    st.divider()

    # --- Trigger Safety Mode manually ---
    st.markdown("### Force Safety Mode")
    st.caption("Useful for demoing the safety-check prompt without a wearable event.")
    cols = st.columns(3)
    for col, lang in zip(cols, ["en", "hi", "pa"]):
        with col:
            if st.button(f"Trigger ({lang})", use_container_width=True):
                text, audio = mgr.enter_safety_mode(lang=lang)
                # Synthesize a TurnResult-shaped object so the elder view shows it
                from core.dialogue import TurnResult
                from core.schemas import (
                    AsrResult, SpeechFeatures, NluResult, BaselineDeviation,
                    FusionDecision,
                )
                from core.logger import new_turn_id, now_iso
                stub = TurnResult(
                    turn_id=new_turn_id(),
                    mode_before="idle",
                    mode_after="safety",
                    asr=AsrResult(text="", language=lang),
                    features=SpeechFeatures(),
                    nlu=NluResult(intent="safety_confirmation", intent_confidence=1.0, emotion="calm"),
                    deviation=BaselineDeviation(max_z=0, exceed_count=0),
                    decision=FusionDecision(outcome="normal_response",
                                            reason="operator-triggered safety mode"),
                    response_text=text,
                    response_lang=lang,
                    response_audio_path=audio,
                    timestamp=now_iso(),
                )
                set_last_turn(stub)
                push_turn(stub)
                st.toast(f"Safety Mode entered ({lang}). Switch to the Elder tab to see and respond.")
                st.rerun()

    st.divider()

    # --- Reminders ---
    st.markdown("### Trigger a reminder")
    st.caption("Pulls from `config/reminders.yaml`. Memory Mode awaits acknowledgement.")
    rems = reminders().get("reminders", [])
    if rems:
        labels = [f"{r['reminder_id']}  —  {r['label']} ({r['language']})" for r in rems]
        pick = st.selectbox("Reminder", labels, key="op_reminder_pick")
        if st.button("Deliver reminder now"):
            r = rems[labels.index(pick)]
            label_field = {"en": "label", "hi": "label_hi", "pa": "label_pa"}[r["language"]]
            label = r.get(label_field) or r["label"]
            text, audio = mgr.deliver_reminder(r["reminder_id"], label, r["language"])
            from core.dialogue import TurnResult
            from core.schemas import (
                AsrResult, SpeechFeatures, NluResult, BaselineDeviation, FusionDecision,
            )
            from core.logger import new_turn_id, now_iso
            stub = TurnResult(
                turn_id=new_turn_id(),
                mode_before="idle",
                mode_after="memory",
                asr=AsrResult(text="", language=r["language"]),
                features=SpeechFeatures(),
                nlu=NluResult(intent="reminder_request", intent_confidence=1.0, emotion="calm"),
                deviation=BaselineDeviation(max_z=0, exceed_count=0),
                decision=FusionDecision(outcome="normal_response",
                                        reason=f"reminder delivered: {r['reminder_id']}"),
                response_text=text,
                response_lang=r["language"],
                response_audio_path=audio,
                timestamp=now_iso(),
            )
            set_last_turn(stub)
            push_turn(stub)
            st.toast("Reminder delivered. Elder can now respond.")
            st.rerun()

    if st.button("Reload reminders from disk"):
        reload_all()
        st.toast("Configs reloaded.")
        st.rerun()

    st.divider()
    render_ai_evidence_panel()

    st.divider()
    st.markdown("### Accuracy / acceptance checks")
    st.caption("Run these in the VS Code terminal. They create JSON reports in data/.")
    st.code("python -m pytest tests -q\npython scripts/evaluate.py --fusion\npython scripts/acceptance_check.py", language="powershell")

    st.divider()

    # --- Baseline ---
    st.markdown("### Baseline")
    snap = baseline_mod.snapshot()
    st.json(snap)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reset baseline", type="secondary"):
            baseline_mod.reset()
            st.toast("Baseline reset.")
            st.rerun()
    with col2:
        if st.button("Delete all user data (logs + baseline + alerts)",
                     type="secondary"):
            baseline_mod.reset()
            n_logs = purge_all()
            n_alerts = alerts_mod.clear_all()
            st.toast(f"Wiped: {n_logs} log files, {n_alerts} alerts, baseline.")
            st.rerun()
