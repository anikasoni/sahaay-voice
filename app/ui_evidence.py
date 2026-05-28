"""Operator-facing AI evidence panel.

Shows exactly where speech processing, NLU, baseline comparison, and fusion are
being used for the latest turn. This is Phase 1 of making the prototype look
and behave like an auditable speech/NLP system rather than a generic chatbot.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from app.state import get_last_turn, get_recent_turns_list
from core import asr as asr_mod
from core import baseline as baseline_mod
from core import evidence as evidence_mod
from core import nlu as nlu_mod
from core.config import DATA_DIR, thresholds
from core.logger import read_recent_turns


def _pct(x: Any) -> str:
    try:
        return f"{100 * float(x):.0f}%"
    except Exception:
        return "—"


def _num(x: Any, suffix: str = "") -> str:
    try:
        return f"{float(x):.3g}{suffix}"
    except Exception:
        return "—"


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(obj)


def render_ai_evidence_panel() -> None:
    st.markdown("### AI pipeline evidence")
    st.caption(
        "This panel exposes the actual PRD/TRD pipeline: audio → ASR → speech features → "
        "NLU → baseline comparator → fusion/escalation → response."
    )

    with st.expander("Model stack currently active", expanded=True):
        asr_info = asr_mod.runtime_info()
        nlu_info = nlu_mod.runtime_info()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("ASR backend", asr_info.get("backend", "—"))
            st.write(
                f"Model: `{asr_info.get('loaded_model_size') or asr_info.get('configured_model_size')}`  "
                f"Device: `{asr_info.get('loaded_device') or asr_info.get('configured_device')}`  "
                f"Compute: `{asr_info.get('loaded_compute_type') or asr_info.get('configured_compute_type')}`"
            )
        with c2:
            st.metric("NLU engine", str(nlu_info.get("engine", "—"))[:32])
            st.write(
                f"Intents: `{nlu_info.get('intent_count')}`  "
                f"Emotions: `{nlu_info.get('emotion_count')}`  "
                f"Rules: `{nlu_info.get('keyword_fast_paths')}`"
            )
        with c3:
            bcfg = thresholds().get("baseline", {})
            st.metric("Baseline trigger", f"z ≥ {bcfg.get('z_high', '—')}")
            st.write(
                f"Min samples: `{bcfg.get('min_samples_before_flagging', '—')}`  "
                f"Exceed count: `{bcfg.get('exceed_count_for_soft_check_in', '—')}`"
            )
        st.markdown("**Raw model metadata**")
        st.json({"asr": asr_info, "nlu": nlu_info, "baseline_config": bcfg})

    turn = get_last_turn()
    if not turn:
        st.info("No turn yet. Record in the Elder tab or trigger Safety/Memory from Operator.")
        return

    ev = evidence_mod.turn_to_evidence(turn)
    asr_ev = ev["asr"]
    feat_ev = ev["speech_features"]
    nlu_ev = ev["nlu"]
    base_ev = ev["baseline"]
    fusion_ev = ev["fusion"]

    st.markdown("#### Latest turn: decision evidence")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ASR confidence", _pct(asr_ev.get("approx_confidence")))
    m2.metric("Intent", str(nlu_ev.get("intent", "—"))[:22], _pct(nlu_ev.get("intent_confidence")))
    m3.metric("Clarity", _pct(feat_ev.get("clarity_score")))
    m4.metric("Baseline max-z", _num(base_ev.get("max_z")))
    m5.metric("Fusion outcome", str(fusion_ev.get("outcome", "—"))[:24], fusion_ev.get("severity", "info"))

    st.markdown(
        f'<div class="sv-card sv-card-info"><b>Decision trace:</b> '
        f'ASR heard <code>{asr_ev.get("transcript") or "(silent)"}</code> '
        f'→ NLU predicted <b>{nlu_ev.get("intent")}</b> '
        f'({nlu_ev.get("intent_confidence")}) '
        f'→ baseline max-z <b>{base_ev.get("max_z")}</b> '
        f'→ fusion chose <b>{fusion_ev.get("outcome")}</b>. '
        f'<br><span class="sv-quiet">Reason: {fusion_ev.get("reason")}</span></div>',
        unsafe_allow_html=True,
    )

    tab_asr, tab_feat, tab_nlu, tab_base, tab_fusion, tab_json = st.tabs(
        ["ASR", "Speech features", "NLU", "Baseline", "Fusion/safety", "Raw JSON"]
    )
    with tab_asr:
        st.markdown("**Automatic Speech Recognition evidence**")
        st.write(f"Transcript: `{asr_ev.get('transcript') or '(silent)'}`")
        st.write(f"Detected language: `{asr_ev.get('detected_language')}`")
        st.write(f"Avg logprob: `{asr_ev.get('avg_logprob')}` → approx confidence `{asr_ev.get('approx_confidence')}`")
        st.write(f"Segments detected: `{asr_ev.get('segment_count')}`")
        if asr_ev.get("segments"):
            st.json(asr_ev.get("segments"))
    with tab_feat:
        st.markdown("**Speech processing evidence**")
        st.caption("These are signal-level features, not medical diagnoses.")
        rows = [
            {"feature": "latency_ms", "value": feat_ev.get("latency_ms"), "definition": "estimated prompt end → first voiced frame", "why it matters": "delayed response can support a safety check", "baseline?": "yes"},
            {"feature": "latency_source", "value": feat_ev.get("latency_source"), "definition": "how latency was estimated", "why it matters": "keeps timing evidence auditable", "baseline?": "no"},
            {"feature": "audio_duration_ms", "value": feat_ev.get("audio_duration_ms"), "definition": "full recorder duration", "why it matters": "separates user recording behavior from speech itself", "baseline?": "no"},
            {"feature": "voiced_duration_ms", "value": feat_ev.get("voiced_duration_ms"), "definition": "frames classified as speech", "why it matters": "denominator for speech rate", "baseline?": "no"},
            {"feature": "unvoiced_total_ms", "value": feat_ev.get("unvoiced_total_ms"), "definition": "all non-speech frames", "why it matters": "QA only; not directly baselined", "baseline?": "no"},
            {"feature": "leading_silence_ms", "value": feat_ev.get("leading_silence_ms"), "definition": "silence before first speech frame", "why it matters": "added to prompt-response latency", "baseline?": "no"},
            {"feature": "trailing_silence_ms", "value": feat_ev.get("trailing_silence_ms"), "definition": "silence after last speech frame", "why it matters": "excluded from pause baseline to avoid button-delay artifacts", "baseline?": "no"},
            {"feature": "internal_pause_total_ms", "value": feat_ev.get("internal_pause_total_ms"), "definition": "unvoiced time between first and last voiced frame", "why it matters": "pause burden inside the utterance", "baseline?": "yes as pause_total_ms"},
            {"feature": "internal_pause_count", "value": feat_ev.get("internal_pause_count"), "definition": "internal pauses ≥ ~200 ms", "why it matters": "hesitation/disfluency proxy", "baseline?": "no"},
            {"feature": "mean_internal_pause_ms", "value": feat_ev.get("mean_internal_pause_ms"), "definition": "internal_pause_total / internal_pause_count", "why it matters": "distinguishes one long pause vs many short pauses", "baseline?": "no"},
            {"feature": "voice_activity_ratio", "value": feat_ev.get("voice_activity_ratio"), "definition": "voiced_duration / audio_duration", "why it matters": "quality-control indicator", "baseline?": "no"},
            {"feature": "speech_rate_cps", "value": feat_ev.get("speech_rate_cps"), "definition": "characters per second over voiced duration", "why it matters": "slow/fast speech deviation", "baseline?": "yes"},
            {"feature": "clarity_score", "value": feat_ev.get("clarity_score"), "definition": "ASR logprob + acoustic SNR proxy", "why it matters": feat_ev.get("clarity_status"), "baseline?": "yes"},
            {"feature": "repetition_score", "value": feat_ev.get("repetition_score"), "definition": "Safety phrase similarity", "why it matters": "used to close/alert Safety Mode", "baseline?": "yes, safety turns"},
        ]
        st.dataframe(rows, hide_index=True, use_container_width=True)
        st.caption(f"Pause definition: `{feat_ev.get('pause_definition')}`. Leading/trailing silence is shown for QA but excluded from the pause baseline.")
    with tab_nlu:
        st.markdown("**Natural Language Understanding evidence**")
        st.json(nlu_ev)
        slots = nlu_ev.get("slots") or {}
        if any(slots.values()):
            st.success(f"Slots extracted: {slots}")
        else:
            st.caption("No slots extracted on this turn.")
    with tab_base:
        st.markdown("**Personal baseline comparator evidence**")
        st.caption("Baseline only becomes meaningful after enough normal samples.")
        st.json({"current_deviation": base_ev, "baseline_snapshot": baseline_mod.snapshot()})
        if not base_ev.get("sufficient_history"):
            st.warning("Baseline history is not mature yet; deviation is shown but not used as strong evidence.")
    with tab_fusion:
        st.markdown("**Fusion and safety policy evidence**")
        st.json(fusion_ev)
        outcome = fusion_ev.get("outcome")
        if outcome in ("safe_deflection", "caregiver_alert"):
            st.error("Caregiver/safety path activated.")
        elif outcome == "soft_check_in":
            st.warning("Soft safety check-in activated.")
        else:
            st.success("No escalation required for this turn.")
        st.info("Safety Mode, clinical deflection, and self-harm handling are template/guardrail paths — not free-form LLM output.")
    with tab_json:
        st.download_button(
            "Download latest evidence JSON",
            data=_safe_json(ev),
            file_name=f"sahaay_pipeline_evidence_{ev['turn'].get('turn_id') or 'latest'}.json",
            mime="application/json",
        )
        st.json(ev)

    st.markdown("#### Recent turn evidence table")
    turns = get_recent_turns_list()
    rows = evidence_mod.evidence_rows(turns[-10:])
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.caption("No recent turns in this session.")

    with st.expander("Recent evidence from JSONL logs"):
        log_rows = evidence_mod.evidence_rows(read_recent_turns(10))
        if log_rows:
            st.dataframe(log_rows, hide_index=True, use_container_width=True)
        else:
            st.caption("No JSONL logs yet.")

    report_path = DATA_DIR / "latest_pipeline_evidence.json"
    try:
        report_path.write_text(_safe_json(ev), encoding="utf-8")
        st.caption(f"Latest evidence auto-saved to `{report_path}`")
    except Exception:
        pass
