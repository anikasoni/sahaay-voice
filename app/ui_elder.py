"""Elder view — voice-first, large mic, current mode, last response.

Uses Streamlit's built-in `st.audio_input` widget (push-to-talk style) since
running an always-on mic loop inside Streamlit is brittle. The mic-loop path
exists in core.audio_capture for a future kiosk wrapper.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import streamlit as st

from app.state import get_manager, get_last_turn, set_last_turn, push_turn
from core.audio_capture import utterance_from_wav
from core.config import AUDIO_DIR
from core import evidence as evidence_mod


_MODE_LABEL = {
    "idle": "Listening", "safety": "Safety Mode",
    "companion": "Companion Mode", "memory": "Memory Mode",
}


def _save_uploaded(audio_bytes, suffix: str = ".wav") -> Path:
    fname = AUDIO_DIR / f"in_{int(time.time()*1000)}{suffix}"
    with open(fname, "wb") as f:
        f.write(audio_bytes)
    return fname


def render_elder() -> None:
    mgr = get_manager()
    mode = mgr.state.mode
    lang = mgr.state.default_lang

    # Mode pill
    label = _MODE_LABEL.get(mode, "Listening")
    pill_class = f"sv-mode-pill sv-mode-{mode}"
    st.markdown(f'<div style="margin: 0.5rem 0;">'
                f'<span class="{pill_class}">{label}</span>'
                f' <span class="sv-quiet">· language: {lang}</span>'
                f'</div>', unsafe_allow_html=True)

    st.markdown("### Speak when you are ready")
    st.caption("Tap the microphone, say something, then stop the recording. "
               "English, Hindi, or Punjabi — whatever feels natural.")

    audio_value = st.audio_input("🎙️ Tap to record")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("End recording / send", type="primary",
                     disabled=audio_value is None, use_container_width=True):
            if audio_value is not None:
                _process(audio_value.read())
    with col_b:
        if st.button("Reset conversation", use_container_width=True):
            mgr.reset()
            set_last_turn(None)
            st.session_state.recent_turns = []
            st.rerun()

    st.divider()

    # Last response panel — big text, audio player.
    turn = get_last_turn()
    if turn:
        st.markdown("### Sahaay said")
        st.markdown(
            f'<div class="sv-card"><div style="font-size: 1.4rem; line-height: 1.5;">'
            f'{turn.response_text}'
            f'</div><div class="sv-quiet" style="margin-top: 0.6rem;">'
            f'language: {turn.response_lang} · '
            f'mode: {turn.mode_after} · '
            f'reason: {turn.decision.reason}'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        if turn.response_audio_path and Path(turn.response_audio_path).exists():
            st.audio(str(turn.response_audio_path), autoplay=True)

        with st.expander("What you said"):
            st.write(turn.asr.text or "_(silent)_")
            st.caption(f"detected language: {turn.asr.language} · "
                       f"clarity: {turn.features.clarity_score:.2f} · "
                       f"speech rate: {turn.features.speech_rate_cps:.1f} cps")

        with st.expander("Developer view: AI evidence for this turn"):
            ev = evidence_mod.turn_to_evidence(turn)
            st.json({
                "asr": ev["asr"],
                "speech_features": ev["speech_features"],
                "nlu": ev["nlu"],
                "baseline": ev["baseline"],
                "fusion": ev["fusion"],
            })
    else:
        st.info("No conversation yet. Record something to begin.")


def _process(audio_bytes: bytes) -> None:
    """Run one full pipeline turn on user audio.

    Stability note:
    Streamlit already reruns once when the Send button is clicked. Calling
    st.rerun() again immediately after ASR/NLU/TTS can cause the browser to
    reconnect on some Windows setups. We therefore update session state and
    let the current run render the result normally.
    """
    mgr = get_manager()
    path = _save_uploaded(audio_bytes)
    try:
        utt = utterance_from_wav(path)
        with st.spinner("Sahaay is listening…"):
            turn = mgr.handle_utterance(utt, audio_path=path)
    except Exception as e:
        st.error(f"Pipeline failed: {type(e).__name__}: {e}")
        return

    set_last_turn(turn)
    push_turn(turn)
