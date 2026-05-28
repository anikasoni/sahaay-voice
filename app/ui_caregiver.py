"""Caregiver view — alerts panel and recent transcript."""
from __future__ import annotations

import streamlit as st

from core import alerts as alerts_mod
from core.logger import read_recent_turns


def render_caregiver() -> None:
    st.markdown("### Caregiver alerts")
    st.caption(
        "In a real deployment these would be sent as SMS / push notifications. "
        "For the prototype, they appear here."
    )

    col_clear, _ = st.columns([1, 4])
    with col_clear:
        if st.button("Clear all alerts"):
            alerts_mod.clear_all()
            st.rerun()

    items = alerts_mod.list_alerts(limit=50)
    if not items:
        st.info("No alerts. All is well.")
        return

    for a in items:
        sev_cls = {"urgent": "sv-card-urgent", "warning": "sv-card-warning"}.get(
            a.severity, "sv-card-info"
        )
        transcript_html = ""
        if a.recent_transcript:
            lines = "<br>".join(t.replace("<", "&lt;") for t in a.recent_transcript[-6:])
            transcript_html = (f'<details style="margin-top: 0.6rem;">'
                               f'<summary class="sv-quiet">Recent transcript</summary>'
                               f'<div style="margin-top: 0.4rem; font-size: 0.9rem;">'
                               f'{lines}</div></details>')
        st.markdown(
            f'<div class="sv-card {sev_cls}">'
            f'<div style="display: flex; justify-content: space-between; '
            f'align-items: baseline;">'
            f'<b style="font-size: 1.05rem;">{a.reason}</b>'
            f'<span class="sv-quiet">{a.timestamp}</span></div>'
            f'<div class="sv-quiet" style="margin-top: 0.3rem;">'
            f'severity: <b>{a.severity}</b> · id: {a.alert_id}</div>'
            f'{transcript_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    with st.expander("Last 10 conversation turns"):
        turns = read_recent_turns(10)
        if not turns:
            st.caption("No turns yet.")
        for t in turns:
            asr_text = (t.get("asr") or {}).get("text", "")
            resp = t.get("response_text", "")
            mode = t.get("mode_after", "")
            outcome = t.get("fusion_outcome", "")
            st.markdown(f"**[{mode} / {outcome}]**  \n"
                        f"🗣 {asr_text or '_(silent)_'}  \n"
                        f"🌿 {resp}")
            st.caption(t.get("timestamp", ""))
