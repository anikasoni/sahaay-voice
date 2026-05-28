"""Streamlit entry point for Sahaay Voice.

Run with:
    streamlit run app/main.py

Three views in tabs (per TRD §4.10):
    Elder      — large mic, current mode, last response.
    Caregiver  — alerts panel with timestamps and recent transcript.
    Operator   — wearable triggers, baseline reset, reminder triggers, logs.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `core` importable when launched via `streamlit run app/main.py`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.ui_elder import render_elder
from app.ui_caregiver import render_caregiver
from app.ui_operator import render_operator
from app.state import get_manager

# ----------------------- page setup ----------------------------------------

st.set_page_config(
    page_title="Sahaay Voice",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom styling — warm, age-appropriate, never alarming.
st.markdown("""
<style>
  :root {
    --sv-bg: #faf6f0;
    --sv-ink: #2d2a26;
    --sv-accent: #c2410c;   /* warm terracotta */
    --sv-soft: #f1e8d8;
    --sv-good: #4d7c0f;
    --sv-warn: #b45309;
    --sv-urgent: #b91c1c;
    --sv-muted: #78716c;
  }
  .stApp { background: var(--sv-bg); color: var(--sv-ink); }
  .stApp, .stApp * { color: var(--sv-ink); }
  .stTabs [data-baseweb="tab"] p { color: var(--sv-ink) !important; }
  .stMarkdown, .stCaption, label, p, span, div { color: var(--sv-ink); }
  h1, h2, h3 { color: var(--sv-ink); font-family: 'Lora', Georgia, serif; }
  .sv-mode-pill {
    display: inline-block; padding: 6px 16px; border-radius: 999px;
    font-weight: 600; letter-spacing: 0.5px; font-size: 0.85rem;
    text-transform: uppercase;
  }
  .sv-mode-idle      { background: #e7e5e4; color: #57534e; }
  .sv-mode-safety    { background: #fee2e2; color: #b91c1c; }
  .sv-mode-companion { background: #dcfce7; color: #166534; }
  .sv-mode-memory    { background: #dbeafe; color: #1e40af; }
  .sv-card {
    background: white; padding: 1.2rem 1.4rem; border-radius: 12px;
    border-left: 4px solid var(--sv-accent); margin: 0.6rem 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }
  .sv-card-urgent  { border-left-color: var(--sv-urgent); background: #fef2f2; }
  .sv-card-warning { border-left-color: var(--sv-warn); background: #fff7ed; }
  .sv-card-info    { border-left-color: var(--sv-muted); background: #fafaf9; }
  .sv-quiet { color: var(--sv-muted); font-size: 0.85rem; }
  .sv-banner {
    background: var(--sv-soft); padding: 0.6rem 1rem; border-radius: 8px;
    font-size: 0.85rem; color: var(--sv-ink);
  }
  .stButton>button {
    border-radius: 10px; padding: 0.5rem 1.2rem; font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)

# Ensure manager exists in session
get_manager()

# ----------------------- header --------------------------------------------

col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown("# 🌿")
with col_title:
    st.markdown("# Sahaay Voice")
    st.markdown(
        '<div class="sv-banner">'
        'A multilingual voice companion for elders. Not a medical device. '
        'Sahaay Voice does <b>not</b> diagnose any condition; it supports safety, '
        'companionship, and gentle reminders, and notifies a caregiver when needed.'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown("")

# ----------------------- tabs ----------------------------------------------

tab_elder, tab_caregiver, tab_operator = st.tabs(
    ["👵 Elder", "👨‍👩‍👧 Caregiver", "🛠 Operator"]
)
with tab_elder:
    render_elder()
with tab_caregiver:
    render_caregiver()
with tab_operator:
    render_operator()
