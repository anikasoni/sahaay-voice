"""Streamlit session-state helpers — keeps singletons across reruns."""
from __future__ import annotations

import streamlit as st

from core.dialogue import DialogueManager


def get_manager() -> DialogueManager:
    if "manager" not in st.session_state:
        st.session_state.manager = DialogueManager()
    return st.session_state.manager


def get_last_turn():
    return st.session_state.get("last_turn")


def set_last_turn(t) -> None:
    st.session_state.last_turn = t


def get_recent_turns_list() -> list:
    return st.session_state.setdefault("recent_turns", [])


def push_turn(t) -> None:
    lst = get_recent_turns_list()
    lst.append(t)
    if len(lst) > 25:
        del lst[: len(lst) - 25]
