"""Tests for core.nlu — covers keyword fast paths and slot extraction.

These do NOT load the heavy XLM-R model; they hit the regex code paths.
"""
from __future__ import annotations

from core import nlu
from core.nlu import _keyword_intent, _extract_slots


def test_keyword_emergency_english():
    assert _keyword_intent("help me i have fallen") == "emergency_help"


def test_keyword_emergency_hindi():
    assert _keyword_intent("bachao main gir gaya") == "emergency_help"


def test_keyword_safety_confirmation():
    assert _keyword_intent("I am okay thank you") == "safety_confirmation"
    assert _keyword_intent("haan main theek hoon") == "safety_confirmation"
    assert _keyword_intent("main theek haan") == "safety_confirmation"


def test_keyword_clinical_question():
    assert _keyword_intent("am I having a stroke") == "clinical_question"
    assert _keyword_intent("kya mujhe heart attack hai") == "clinical_question"


def test_keyword_returns_none_for_chitchat():
    assert _keyword_intent("plain words with no domain signal") is None


def test_extract_slots_time_and_medicine():
    slots = _extract_slots("remind me to take my blood pressure at 8 pm")
    assert slots.medicine_name == "blood pressure"
    assert slots.time is not None


def test_extract_slots_person_hindi():
    slots = _extract_slots("beta ko phone karo")
    assert slots.person == "beta"


def test_extract_slots_empty_yields_empty():
    slots = _extract_slots("namaste")
    assert slots.medicine_name is None
    assert slots.time is None
    assert slots.person is None


def test_domain_keywords_cover_acceptance_flows():
    cases = {
        "I want to die": "self_harm_risk",
        "Am I having a stroke": "clinical_question",
        "done": "task_acknowledgement",
        "ho gaya": "task_acknowledgement",
        "I feel lonely today": "loneliness_expression",
        "call my son": "caregiver_request",
        "where am I": "confusion_or_disorientation",
    }
    for text, expected in cases.items():
        assert nlu.analyze(text).intent == expected
