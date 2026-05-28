"""Tests for core.fusion. One scenario per outcome row in TRD §4.7.2."""
from __future__ import annotations

import time

import pytest

from core.fusion import decide
from core.schemas import (
    BaselineDeviation, NluResult, SpeechFeatures, WearableEvent,
)


def _nlu(intent="casual_chat", emotion="calm"):
    return NluResult(intent=intent, intent_confidence=0.8, emotion=emotion)


def _dev(max_z=0.0, exceed=0, sufficient=True):
    return BaselineDeviation(max_z=max_z, exceed_count=exceed,
                             sufficient_history=sufficient)


def _wearable(t="normal", sev="low"):
    return WearableEvent(type=t, severity=sev,
                         timestamp_ms=int(time.time()*1000), age_ms=0)


def test_normal_chat_yields_normal_response():
    d = decide(_nlu(), SpeechFeatures(), _dev(), None)
    assert d.outcome == "normal_response"


def test_clinical_question_safe_deflection():
    d = decide(_nlu(intent="clinical_question"), SpeechFeatures(), _dev(), None)
    assert d.outcome == "safe_deflection"
    assert d.notify_caregiver is True


def test_emergency_help_escalates():
    d = decide(_nlu(intent="emergency_help"), SpeechFeatures(), _dev(), None)
    assert d.outcome == "caregiver_alert"
    assert d.severity == "urgent"


def test_high_fall_when_safety_pending_with_bad_reply():
    # Fall reported AND we're already waiting AND repetition was poor → alert
    feats = SpeechFeatures(repetition_score=0.1, clarity_score=0.2)
    d = decide(
        _nlu(intent="safety_confirmation"),
        feats,
        _dev(),
        _wearable("fall_detected", "high"),
        safety_pending=True,
        is_safety_reply=True,
    )
    assert d.outcome == "caregiver_alert"
    assert "fall" in d.reason.lower()


def test_high_fall_without_pending_enters_safety():
    """Fall just dropped — decide returns normal_response with a 'fall' reason;
    the dialogue manager interprets that as 'enter safety mode'."""
    d = decide(_nlu(), SpeechFeatures(), _dev(),
               _wearable("fall_detected", "high"))
    assert d.outcome == "normal_response"
    assert "fall" in d.reason.lower()


def test_safety_pending_unclear_reply_alerts():
    feats = SpeechFeatures(repetition_score=0.1, clarity_score=0.1)
    d = decide(
        _nlu(intent="safety_confirmation"),
        feats, _dev(), None,
        safety_pending=True, is_safety_reply=True,
    )
    assert d.outcome == "caregiver_alert"


def test_safety_pending_good_reply_normal():
    feats = SpeechFeatures(repetition_score=0.9, clarity_score=0.9)
    d = decide(
        _nlu(intent="safety_confirmation"),
        feats, _dev(), None,
        safety_pending=True, is_safety_reply=True,
    )
    assert d.outcome == "normal_response"


def test_reminder_unack_escalates_after_max():
    d = decide(_nlu(), SpeechFeatures(), _dev(), None,
               reminder_unack_count=3)  # max_repeats_default(2) + 1
    assert d.outcome == "caregiver_alert"
    assert "reminder" in d.reason.lower()


def test_strong_deviation_yields_soft_checkin():
    d = decide(_nlu(), SpeechFeatures(),
               _dev(max_z=3.0, exceed=3, sufficient=True), None)
    assert d.outcome == "soft_check_in"


def test_deviation_without_history_does_nothing():
    d = decide(_nlu(), SpeechFeatures(),
               _dev(max_z=10.0, exceed=5, sufficient=False), None)
    assert d.outcome == "normal_response"


def test_self_harm_escalates():
    d = decide(_nlu(intent="self_harm_risk", emotion="sad"), SpeechFeatures(), _dev(), None)
    assert d.outcome == "caregiver_alert"
    assert d.severity == "urgent"


def test_reminder_no_reply_repeats_before_escalation():
    d = decide(_nlu(intent="no_reply"), SpeechFeatures(), _dev(), None,
               reminder_active=True, reminder_unack_count=0, no_reply=True)
    assert d.outcome == "reminder_repeat"


def test_inactivity_triggers_soft_checkin():
    d = decide(_nlu(), SpeechFeatures(), _dev(), _wearable("inactivity_long", "medium"))
    assert d.outcome == "soft_check_in"
