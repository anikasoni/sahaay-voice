"""Manager-level scenario tests.

Skip the heavy ASR/NLU model load; we feed synthetic ASR results and assert
that the dialogue manager + fusion produce the right outcomes for the
acceptance scenarios in PRD §10.

These run in a few seconds and don't need a GPU.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from core import baseline as baseline_mod
from core.audio_capture import SAMPLE_RATE, Utterance
from core.dialogue import DialogueManager
from core.schemas import AsrResult, NluResult


# Synthetic 1-second tone — enough for the feature-extractor to do its thing.
def _utt() -> Utterance:
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return Utterance(audio=audio, sample_rate=SAMPLE_RATE,
                     start_ms=0, end_ms=1000)


@pytest.fixture(autouse=True)
def _temp_db(monkeypatch, tmp_path):
    """Isolate baseline DB and logs per test."""
    from core import config as config_mod
    db = tmp_path / "test_baseline.sqlite"
    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(baseline_mod, "DB_PATH", db)
    monkeypatch.setattr(config_mod, "LOG_DIR", logs)
    # logger module captured LOG_DIR at import → patch it too
    import core.logger as logger_mod
    monkeypatch.setattr(logger_mod, "LOG_DIR", logs)
    yield


def _mock_asr(text: str, lang: str = "en"):
    return AsrResult(text=text, language=lang, avg_logprob=-0.2)


def _mock_nlu(intent: str, emotion: str = "calm"):
    return NluResult(intent=intent, intent_confidence=0.9, emotion=emotion)


# ---------------------------------------------------------------------------

def test_clinical_question_routes_to_safe_deflection():
    """PRD §10: 'Am I having a stroke?' → no diagnosis, caregiver alert raised."""
    mgr = DialogueManager()
    with patch("core.dialogue.asr_mod.transcribe", return_value=_mock_asr("am i having a stroke")), \
         patch("core.dialogue.nlu_mod.analyze", return_value=_mock_nlu("clinical_question")), \
         patch("core.dialogue.tts_mod.synthesize", return_value=None):
        result = mgr.handle_utterance(_utt())
    assert result.decision.outcome == "safe_deflection"
    assert "stroke" not in result.response_text.lower()


def test_safety_mode_unclear_reply_escalates():
    """PRD §10 Safety Mode: failed confirmation → caregiver alert."""
    mgr = DialogueManager()
    with patch("core.dialogue.tts_mod.synthesize", return_value=None):
        mgr.enter_safety_mode(lang="en")
    # Now the elder mumbles something unrelated
    with patch("core.dialogue.asr_mod.transcribe",
               return_value=AsrResult(text="ugh huh", language="en", avg_logprob=-2.0)), \
         patch("core.dialogue.nlu_mod.analyze",
               return_value=_mock_nlu("casual_chat")), \
         patch("core.dialogue.tts_mod.synthesize", return_value=None):
        result = mgr.handle_utterance(_utt())
    assert result.decision.outcome == "caregiver_alert"


def test_safety_mode_clear_reply_confirms():
    mgr = DialogueManager()
    with patch("core.dialogue.tts_mod.synthesize", return_value=None):
        mgr.enter_safety_mode(lang="en")
    # Elder says the phrase clearly
    with patch("core.dialogue.asr_mod.transcribe",
               return_value=AsrResult(text="i am okay", language="en", avg_logprob=-0.1)), \
         patch("core.dialogue.nlu_mod.analyze",
               return_value=_mock_nlu("safety_confirmation")), \
         patch("core.dialogue.tts_mod.synthesize", return_value=None):
        result = mgr.handle_utterance(_utt())
    assert result.decision.outcome == "normal_response"
    assert result.mode_after == "idle"


def test_companion_self_harm_routes_to_caregiver():
    """PRD FR-27: self-harm phrasing → gentle response + caregiver alert."""
    mgr = DialogueManager()
    with patch("core.dialogue.asr_mod.transcribe",
               return_value=_mock_asr("i feel so alone i want to die")), \
         patch("core.dialogue.nlu_mod.analyze",
               return_value=_mock_nlu("loneliness_expression", "sad")), \
         patch("core.dialogue.tts_mod.synthesize", return_value=None):
        result = mgr.handle_utterance(_utt())
    assert result.decision.outcome == "caregiver_alert"
    assert "family" in result.response_text.lower()


def test_memory_mode_ack_closes_reminder():
    mgr = DialogueManager()
    with patch("core.dialogue.tts_mod.synthesize", return_value=None):
        mgr.deliver_reminder("med-test", "BP medicine", "en")
    with patch("core.dialogue.asr_mod.transcribe",
               return_value=_mock_asr("done")), \
         patch("core.dialogue.nlu_mod.analyze",
               return_value=_mock_nlu("safety_confirmation")), \
         patch("core.dialogue.tts_mod.synthesize", return_value=None):
        result = mgr.handle_utterance(_utt())
    assert result.mode_after == "idle"
    assert mgr.state.active_reminder_id is None


def test_safety_mode_no_reply_escalates():
    mgr = DialogueManager()
    with patch("core.dialogue.tts_mod.synthesize", return_value=None):
        mgr.enter_safety_mode(lang="en")
        result = mgr.handle_no_reply("timeout")
    assert result.decision.outcome == "caregiver_alert"
    assert result.mode_after == "idle"


def test_memory_mode_unrelated_reply_repeats_not_ack():
    mgr = DialogueManager()
    with patch("core.dialogue.tts_mod.synthesize", return_value=None):
        mgr.deliver_reminder("med-test", "BP medicine", "en")
    with patch("core.dialogue.asr_mod.transcribe",
               return_value=_mock_asr("what is the weather")), \
         patch("core.dialogue.nlu_mod.analyze",
               return_value=_mock_nlu("casual_chat")), \
         patch("core.dialogue.tts_mod.synthesize", return_value=None):
        result = mgr.handle_utterance(_utt())
    assert result.mode_after == "memory"
    assert mgr.state.active_reminder_id == "med-test"
    assert "reminder" in result.response_text.lower()


def test_weak_asr_asks_repeat_and_does_not_pretend_understood():
    """Phase 1.6: weak ordinary ASR should trigger a repeat prompt, not casual_chat."""
    mgr = DialogueManager()
    weak_asr = AsrResult(text="I only today.", language="en", avg_logprob=-0.871)
    with patch("core.dialogue.asr_mod.transcribe", return_value=weak_asr), \
         patch("core.dialogue.nlu_mod.analyze", return_value=NluResult(
             intent="casual_chat", intent_confidence=0.164, emotion="confused", emotion_confidence=0.428
         )), \
         patch("core.dialogue.tts_mod.synthesize", return_value=None):
        result = mgr.handle_utterance(_utt())
    assert result.nlu.intent == "unclear_speech"
    assert result.decision.outcome == "normal_response"
    assert "asking elder to repeat" in result.decision.reason
    assert "repeat" in result.response_text.lower()


def test_weak_asr_does_not_block_critical_clinical_guardrail():
    """Low ASR confidence should not hide clinical/self-harm safety guardrails."""
    mgr = DialogueManager()
    weak_asr = AsrResult(text="am i having a stroke", language="en", avg_logprob=-1.0)
    with patch("core.dialogue.asr_mod.transcribe", return_value=weak_asr), \
         patch("core.dialogue.nlu_mod.analyze", return_value=_mock_nlu("clinical_question")), \
         patch("core.dialogue.tts_mod.synthesize", return_value=None):
        result = mgr.handle_utterance(_utt())
    assert result.decision.outcome == "safe_deflection"
