"""Tests for core.features."""
from __future__ import annotations

import numpy as np
import pytest

from core.audio_capture import SAMPLE_RATE, Utterance
from core.features import _pause_stats, _clarity_score, repetition_score, extract
from core.schemas import AsrResult


def _silence(ms: int) -> np.ndarray:
    return np.zeros(SAMPLE_RATE * ms // 1000, dtype=np.float32)


def _tone(ms: int, freq: float = 200.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(SAMPLE_RATE * ms // 1000) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_pause_stats_pure_speech():
    """All-speech audio should have ~0 pauses."""
    audio = _tone(2000)
    speech_ms, pause_ms, n = _pause_stats(audio)
    assert speech_ms > 1500
    assert n == 0


def test_pause_stats_with_pause():
    """speech-pause-speech should detect exactly 1 pause."""
    audio = np.concatenate([_tone(1000), _silence(800), _tone(1000)])
    speech_ms, pause_ms, n = _pause_stats(audio)
    assert n == 1
    assert pause_ms > 500


def test_clarity_score_bounds():
    """Output must be in [0, 1]."""
    audio = _tone(500)
    assert 0.0 <= _clarity_score(-0.1, audio) <= 1.0
    assert 0.0 <= _clarity_score(-2.0, audio) <= 1.0


def test_clarity_score_monotonic_in_asr_logprob():
    audio = _tone(500)
    low = _clarity_score(-2.0, audio)
    high = _clarity_score(-0.1, audio)
    assert high > low


def test_repetition_score_exact_match():
    assert repetition_score("i am okay", "I am okay") == pytest.approx(1.0, abs=1e-6)


def test_repetition_score_close_match():
    # missing one word should still score reasonably high
    score = repetition_score("main theek hoon", "haan main theek hoon")
    assert score > 0.6


def test_repetition_score_unrelated():
    score = repetition_score("i am okay", "where am i")
    assert score < 0.6


def test_repetition_score_empty_reply():
    assert repetition_score("i am okay", "") == 0.0


def test_extract_full():
    utt = Utterance(audio=_tone(2000), sample_rate=SAMPLE_RATE,
                    start_ms=0, end_ms=2000)
    asr = AsrResult(text="hello there", language="en", avg_logprob=-0.3)
    feats = extract(utt, asr)
    assert feats.speech_duration_ms > 0
    assert feats.speech_rate_cps > 0
    assert 0 <= feats.clarity_score <= 1


def test_phase15_trailing_silence_excluded_from_internal_pause():
    """Leaving recorder open after speaking should not inflate internal pauses."""
    audio = np.concatenate([_tone(1000), _silence(1200)])
    utt = Utterance(audio=audio, sample_rate=SAMPLE_RATE, start_ms=0, end_ms=2200)
    asr = AsrResult(text="hello", language="en", avg_logprob=-0.2)
    feats = extract(utt, asr)
    assert feats.audio_duration_ms >= 2000
    assert feats.trailing_silence_ms >= 1000
    assert feats.internal_pause_total_ms == 0
    assert feats.pause_total_ms == 0


def test_phase15_internal_pause_is_separated_from_leading_trailing():
    audio = np.concatenate([_silence(500), _tone(700), _silence(600), _tone(700), _silence(500)])
    utt = Utterance(audio=audio, sample_rate=SAMPLE_RATE, start_ms=0, end_ms=3000)
    asr = AsrResult(text="hello again", language="en", avg_logprob=-0.2)
    feats = extract(utt, asr, prompt_end_ts_ms=1000)
    assert feats.leading_silence_ms >= 400
    assert feats.trailing_silence_ms >= 400
    assert feats.internal_pause_count == 1
    assert feats.internal_pause_total_ms >= 500
    assert feats.latency_ms >= 1400  # prompt-to-recording start + leading silence
    assert feats.latency_source == "prompt_end_to_first_voiced_estimated"
