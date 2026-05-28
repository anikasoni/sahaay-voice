"""Speech feature extraction.

Per TRD §4.3 — produces a SpeechFeatures record from an Utterance + ASR result.
None of these features constitute a medical finding (FR-10 in PRD).

Phase 1.5 makes the feature layer auditable:
- separates total audio duration from voiced duration;
- separates leading/trailing silence from internal pauses;
- estimates prompt-response latency in Streamlit push-to-talk mode;
- keeps backward-compatible baseline fields while exposing definitions.
"""
from __future__ import annotations

import math
import re
from typing import Optional

import numpy as np

from core.audio_capture import SAMPLE_RATE, Utterance
from core.schemas import AsrResult, SpeechFeatures


# --- frame / voice activity analysis ---------------------------------------

def _frame_energies(audio: np.ndarray, frame_ms: int = 20) -> np.ndarray:
    """RMS energy per fixed frame."""
    n = SAMPLE_RATE * frame_ms // 1000
    if len(audio) == 0:
        return np.array([], dtype=np.float32)
    if len(audio) < n:
        return np.array([np.sqrt(np.mean(audio ** 2) + 1e-12)])
    nframes = len(audio) // n
    framed = audio[: nframes * n].reshape(nframes, n)
    return np.sqrt(np.mean(framed ** 2, axis=1) + 1e-12)


def _voiced_mask(audio: np.ndarray, frame_ms: int = 20) -> np.ndarray:
    """Adaptive energy-based VAD mask used for explainable feature extraction.

    This is not the capture-time VAD. It is a lightweight post-hoc analyzer that
    lets us expose speech/pause features even for uploaded/recorded clips.
    """
    energies = _frame_energies(audio, frame_ms)
    if len(energies) == 0:
        return np.array([], dtype=bool)

    # Truly silent / near-silent recordings.
    if float(np.max(energies)) < 1e-4:
        return np.zeros_like(energies, dtype=bool)

    noise = float(np.percentile(energies, 20))
    speech_ref = float(np.percentile(energies, 90))
    # If the file is all one steady tone/noise, still set a sensible threshold.
    if speech_ref <= noise + 1e-6:
        thr = 0.35 * speech_ref
    else:
        thr = noise + 0.18 * (speech_ref - noise)
    thr = max(1e-5, thr)
    voiced = energies > thr

    # Remove tiny one-frame islands to avoid over-counting clicks/noise.
    if len(voiced) >= 3:
        cleaned = voiced.copy()
        for i in range(1, len(voiced) - 1):
            if voiced[i] and not voiced[i - 1] and not voiced[i + 1]:
                cleaned[i] = False
        voiced = cleaned
    return voiced


def _run_lengths(mask: np.ndarray) -> list[tuple[bool, int, int]]:
    """Return contiguous runs as (value, start_index, length)."""
    if len(mask) == 0:
        return []
    out: list[tuple[bool, int, int]] = []
    start = 0
    val = bool(mask[0])
    for i in range(1, len(mask)):
        if bool(mask[i]) != val:
            out.append((val, start, i - start))
            start = i
            val = bool(mask[i])
    out.append((val, start, len(mask) - start))
    return out


def _speech_activity_stats(audio: np.ndarray, frame_ms: int = 20) -> dict[str, int | float | str]:
    """Return auditable activity/pause stats.

    Definitions:
    - audio_duration_ms: full recorder duration.
    - voiced_duration_ms: frames classified as speech.
    - unvoiced_total_ms: all non-voiced frames, including leading/trailing.
    - leading/trailing_silence_ms: silence before first / after last voiced frame.
    - internal_pause_total_ms: unvoiced frames between first and last voiced frames.
      This is what feeds the baseline as pause_total_ms.
    - internal_pause_count: contiguous internal pauses >= 200 ms.
    """
    audio_duration_ms = int(round(len(audio) / SAMPLE_RATE * 1000)) if len(audio) else 0
    voiced = _voiced_mask(audio, frame_ms)
    nframes = len(voiced)
    if nframes == 0:
        return {
            "audio_duration_ms": audio_duration_ms,
            "voiced_duration_ms": 0,
            "unvoiced_total_ms": audio_duration_ms,
            "leading_silence_ms": audio_duration_ms,
            "trailing_silence_ms": 0,
            "internal_pause_total_ms": 0,
            "internal_pause_count": 0,
            "mean_internal_pause_ms": 0.0,
            "voice_activity_ratio": 0.0,
            "pause_definition": "internal_unvoiced_between_first_and_last_voiced",
        }

    voiced_frames = int(voiced.sum())
    voiced_duration_ms = voiced_frames * frame_ms
    unvoiced_total_ms = max(0, nframes * frame_ms - voiced_duration_ms)

    if voiced_frames == 0:
        return {
            "audio_duration_ms": audio_duration_ms,
            "voiced_duration_ms": 0,
            "unvoiced_total_ms": max(audio_duration_ms, unvoiced_total_ms),
            "leading_silence_ms": audio_duration_ms,
            "trailing_silence_ms": 0,
            "internal_pause_total_ms": 0,
            "internal_pause_count": 0,
            "mean_internal_pause_ms": 0.0,
            "voice_activity_ratio": 0.0,
            "pause_definition": "internal_unvoiced_between_first_and_last_voiced",
        }

    first_voice = int(np.argmax(voiced))
    last_voice = int(len(voiced) - 1 - np.argmax(voiced[::-1]))
    leading_ms = first_voice * frame_ms
    trailing_ms = max(0, (nframes - 1 - last_voice) * frame_ms)

    internal = voiced[first_voice:last_voice + 1]
    internal_unvoiced_frames = int((~internal).sum())
    internal_pause_total_ms = internal_unvoiced_frames * frame_ms

    min_pause_frames = max(1, 200 // frame_ms)
    internal_pause_count = 0
    for val, _start, length in _run_lengths(~internal):
        if val and length >= min_pause_frames:
            internal_pause_count += 1

    mean_internal_pause_ms = (
        internal_pause_total_ms / internal_pause_count if internal_pause_count > 0 else 0.0
    )
    voice_activity_ratio = voiced_duration_ms / max(1, nframes * frame_ms)

    return {
        "audio_duration_ms": audio_duration_ms,
        "voiced_duration_ms": int(voiced_duration_ms),
        "unvoiced_total_ms": int(max(0, unvoiced_total_ms)),
        "leading_silence_ms": int(leading_ms),
        "trailing_silence_ms": int(trailing_ms),
        "internal_pause_total_ms": int(internal_pause_total_ms),
        "internal_pause_count": int(internal_pause_count),
        "mean_internal_pause_ms": float(round(mean_internal_pause_ms, 3)),
        "voice_activity_ratio": float(round(voice_activity_ratio, 3)),
        "pause_definition": "internal_unvoiced_between_first_and_last_voiced",
    }


def _pause_stats(audio: np.ndarray, frame_ms: int = 20) -> tuple[int, int, int]:
    """Backward-compatible helper returning (voiced_ms, internal_pause_ms, count)."""
    stats = _speech_activity_stats(audio, frame_ms)
    return (
        int(stats["voiced_duration_ms"]),
        int(stats["internal_pause_total_ms"]),
        int(stats["internal_pause_count"]),
    )


# --- clarity ----------------------------------------------------------------

def _clarity_score(asr_avg_logprob: float, audio: np.ndarray) -> float:
    """Map ASR confidence + audio SNR proxy into a 0..1 clarity score.

    avg_logprob is typically in [-1.0, 0.0]; closer to 0 = more confident.
    """
    asr_part = 1.0 / (1.0 + math.exp(-(asr_avg_logprob + 0.6) * 5))

    energies = _frame_energies(audio, frame_ms=20)
    if len(energies) < 3 or float(np.max(energies)) < 1e-5:
        snr_part = 0.0
    else:
        sorted_e = np.sort(energies)
        n = len(sorted_e)
        low = float(np.mean(sorted_e[: max(1, n // 3)])) + 1e-9
        high = float(np.mean(sorted_e[-max(1, n // 3):])) + 1e-9
        snr_db = 20 * math.log10(high / low)
        snr_part = max(0.0, min(1.0, snr_db / 30.0))

    return float(max(0.0, min(1.0, 0.7 * asr_part + 0.3 * snr_part)))


# --- repetition (Safety Mode only) -----------------------------------------

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _normalize(s: str) -> str:
    return " ".join(_WORD_RE.findall(s.lower()))


def _levenshtein_ratio(a: str, b: str) -> float:
    a, b = _normalize(a), _normalize(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    dist = prev[-1]
    return 1 - dist / max(len(a), len(b))


def repetition_score(target: str, said: str) -> float:
    """Phonetic-leaning + edit-distance similarity, used in Safety Mode."""
    if not said.strip():
        return 0.0
    base = _levenshtein_ratio(target, said)
    target_tokens = set(_normalize(target).split())
    said_tokens = set(_normalize(said).split())
    overlap = len(target_tokens & said_tokens) / max(1, len(target_tokens))
    return float(max(0.0, min(1.0, base + 0.2 * overlap)))


# --- main entry point -------------------------------------------------------

def _latency_status(latency_ms: int) -> str:
    if latency_ms <= 0:
        return "not_measured"
    if latency_ms <= 3000:
        return "prompt_reply_fast"
    if latency_ms <= 8000:
        return "prompt_reply_delayed"
    return "prompt_reply_slow"


def extract(
    utt: Utterance,
    asr: AsrResult,
    prompt_end_ts_ms: Optional[int] = None,
    expected_repetition: Optional[str] = None,
) -> SpeechFeatures:
    """Build a SpeechFeatures record from an utterance and its ASR result.

    Args:
        utt: the captured Utterance.
        asr: ASR output.
        prompt_end_ts_ms: in the Streamlit UI this should be the estimated delay
            from prompt end to recording start. We add leading_silence_ms to
            estimate delay to first voiced frame. Pass None when no system prompt
            is relevant.
        expected_repetition: target phrase, only set in Safety Mode.
    """
    stats = _speech_activity_stats(utt.audio)
    voiced_ms = int(stats["voiced_duration_ms"])
    internal_pause_ms = int(stats["internal_pause_total_ms"])
    internal_pause_n = int(stats["internal_pause_count"])
    leading_ms = int(stats["leading_silence_ms"])

    if prompt_end_ts_ms is not None:
        latency_ms = max(0, int(prompt_end_ts_ms) + leading_ms)
        latency_source = "prompt_end_to_first_voiced_estimated"
    else:
        latency_ms = 0
        latency_source = "not_measured_no_active_prompt"

    text_len = len(asr.text.strip())
    rate = (text_len / (voiced_ms / 1000.0)) if voiced_ms > 0 else 0.0

    clarity = _clarity_score(asr.avg_logprob, utt.audio)
    rep = repetition_score(expected_repetition, asr.text) if expected_repetition else 0.0

    return SpeechFeatures(
        latency_ms=int(latency_ms),
        latency_source=latency_source,
        latency_status=_latency_status(latency_ms),
        speech_duration_ms=int(voiced_ms),
        pause_total_ms=int(internal_pause_ms),
        pause_count=int(internal_pause_n),
        audio_duration_ms=int(stats["audio_duration_ms"]),
        voiced_duration_ms=int(voiced_ms),
        unvoiced_total_ms=int(stats["unvoiced_total_ms"]),
        leading_silence_ms=int(stats["leading_silence_ms"]),
        trailing_silence_ms=int(stats["trailing_silence_ms"]),
        internal_pause_total_ms=int(internal_pause_ms),
        internal_pause_count=int(internal_pause_n),
        mean_internal_pause_ms=float(stats["mean_internal_pause_ms"]),
        voice_activity_ratio=float(stats["voice_activity_ratio"]),
        pause_definition=str(stats["pause_definition"]),
        speech_rate_cps=float(round(rate, 3)),
        clarity_score=float(round(clarity, 3)),
        repetition_score=float(round(rep, 3)),
        asr_avg_logprob=float(round(asr.avg_logprob, 3)),
    )
