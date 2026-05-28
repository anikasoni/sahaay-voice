"""AI pipeline evidence helpers.

This module makes the PRD/TRD pipeline auditable. It converts a TurnResult
(ASR -> speech features -> NLU -> baseline -> fusion -> response) into a flat,
human-readable evidence record for the Operator UI and JSON reports.

It intentionally has no Streamlit dependency so tests/scripts can reuse it.
"""
from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping


def _round(x: Any, ndigits: int = 3) -> Any:
    try:
        return round(float(x), ndigits)
    except Exception:
        return x


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {}


def asr_confidence_from_logprob(avg_logprob: float | int | None) -> float:
    """Approximate ASR confidence in 0..1 from Whisper segment avg_logprob."""
    try:
        lp = float(avg_logprob or 0.0)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, math.exp(lp)))


def feature_status(value: float, direction: str = "higher_better") -> str:
    """Coarse display-only status for a scalar evidence value."""
    try:
        v = float(value)
    except Exception:
        return "unknown"
    if direction == "higher_better":
        if v >= 0.75:
            return "strong"
        if v >= 0.45:
            return "usable"
        return "weak"
    if direction == "lower_better":
        if v <= 0.25:
            return "strong"
        if v <= 0.55:
            return "usable"
        return "weak"
    return "observed"


def turn_to_evidence(turn: Any) -> dict[str, Any]:
    """Return nested evidence for a full pipeline turn.

    Works for both real TurnResult objects and dicts loaded from JSONL logs.
    """
    if isinstance(turn, Mapping):
        mode_before = turn.get("mode_before") or turn.get("mode")
        mode_after = turn.get("mode_after")
        asr = _as_dict(turn.get("asr"))
        features = _as_dict(turn.get("features"))
        nlu = _as_dict(turn.get("nlu"))
        deviation = _as_dict(turn.get("baseline_deviation") or turn.get("deviation"))
        decision = {
            "outcome": turn.get("fusion_outcome") or _as_dict(turn.get("decision")).get("outcome"),
            "reason": turn.get("fusion_reason") or _as_dict(turn.get("decision")).get("reason"),
            "severity": _as_dict(turn.get("decision")).get("severity", "info"),
        }
        response_text = turn.get("response_text", "")
        response_lang = turn.get("response_language") or turn.get("response_lang")
        audio_path = turn.get("audio_path")
        timestamp = turn.get("timestamp")
        turn_id = turn.get("turn_id")
    else:
        mode_before = getattr(turn, "mode_before", None)
        mode_after = getattr(turn, "mode_after", None)
        asr = _as_dict(getattr(turn, "asr", None))
        features = _as_dict(getattr(turn, "features", None))
        nlu = _as_dict(getattr(turn, "nlu", None))
        deviation = _as_dict(getattr(turn, "deviation", None))
        decision = _as_dict(getattr(turn, "decision", None))
        response_text = getattr(turn, "response_text", "")
        response_lang = getattr(turn, "response_lang", None)
        audio_path = str(getattr(turn, "audio_path", "") or "")
        timestamp = getattr(turn, "timestamp", None)
        turn_id = getattr(turn, "turn_id", None)

    segments = asr.get("segments") or []
    avg_logprob = asr.get("avg_logprob", 0.0)
    asr_conf = asr_confidence_from_logprob(avg_logprob)
    speech_ms = float(features.get("speech_duration_ms") or features.get("voiced_duration_ms") or 0.0)
    pause_ms = float(features.get("pause_total_ms") or features.get("internal_pause_total_ms") or 0.0)
    pause_count = int(features.get("pause_count") or features.get("internal_pause_count") or 0)
    audio_ms = float(features.get("audio_duration_ms") or (speech_ms + pause_ms) or 0.0)
    voiced_ms = float(features.get("voiced_duration_ms") or speech_ms)
    unvoiced_total_ms = float(features.get("unvoiced_total_ms") or max(0.0, audio_ms - voiced_ms))
    leading_ms = float(features.get("leading_silence_ms") or 0.0)
    trailing_ms = float(features.get("trailing_silence_ms") or 0.0)
    internal_pause_ms = float(features.get("internal_pause_total_ms") or pause_ms)
    internal_pause_count = int(features.get("internal_pause_count") or pause_count)
    voice_activity_ratio = float(features.get("voice_activity_ratio") or (voiced_ms / audio_ms if audio_ms > 0 else 0.0))
    mean_pause_ms = float(features.get("mean_internal_pause_ms") or (internal_pause_ms / internal_pause_count if internal_pause_count > 0 else 0.0))

    slots = _as_dict(nlu.get("slots"))
    per_z = deviation.get("per_feature_z") or {}
    if not isinstance(per_z, Mapping):
        per_z = {}

    return {
        "turn": {
            "turn_id": turn_id,
            "timestamp": timestamp,
            "mode_before": mode_before,
            "mode_after": mode_after,
            "audio_path": audio_path,
        },
        "asr": {
            "transcript": asr.get("text", ""),
            "detected_language": asr.get("language"),
            "avg_logprob": _round(avg_logprob),
            "approx_confidence": _round(asr_conf),
            "confidence_status": feature_status(asr_conf),
            "segment_count": len(segments),
            "segments": segments,
        },
        "speech_features": {
            "latency_ms": int(features.get("latency_ms") or 0),
            "latency_source": features.get("latency_source") or "unknown",
            "latency_status": features.get("latency_status") or "unknown",
            "audio_duration_ms": int(audio_ms),
            "voiced_duration_ms": int(voiced_ms),
            "speech_duration_ms": int(speech_ms),
            "unvoiced_total_ms": int(unvoiced_total_ms),
            "leading_silence_ms": int(leading_ms),
            "trailing_silence_ms": int(trailing_ms),
            "internal_pause_total_ms": int(internal_pause_ms),
            "internal_pause_count": internal_pause_count,
            "pause_total_ms": int(pause_ms),
            "pause_count": pause_count,
            "mean_internal_pause_ms": _round(mean_pause_ms),
            "mean_pause_ms": _round(mean_pause_ms),
            "voice_activity_ratio": _round(voice_activity_ratio),
            "pause_definition": features.get("pause_definition") or "internal_unvoiced_between_first_and_last_voiced",
            "speech_rate_cps": _round(features.get("speech_rate_cps") or 0.0),
            "clarity_score": _round(features.get("clarity_score") or 0.0),
            "clarity_status": feature_status(features.get("clarity_score") or 0.0),
            "repetition_score": _round(features.get("repetition_score") or 0.0),
            "asr_avg_logprob": _round(features.get("asr_avg_logprob") or avg_logprob),
        },
        "nlu": {
            "intent": nlu.get("intent"),
            "intent_confidence": _round(nlu.get("intent_confidence") or 0.0),
            "emotion": nlu.get("emotion"),
            "emotion_confidence": _round(nlu.get("emotion_confidence") or 0.0),
            "slots": slots,
        },
        "baseline": {
            "max_z": _round(deviation.get("max_z") or 0.0),
            "exceed_count": int(deviation.get("exceed_count") or 0),
            "sufficient_history": bool(deviation.get("sufficient_history", False)),
            "per_feature_z": {str(k): _round(v) for k, v in per_z.items()},
        },
        "fusion": {
            "outcome": decision.get("outcome"),
            "reason": decision.get("reason"),
            "severity": decision.get("severity", "info"),
            "notify_caregiver": bool(decision.get("notify_caregiver", False)),
        },
        "response": {
            "text": response_text,
            "language": response_lang,
        },
    }


def flatten_evidence(ev: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten nested evidence into one row for dataframe/CSV display."""
    return {
        "time": (ev.get("turn", {}) or {}).get("timestamp"),
        "turn_id": (ev.get("turn", {}) or {}).get("turn_id"),
        "mode": f"{(ev.get('turn', {}) or {}).get('mode_before')}→{(ev.get('turn', {}) or {}).get('mode_after')}",
        "transcript": (ev.get("asr", {}) or {}).get("transcript"),
        "lang": (ev.get("asr", {}) or {}).get("detected_language"),
        "asr_conf": (ev.get("asr", {}) or {}).get("approx_confidence"),
        "intent": (ev.get("nlu", {}) or {}).get("intent"),
        "intent_conf": (ev.get("nlu", {}) or {}).get("intent_confidence"),
        "emotion": (ev.get("nlu", {}) or {}).get("emotion"),
        "clarity": (ev.get("speech_features", {}) or {}).get("clarity_score"),
        "latency_ms": (ev.get("speech_features", {}) or {}).get("latency_ms"),
        "audio_ms": (ev.get("speech_features", {}) or {}).get("audio_duration_ms"),
        "voiced_ms": (ev.get("speech_features", {}) or {}).get("voiced_duration_ms"),
        "speech_rate_cps": (ev.get("speech_features", {}) or {}).get("speech_rate_cps"),
        "internal_pause_ms": (ev.get("speech_features", {}) or {}).get("internal_pause_total_ms"),
        "trailing_silence_ms": (ev.get("speech_features", {}) or {}).get("trailing_silence_ms"),
        "repetition": (ev.get("speech_features", {}) or {}).get("repetition_score"),
        "max_z": (ev.get("baseline", {}) or {}).get("max_z"),
        "exceed_count": (ev.get("baseline", {}) or {}).get("exceed_count"),
        "outcome": (ev.get("fusion", {}) or {}).get("outcome"),
        "severity": (ev.get("fusion", {}) or {}).get("severity"),
        "reason": (ev.get("fusion", {}) or {}).get("reason"),
    }


def evidence_rows(turns: list[Any]) -> list[dict[str, Any]]:
    return [flatten_evidence(turn_to_evidence(t)) for t in turns]
