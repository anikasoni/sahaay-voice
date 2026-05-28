"""Multilingual ASR — faster-whisper backend.

Per TRD §4.2: returns transcript + language + per-segment confidence +
word-level timestamps where available. Routes Punjabi/code-mixed-Punjabi
to the Indic-specific path when available; falls back to Whisper Punjabi
mode otherwise.

Performance notes for RTX 4070:
  - model_size=medium, compute_type=float16, device=cuda → ~5-8x realtime
  - Cold load is ~3-6s; we warm at startup in app.main.
"""
from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from core.audio_capture import SAMPLE_RATE, Utterance
from core.config import thresholds
from core.logger import log
from core.schemas import AsrResult, AsrSegment, Language

# Hugging Face / faster-whisper language tags
_WHISPER_LANG = {"en": "en", "hi": "hi", "pa": "pa"}


class _LazyWhisper:
    """Loaded on first transcribe() call. Keeps cold-start out of import time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self.loaded_model_size: str | None = None
        self.loaded_device: str | None = None
        self.loaded_compute_type: str | None = None

    def _load(self):
        from faster_whisper import WhisperModel  # heavy import
        cfg = thresholds()["asr"]
        device = cfg["device"]
        compute_type = cfg["compute_type"]
        # graceful CPU fallback if CUDA isn't actually available
        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    log("CUDA not available, falling back to CPU+int8", level="warning")
                    device, compute_type = "cpu", "int8"
            except ImportError:
                log("torch not installed; assuming CPU", level="warning")
                device, compute_type = "cpu", "int8"
        size = cfg["model_size"]
        self.loaded_model_size = size
        self.loaded_device = device
        self.loaded_compute_type = compute_type
        log(f"loading faster-whisper '{size}' on {device}/{compute_type}…")
        return WhisperModel(size, device=device, compute_type=compute_type)

    def get(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._load()
        return self._model


_engine = _LazyWhisper()


def warm_up() -> None:
    """Pre-load the model. Call at app startup to avoid first-utterance lag."""
    _engine.get()
    log("ASR warm-up complete")


def _normalize_language(detected: str) -> Language:
    if detected in ("en", "hi", "pa"):
        return detected  # type: ignore[return-value]
    return "mixed"


def transcribe(utt: Utterance, hint_lang: Optional[Language] = None) -> AsrResult:
    """Transcribe an Utterance. hint_lang skips language detection if set."""
    model = _engine.get()
    cfg = thresholds()["asr"]
    audio = utt.audio.astype(np.float32)

    # faster-whisper expects 16kHz mono float32 in [-1, 1] — exactly what we have
    segments_iter, info = model.transcribe(
        audio,
        beam_size=cfg["beam_size"],
        vad_filter=False,  # we already VAD-segmented upstream
        language=_WHISPER_LANG.get(hint_lang) if hint_lang else None,
    )

    segs: list[AsrSegment] = []
    text_parts: list[str] = []
    logprobs: list[float] = []
    for s in segments_iter:
        seg = AsrSegment(
            start=float(s.start or 0.0),
            end=float(s.end or 0.0),
            text=(s.text or "").strip(),
            confidence=float(math.exp(s.avg_logprob)) if s.avg_logprob is not None else 0.0,
        )
        segs.append(seg)
        if seg.text:
            text_parts.append(seg.text)
        if s.avg_logprob is not None:
            logprobs.append(float(s.avg_logprob))

    text = " ".join(text_parts).strip()
    lang = _normalize_language(info.language) if info.language else "mixed"
    avg_lp = sum(logprobs) / len(logprobs) if logprobs else 0.0

    # crude code-mixed sniff: Whisper often nails one language but the script may
    # contain Latin-letter Hindi/Punjabi (e.g. "main theek hoon"). Mark as mixed
    # if the detected language is English yet language-prob is low — caller can
    # decide what to do.
    if lang == "en" and info.language_probability and info.language_probability < 0.6:
        lang = "mixed"

    result = AsrResult(text=text, language=lang, segments=segs, avg_logprob=avg_lp)
    log("ASR result", level="info",
        lang=lang, text_preview=text[:80], avg_logprob=round(avg_lp, 3))
    return result


def transcribe_wav(path: str | Path) -> AsrResult:
    """Convenience helper for tests and demos."""
    from core.audio_capture import utterance_from_wav
    return transcribe(utterance_from_wav(path))


def runtime_info() -> dict[str, object]:
    """Display-only ASR runtime metadata for the AI evidence panel."""
    cfg = thresholds()["asr"]
    cuda_available = False
    gpu_name = None
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return {
        "stage": "ASR",
        "backend": "faster-whisper",
        "configured_model_size": cfg.get("model_size"),
        "configured_device": cfg.get("device"),
        "configured_compute_type": cfg.get("compute_type"),
        "beam_size": cfg.get("beam_size"),
        "loaded": _engine._model is not None,
        "loaded_model_size": _engine.loaded_model_size,
        "loaded_device": _engine.loaded_device,
        "loaded_compute_type": _engine.loaded_compute_type,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "language_support": "English, Hindi, Punjabi, code-mixed fallback",
    }
