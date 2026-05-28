"""Multilingual Text-to-Speech.

Per TRD §3.2 and §4.9 and §10:
- Pre-synthesize Safety Mode prompts at startup to avoid demo failure.
- Fall back to a documented alternative if the primary engine fails (FR-fallback).
- gTTS chosen as the primary engine for the demo: zero setup, works for en/hi/pa.

Set `engine_primary: xtts` in thresholds.yaml to switch to local Coqui XTTS-v2.
"""
from __future__ import annotations

import hashlib
import io
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from core.config import AUDIO_DIR, thresholds
from core.logger import log
from core.schemas import Language

_CACHE_DIR = AUDIO_DIR / "tts_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_GTTS_LANG = {"en": "en", "hi": "hi", "pa": "pa"}


# --- engines ----------------------------------------------------------------

def _synth_gtts(text: str, lang: Language, out_path: Path) -> bool:
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=_GTTS_LANG.get(lang, "en"), slow=False)
        tts.save(str(out_path))
        return True
    except Exception as e:
        log(f"gTTS failed: {e}", level="warning")
        return False


def _synth_pyttsx3(text: str, lang: Language, out_path: Path) -> bool:
    """Offline fallback — no network needed. Punjabi voice not guaranteed."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.save_to_file(text, str(out_path))
        engine.runAndWait()
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception as e:
        log(f"pyttsx3 failed: {e}", level="warning")
        return False


class _XttsLazy:
    def __init__(self): self._model = None; self._lock = threading.Lock()

    def get(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                try:
                    from TTS.api import TTS as CoquiTTS  # type: ignore
                    log("loading Coqui XTTS-v2…")
                    self._model = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=_cuda_ok())
                except Exception as e:
                    log(f"XTTS init failed; will fall back: {e}", level="warning")
                    self._model = False
        return self._model


_xtts = _XttsLazy()


def _cuda_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _synth_xtts(text: str, lang: Language, out_path: Path) -> bool:
    mdl = _xtts.get()
    if not mdl:
        return False
    try:
        mdl.tts_to_file(
            text=text,
            file_path=str(out_path),
            language=lang if lang in ("en", "hi") else "en",
        )
        return True
    except Exception as e:
        log(f"XTTS synth failed: {e}", level="warning")
        return False


# --- main entry -------------------------------------------------------------

def _cache_key(text: str, lang: Language, engine: str) -> Path:
    h = hashlib.sha1(f"{engine}|{lang}|{text}".encode("utf-8")).hexdigest()[:16]
    return _CACHE_DIR / f"{lang}_{engine}_{h}.mp3"


def synthesize(text: str, lang: Language = "en") -> Optional[Path]:
    """Synth `text` in `lang` and return the audio path. None if all engines fail."""
    cfg = thresholds().get("tts", {})
    if cfg.get("enabled", True) is False:
        return None
    if not text.strip():
        return None
    engine = cfg["engine_punjabi"] if lang == "pa" else cfg["engine_primary"]

    out = _cache_key(text, lang, engine)
    if out.exists():
        return out

    # Engine priority chain
    if engine == "gtts" and _synth_gtts(text, lang, out):
        return out
    if engine == "xtts" and _synth_xtts(text, lang, out):
        return out
    if engine == "pyttsx3" and _synth_pyttsx3(text, lang, out):
        return out

    # Fallbacks
    for fb in ("gtts", "pyttsx3"):
        if fb == engine:
            continue
        out_fb = _cache_key(text, lang, fb)
        if out_fb.exists():
            return out_fb
        ok = (_synth_gtts if fb == "gtts" else _synth_pyttsx3)(text, lang, out_fb)
        if ok:
            log(f"TTS primary {engine} failed → fell back to {fb}", level="warning")
            return out_fb
    log(f"All TTS engines failed for lang={lang}", level="error")
    return None


def precache_safety_prompts() -> None:
    """Pre-synthesize Safety Mode prompts so a network blip can't break a demo."""
    cfg = thresholds().get("tts", {})
    if cfg.get("enabled", True) is False:
        return
    if not cfg.get("cache_safety_prompts", True):
        return
    from core.config import prompts as _prompts
    safety = _prompts().get("safety", {})
    for key in ("ask_confirmation", "ask_repeat", "confirmed_ok", "escalated"):
        entry = safety.get(key, {})
        for lang in ("en", "hi", "pa"):
            text = entry.get(lang)
            if text:
                synthesize(text, lang)  # type: ignore[arg-type]
    log("safety prompts pre-cached")


def play(path: Path) -> None:
    """Play audio (Streamlit handles this client-side; this is for CLI/tests)."""
    try:
        import soundfile as sf
        import sounddevice as sd
        data, sr = sf.read(str(path), dtype="float32")
        sd.play(data, sr)
        sd.wait()
    except Exception as e:
        log(f"audio playback failed: {e}", level="warning")
