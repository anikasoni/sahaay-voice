"""Audio capture + Voice Activity Detection.

Continuous 16kHz mono mic stream. Uses webrtcvad (lightweight, ships in pip)
as the primary VAD. silero-vad listed in TRD as preferred — wire it in by
swapping the `detect_voiced` implementation; the rest of this module doesn't
care which detector is used.

Per TRD §4.1: emits complete utterances (start/end/buffer) when end-of-speech
is detected, with configurable silence trailing and min/max duration.
"""
from __future__ import annotations

import collections
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import numpy as np

from core.config import AUDIO_DIR, thresholds
from core.logger import log

SAMPLE_RATE = 16_000  # required by Whisper + webrtcvad


@dataclass
class Utterance:
    audio: np.ndarray            # float32 mono in [-1, 1]
    sample_rate: int
    start_ms: int                # since stream start
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def save_wav(self, path: Optional[Path] = None) -> Path:
        import soundfile as sf
        path = path or (AUDIO_DIR / f"utt_{int(time.time()*1000)}.wav")
        sf.write(str(path), self.audio, self.sample_rate)
        return path


class _WebRtcVadWrapper:
    """Thin adapter so we can swap to silero-vad later without changes upstream."""

    def __init__(self, aggressiveness: int):
        try:
            import webrtcvad
        except ImportError as e:
            raise ImportError(
                "webrtcvad not installed. `pip install webrtcvad` "
                "(or webrtcvad-wheels on Windows)."
            ) from e
        self._vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, frame_pcm16: bytes, sample_rate: int) -> bool:
        return self._vad.is_speech(frame_pcm16, sample_rate)


def _float32_to_pcm16(frame: np.ndarray) -> bytes:
    return (np.clip(frame, -1, 1) * 32767).astype(np.int16).tobytes()


class AudioCapture:
    """Mic stream with VAD-segmented utterance emission."""

    def __init__(self) -> None:
        cfg = thresholds()["vad"]
        self.frame_ms: int = cfg["frame_ms"]
        self.min_utt_ms: int = cfg["min_utterance_ms"]
        self.max_utt_ms: int = cfg["max_utterance_ms"]
        self.trailing_silence_ms: int = cfg["trailing_silence_ms"]
        self.aggressiveness: int = cfg["aggressiveness"]
        self.frame_samples = SAMPLE_RATE * self.frame_ms // 1000
        self._vad = _WebRtcVadWrapper(self.aggressiveness)
        self._stream = None
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._running = False
        self._t0_ms = 0

    # --- streaming controls -------------------------------------------------

    def start(self) -> None:
        import sounddevice as sd  # imported here so import-time doesn't fail headless

        def _cb(indata, _frames, _ti, _status):
            # indata shape: (n, 1); take channel 0
            self._q.put(indata[:, 0].copy().astype(np.float32))

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=self.frame_samples,
            callback=_cb,
        )
        self._stream.start()
        self._running = True
        self._t0_ms = int(time.time() * 1000)
        log("audio capture started", level="info", sample_rate=SAMPLE_RATE)

    def stop(self) -> None:
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        log("audio capture stopped")

    # --- utterance segmentation --------------------------------------------

    def utterances(self) -> Generator[Utterance, None, None]:
        """Yield Utterance objects as the user finishes speaking."""
        ring: collections.deque[np.ndarray] = collections.deque()
        voiced: list[np.ndarray] = []
        in_speech = False
        silence_run_ms = 0
        speech_run_ms = 0
        utt_start_ms = 0

        while self._running:
            try:
                frame = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if frame.shape[0] != self.frame_samples:
                # sounddevice can deliver short final frames; skip them
                continue

            pcm = _float32_to_pcm16(frame)
            is_voice = self._vad.is_speech(pcm, SAMPLE_RATE)

            if not in_speech:
                ring.append(frame)
                if len(ring) > 10:  # keep ~300ms pre-roll
                    ring.popleft()
                if is_voice:
                    in_speech = True
                    voiced = list(ring)
                    utt_start_ms = int(time.time() * 1000) - self._t0_ms
                    silence_run_ms = 0
                    speech_run_ms = self.frame_ms * len(voiced)
            else:
                voiced.append(frame)
                if is_voice:
                    silence_run_ms = 0
                    speech_run_ms += self.frame_ms
                else:
                    silence_run_ms += self.frame_ms

                if (silence_run_ms >= self.trailing_silence_ms
                        and speech_run_ms >= self.min_utt_ms) \
                        or speech_run_ms >= self.max_utt_ms:
                    audio = np.concatenate(voiced)
                    end_ms = int(time.time() * 1000) - self._t0_ms
                    yield Utterance(
                        audio=audio,
                        sample_rate=SAMPLE_RATE,
                        start_ms=utt_start_ms,
                        end_ms=end_ms,
                    )
                    voiced.clear()
                    ring.clear()
                    in_speech = False
                    silence_run_ms = 0
                    speech_run_ms = 0


def utterance_from_wav(path: str | Path) -> Utterance:
    """Convenience: build an Utterance from a .wav file (for tests, demo replay)."""
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    duration_ms = int(len(audio) / SAMPLE_RATE * 1000)
    return Utterance(audio=audio, sample_rate=SAMPLE_RATE,
                     start_ms=0, end_ms=duration_ms)
