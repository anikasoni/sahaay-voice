"""Stable command-line baseline calibrator using local microphone.

This bypasses Streamlit/browser recording but uses the same ASR and speech-feature
modules. It updates the baseline only from clean calibration-quality samples and
never raises caregiver alerts during calibration.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sounddevice as sd
import soundfile as sf

from core.audio_capture import utterance_from_wav
from core.config import AUDIO_DIR, thresholds
from core import asr as asr_mod
from core import features as features_mod
from core import nlu as nlu_mod
from core import baseline as baseline_mod

PROMPTS = [
    "Hello, how are you?",
    "I am feeling okay today.",
    "I had breakfast this morning.",
    "I am sitting in my room.",
    "The weather is nice today.",
    "I will drink water now.",
    "I am doing fine.",
    "I watched television today.",
    "I read the newspaper today.",
    "I feel calm right now.",
    "I had lunch in the afternoon.",
    "I am resting comfortably.",
]


def record_wav(path: Path, seconds: float, samplerate: int) -> None:
    print(f"Recording {seconds:.1f}s... speak now.")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    sf.write(path, audio, samplerate)


def quality_status(asr, feats) -> tuple[bool, str, float]:
    cfg = thresholds().get("asr", {})
    try:
        asr_conf = max(0.0, min(1.0, math.exp(float(asr.avg_logprob))))
    except Exception:
        asr_conf = 0.0
    min_conf = float(cfg.get("confidence_weak_threshold", 0.45))
    min_clarity = 0.60  # calibration should be cleaner than ordinary use
    min_voiced = int(cfg.get("min_voiced_ms", 700))
    if asr.is_empty():
        return False, "empty transcript", asr_conf
    if asr_conf < min_conf:
        return False, f"low ASR confidence {asr_conf:.2f}", asr_conf
    if feats.clarity_score < min_clarity:
        return False, f"low clarity {feats.clarity_score:.2f}", asr_conf
    if feats.voiced_duration_ms < min_voiced:
        return False, f"too little voiced speech {feats.voiced_duration_ms} ms", asr_conf
    return True, "accepted", asr_conf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="number of accepted baseline samples to collect")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--samplerate", type=int, default=16000)
    ap.add_argument("--max-attempts", type=int, default=18, help="stop after this many attempts even if n accepted not reached")
    args = ap.parse_args()

    baseline_mod.reset()
    print("Baseline reset. Calibration mode will NOT trigger caregiver alerts.")
    print(f"Target: {args.n} accepted clean baseline samples. Poor-quality samples are skipped.\n")

    accepted = 0
    attempt = 0
    while accepted < args.n and attempt < args.max_attempts:
        sentence = PROMPTS[attempt % len(PROMPTS)]
        attempt += 1
        print("\n" + "=" * 60)
        print(f"Attempt {attempt} | accepted {accepted}/{args.n}")
        print(f"Say clearly: {sentence}")
        input("Press Enter, then speak after recording starts...")
        path = AUDIO_DIR / f"calib_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
        record_wav(path, args.seconds, args.samplerate)
        utt = utterance_from_wav(path)

        asr = asr_mod.transcribe(utt)
        feats = features_mod.extract(utt, asr, prompt_end_ts_ms=None, expected_repetition=None)
        nlu = nlu_mod.analyze(asr.text) if not asr.is_empty() else None
        ok, reason, asr_conf = quality_status(asr, feats)

        print(f"Transcript: {asr.text}")
        print(f"ASR confidence: {asr_conf:.3f} | clarity: {feats.clarity_score:.3f} | voiced_ms: {feats.voiced_duration_ms}")
        print(f"speech_rate: {feats.speech_rate_cps:.3f} | internal_pause_ms: {feats.pause_total_ms}")
        if nlu:
            print(f"NLU preview: {nlu.intent} ({nlu.intent_confidence:.3f}), emotion={nlu.emotion}")

        if not ok:
            print(f"SKIPPED: {reason}")
            continue

        baseline_mod.update(feats, abnormal=False)
        accepted += 1
        snap = baseline_mod.snapshot()
        dev = baseline_mod.compare(feats)
        print(f"ACCEPTED baseline sample {accepted}/{args.n}")
        print(f"Baseline sample_count: {snap.get('sample_count')} | sufficient: {dev.sufficient_history}")

    snap = baseline_mod.snapshot()
    print("\nDone.")
    print(f"Accepted samples: {snap.get('sample_count', 0)}")
    if int(snap.get("sample_count", 0)) < args.n:
        print("WARNING: fewer than target samples accepted. Re-run calibrator or reduce --n only for demo.")
    else:
        print("Baseline calibration complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
