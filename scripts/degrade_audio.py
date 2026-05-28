"""Generate simulated degraded speech for baseline-deviation evaluation.

Per PRD §5.3 ("simulated degraded speech is used as a controlled proxy")
and TRD §8.1 ("degradation is produced by time-stretching, inserting pauses,
and adding mild disfluencies").

Usage:
    python scripts/degrade_audio.py input.wav --out degraded.wav --severity mild
    python scripts/degrade_audio.py input.wav --severity severe
    python scripts/degrade_audio.py --batch ./samples/clean/ --out-dir ./samples/degraded/

Severity presets:
    mild   — slight slow-down, 1 extra pause, mild low-pass filter.
    medium — 30% slow-down, 2 pauses, noise added.
    severe — 50% slow-down, 4 pauses, heavy filter + noise.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


_PRESETS = {
    "mild":   {"stretch": 1.15, "n_pauses": 1, "pause_ms": 300, "noise_db": -40, "lpf_cutoff": 4000},
    "medium": {"stretch": 1.30, "n_pauses": 2, "pause_ms": 500, "noise_db": -30, "lpf_cutoff": 2500},
    "severe": {"stretch": 1.55, "n_pauses": 4, "pause_ms": 700, "noise_db": -22, "lpf_cutoff": 1600},
}


def _time_stretch(audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """rate > 1 → slower. Uses librosa phase vocoder."""
    import librosa
    return librosa.effects.time_stretch(audio, rate=1.0 / rate)


def _insert_pauses(audio: np.ndarray, sr: int, n: int, ms: int) -> np.ndarray:
    if n <= 0 or len(audio) == 0:
        return audio
    samples_pause = int(sr * ms / 1000)
    chunks = np.array_split(audio, n + 1)
    silence = np.zeros(samples_pause, dtype=audio.dtype)
    out = []
    for i, c in enumerate(chunks):
        out.append(c)
        if i < len(chunks) - 1:
            out.append(silence)
    return np.concatenate(out)


def _add_noise(audio: np.ndarray, target_db: float) -> np.ndarray:
    rms = np.sqrt(np.mean(audio ** 2) + 1e-12)
    target_rms = rms * 10 ** (target_db / 20)
    noise = np.random.default_rng(7).normal(0, target_rms, size=len(audio)).astype(audio.dtype)
    return audio + noise


def _lowpass(audio: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    sos = butter(N=4, Wn=cutoff, btype="lowpass", fs=sr, output="sos")
    return sosfilt(sos, audio).astype(audio.dtype)


def degrade(audio: np.ndarray, sr: int, severity: str = "medium") -> np.ndarray:
    p = _PRESETS[severity]
    a = _time_stretch(audio, sr, p["stretch"])
    a = _insert_pauses(a, sr, p["n_pauses"], p["pause_ms"])
    a = _lowpass(a, sr, p["lpf_cutoff"])
    a = _add_noise(a, p["noise_db"])
    peak = float(np.max(np.abs(a))) or 1.0
    if peak > 1.0:
        a = a / peak * 0.95
    return a.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", help="input wav file")
    ap.add_argument("--out", type=Path, help="output wav file")
    ap.add_argument("--severity", choices=list(_PRESETS), default="medium")
    ap.add_argument("--batch", type=Path, help="degrade all wavs in a folder")
    ap.add_argument("--out-dir", type=Path, default=Path("./degraded"))
    args = ap.parse_args()

    import soundfile as sf

    def _process_one(in_path: Path, out_path: Path) -> None:
        audio, sr = sf.read(str(in_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        out = degrade(audio, sr, args.severity)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), out, sr)
        print(f"{in_path}  →  {out_path}  ({args.severity})")

    if args.batch:
        for wav in sorted(args.batch.glob("*.wav")):
            out = args.out_dir / f"{wav.stem}_{args.severity}.wav"
            _process_one(wav, out)
    else:
        if not args.input:
            ap.error("provide an input file or use --batch")
        out = args.out or Path(args.input).with_name(
            Path(args.input).stem + f"_{args.severity}.wav"
        )
        _process_one(Path(args.input), out)


if __name__ == "__main__":
    main()
