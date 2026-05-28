"""Evaluation harness for Sahaay Voice.

Implements the four eval families from TRD §8:

  1. ASR — WER per language on a small held-out clean set.
  2. NLU — intent classification accuracy + macro-F1 on a labeled jsonl.
  3. Baseline — precision/recall of deviation detection on clean vs
     degraded utterance pairs (PRD §10 acceptance criterion).
  4. Fusion — pass/fail on a list of scripted scenarios.

Usage:
    # Run everything that has data; skip what doesn't.
    python scripts/evaluate.py --all

    # Individual suites:
    python scripts/evaluate.py --asr --asr-dir ./samples/clean_with_refs/
    python scripts/evaluate.py --nlu --nlu-data data/intent_test.jsonl
    python scripts/evaluate.py --baseline --clean-dir ./samples/clean/ --degraded-dir ./samples/degraded/
    python scripts/evaluate.py --fusion

Data conventions:
  ASR set:     a folder of .wav files; each `foo.wav` has a sibling `foo.txt`
               (reference transcript) and an optional `foo.lang` file
               containing one of {en,hi,pa}.
  NLU set:     jsonl with rows {"text": ..., "intent": ...}.
  Baseline:    paired folders — same filenames in clean/ and degraded/.

All results dumped to data/eval_report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# 1. ASR — Word Error Rate
# ---------------------------------------------------------------------------

def _wer(ref: str, hyp: str) -> float:
    ref_words = ref.lower().split()
    hyp_words = hyp.lower().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    # Levenshtein on word tokens
    prev = list(range(len(hyp_words) + 1))
    for i, r in enumerate(ref_words, 1):
        curr = [i] + [0] * len(hyp_words)
        for j, h in enumerate(hyp_words, 1):
            cost = 0 if r == h else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1] / len(ref_words)


def eval_asr(folder: Path) -> dict:
    from core import asr as asr_mod
    from core.audio_capture import utterance_from_wav

    rows = []
    by_lang: dict[str, list[float]] = defaultdict(list)
    wavs = sorted(folder.glob("*.wav"))
    if not wavs:
        return {"skipped": True, "reason": f"no wavs in {folder}"}

    for wav in wavs:
        ref_path = wav.with_suffix(".txt")
        if not ref_path.exists():
            continue
        ref = ref_path.read_text(encoding="utf-8").strip()
        lang_path = wav.with_suffix(".lang")
        hint_lang = lang_path.read_text().strip() if lang_path.exists() else None

        utt = utterance_from_wav(wav)
        t0 = time.time()
        res = asr_mod.transcribe(utt, hint_lang=hint_lang)  # type: ignore[arg-type]
        rtf = (time.time() - t0) / max(0.001, utt.duration_ms / 1000)
        wer = _wer(ref, res.text)
        rows.append({
            "file": wav.name, "ref": ref, "hyp": res.text,
            "lang_hint": hint_lang, "lang_detected": res.language,
            "wer": round(wer, 3), "rtf": round(rtf, 3),
        })
        by_lang[hint_lang or res.language].append(wer)

    out = {
        "n_files": len(rows),
        "per_language_wer": {l: round(sum(v) / len(v), 3) for l, v in by_lang.items()},
        "overall_wer": round(sum(r["wer"] for r in rows) / max(1, len(rows)), 3),
        "rows": rows,
    }
    return out


# ---------------------------------------------------------------------------
# 2. NLU — accuracy and macro-F1
# ---------------------------------------------------------------------------

def eval_nlu(path: Path) -> dict:
    if not path.exists():
        return {"skipped": True, "reason": f"no test set at {path}"}
    from core import nlu as nlu_mod

    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_label_tp: dict[str, int] = defaultdict(int)
    by_label_fp: dict[str, int] = defaultdict(int)
    by_label_fn: dict[str, int] = defaultdict(int)
    correct = 0
    preds = []

    for r in rows:
        res = nlu_mod.analyze(r["text"])
        pred = res.intent
        gold = r["intent"]
        preds.append({"text": r["text"], "gold": gold, "pred": pred,
                      "conf": res.intent_confidence})
        if pred == gold:
            correct += 1
            by_label_tp[gold] += 1
        else:
            by_label_fp[pred] += 1
            by_label_fn[gold] += 1

    labels = set(by_label_tp) | set(by_label_fp) | set(by_label_fn)
    per_label_f1 = {}
    for l in labels:
        tp, fp, fn = by_label_tp[l], by_label_fp[l], by_label_fn[l]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        rcl = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * rcl / (p + rcl) if (p + rcl) else 0.0
        per_label_f1[l] = {"precision": round(p, 3), "recall": round(rcl, 3),
                            "f1": round(f1, 3), "tp": tp, "fp": fp, "fn": fn}

    macro_f1 = sum(v["f1"] for v in per_label_f1.values()) / max(1, len(per_label_f1))
    return {
        "n": len(rows),
        "accuracy": round(correct / max(1, len(rows)), 3),
        "macro_f1": round(macro_f1, 3),
        "per_label": per_label_f1,
        "predictions": preds[:50],   # cap for log size
    }


# ---------------------------------------------------------------------------
# 3. Baseline — detection rate on clean vs degraded pairs
# ---------------------------------------------------------------------------

def eval_baseline(clean_dir: Path, degraded_dir: Path) -> dict:
    """Build the baseline from clean utterances, then check that degraded
    ones get flagged with a high max-z while clean ones don't."""
    from core import asr as asr_mod
    from core import baseline as baseline_mod
    from core import features as feat_mod
    from core.audio_capture import utterance_from_wav

    if not clean_dir.exists() or not degraded_dir.exists():
        return {"skipped": True, "reason": "clean or degraded dir missing"}

    clean_wavs = sorted(clean_dir.glob("*.wav"))
    degraded_wavs = sorted(degraded_dir.glob("*.wav"))
    if len(clean_wavs) < 6:
        return {"skipped": True, "reason": "need >= 6 clean wavs to seed baseline"}

    # Reset baseline so this eval is reproducible
    baseline_mod.reset()
    n_seed = max(5, len(clean_wavs) // 2)
    seed = clean_wavs[:n_seed]
    held_out_clean = clean_wavs[n_seed:]

    for w in seed:
        utt = utterance_from_wav(w)
        asr = asr_mod.transcribe(utt)
        feats = feat_mod.extract(utt, asr)
        baseline_mod.update(feats, abnormal=False)

    # Now score the held-out clean and the degraded set
    clean_zs = []
    for w in held_out_clean:
        utt = utterance_from_wav(w)
        asr = asr_mod.transcribe(utt)
        feats = feat_mod.extract(utt, asr)
        dev = baseline_mod.compare(feats)
        clean_zs.append(dev.max_z)

    degraded_zs = []
    for w in degraded_wavs:
        utt = utterance_from_wav(w)
        asr = asr_mod.transcribe(utt)
        feats = feat_mod.extract(utt, asr)
        dev = baseline_mod.compare(feats)
        degraded_zs.append(dev.max_z)

    # Detection threshold = thresholds.yaml z_high; report precision/recall
    from core.config import thresholds
    z_high = thresholds()["baseline"]["z_high"]
    tp = sum(1 for z in degraded_zs if z >= z_high)
    fn = len(degraded_zs) - tp
    fp = sum(1 for z in clean_zs if z >= z_high)
    tn = len(clean_zs) - fp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "z_high": z_high,
        "n_seed": len(seed),
        "n_clean_eval": len(held_out_clean),
        "n_degraded_eval": len(degraded_wavs),
        "clean_zs": [round(z, 2) for z in clean_zs],
        "degraded_zs": [round(z, 2) for z in degraded_zs],
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
    }


# ---------------------------------------------------------------------------
# 4. Fusion — scripted scenario battery
# ---------------------------------------------------------------------------

def eval_fusion() -> dict:
    """Run a fixed set of scripted scenarios through the fusion decider.
    Mirrors and extends TRD §4.7.2 + PRD §10 acceptance scenarios."""
    import time as _time
    from core.fusion import decide
    from core.schemas import (
        BaselineDeviation, NluResult, SpeechFeatures, WearableEvent,
    )

    def _now_ms(): return int(_time.time() * 1000)

    scenarios = [
        {
            "name": "calm chit-chat",
            "args": dict(
                nlu=NluResult(intent="casual_chat", intent_confidence=0.9, emotion="calm"),
                features=SpeechFeatures(),
                deviation=BaselineDeviation(max_z=0.5, exceed_count=0, sufficient_history=True),
                wearable=None,
            ),
            "expected": "normal_response",
        },
        {
            "name": "clinical question → safe deflection",
            "args": dict(
                nlu=NluResult(intent="clinical_question", intent_confidence=0.95, emotion="anxious"),
                features=SpeechFeatures(),
                deviation=BaselineDeviation(sufficient_history=True),
                wearable=None,
            ),
            "expected": "safe_deflection",
        },
        {
            "name": "explicit emergency",
            "args": dict(
                nlu=NluResult(intent="emergency_help", intent_confidence=0.95, emotion="anxious"),
                features=SpeechFeatures(),
                deviation=BaselineDeviation(sufficient_history=True),
                wearable=None,
            ),
            "expected": "caregiver_alert",
        },
        {
            "name": "fall (no pending safety) → enter safety",
            "args": dict(
                nlu=NluResult(intent="casual_chat", intent_confidence=0.5, emotion="calm"),
                features=SpeechFeatures(),
                deviation=BaselineDeviation(sufficient_history=True),
                wearable=WearableEvent(type="fall_detected", severity="high",
                                       timestamp_ms=_now_ms(), age_ms=0),
            ),
            "expected": "normal_response",   # manager interprets as safety mode entry
        },
        {
            "name": "fall + safety pending + bad reply → alert",
            "args": dict(
                nlu=NluResult(intent="safety_confirmation", intent_confidence=0.3, emotion="confused"),
                features=SpeechFeatures(repetition_score=0.1, clarity_score=0.1),
                deviation=BaselineDeviation(sufficient_history=True),
                wearable=WearableEvent(type="fall_detected", severity="high",
                                       timestamp_ms=_now_ms(), age_ms=0),
                safety_pending=True, is_safety_reply=True,
            ),
            "expected": "caregiver_alert",
        },
        {
            "name": "safety pending + clear good reply → normal",
            "args": dict(
                nlu=NluResult(intent="safety_confirmation", intent_confidence=0.9, emotion="calm"),
                features=SpeechFeatures(repetition_score=0.9, clarity_score=0.9),
                deviation=BaselineDeviation(sufficient_history=True),
                wearable=None,
                safety_pending=True, is_safety_reply=True,
            ),
            "expected": "normal_response",
        },
        {
            "name": "reminder unack escalates",
            "args": dict(
                nlu=NluResult(intent="casual_chat", intent_confidence=0.6, emotion="calm"),
                features=SpeechFeatures(),
                deviation=BaselineDeviation(sufficient_history=True),
                wearable=None,
                reminder_unack_count=3,
            ),
            "expected": "caregiver_alert",
        },
        {
            "name": "strong baseline deviation → soft check-in",
            "args": dict(
                nlu=NluResult(intent="casual_chat", intent_confidence=0.7, emotion="calm"),
                features=SpeechFeatures(),
                deviation=BaselineDeviation(max_z=3.0, exceed_count=3, sufficient_history=True),
                wearable=None,
            ),
            "expected": "soft_check_in",
        },
        {
            "name": "mild deviation without history → normal",
            "args": dict(
                nlu=NluResult(intent="casual_chat", intent_confidence=0.7, emotion="calm"),
                features=SpeechFeatures(),
                deviation=BaselineDeviation(max_z=5.0, exceed_count=5, sufficient_history=False),
                wearable=None,
            ),
            "expected": "normal_response",
        },
    ]

    rows = []
    passes = 0
    for s in scenarios:
        d = decide(**s["args"])
        ok = d.outcome == s["expected"]
        passes += int(ok)
        rows.append({
            "name": s["name"], "expected": s["expected"],
            "got": d.outcome, "reason": d.reason, "pass": ok,
        })
    return {
        "n": len(scenarios),
        "pass": passes,
        "fail": len(scenarios) - passes,
        "pass_rate": round(passes / len(scenarios), 3),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="run every suite for which data exists")
    ap.add_argument("--asr", action="store_true")
    ap.add_argument("--asr-dir", type=Path, default=ROOT / "data" / "asr_eval")
    ap.add_argument("--nlu", action="store_true")
    ap.add_argument("--nlu-data", type=Path, default=ROOT / "data" / "intent_test.jsonl")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--clean-dir", type=Path, default=ROOT / "data" / "clean")
    ap.add_argument("--degraded-dir", type=Path, default=ROOT / "data" / "degraded")
    ap.add_argument("--fusion", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "eval_report.json")
    args = ap.parse_args()

    report: dict = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    if args.all or args.asr:
        print("=" * 60)
        print("ASR evaluation")
        print("=" * 60)
        report["asr"] = eval_asr(args.asr_dir)
        print(json.dumps(report["asr"], indent=2, ensure_ascii=False)[:800])

    if args.all or args.nlu:
        print("=" * 60)
        print("NLU evaluation")
        print("=" * 60)
        report["nlu"] = eval_nlu(args.nlu_data)
        if "accuracy" in report["nlu"]:
            print(f"accuracy: {report['nlu']['accuracy']}, "
                  f"macro_f1: {report['nlu']['macro_f1']}")
        else:
            print(report["nlu"])

    if args.all or args.baseline:
        print("=" * 60)
        print("Baseline-deviation evaluation")
        print("=" * 60)
        report["baseline"] = eval_baseline(args.clean_dir, args.degraded_dir)
        print(json.dumps(report["baseline"], indent=2)[:800])

    if args.all or args.fusion:
        print("=" * 60)
        print("Fusion scenario evaluation")
        print("=" * 60)
        report["fusion"] = eval_fusion()
        for row in report["fusion"]["rows"]:
            mark = "✓" if row["pass"] else "✗"
            print(f"  {mark}  {row['name']:50s}  → {row['got']}")
        print(f"\n  {report['fusion']['pass']}/{report['fusion']['n']} passed "
              f"({report['fusion']['pass_rate'] * 100:.0f}%)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nfull report → {args.out}")


if __name__ == "__main__":
    main()
