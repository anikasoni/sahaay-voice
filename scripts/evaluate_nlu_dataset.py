"""Evaluate Sahaay Voice NLU on the bundled project-specific benchmark.

This is a report-facing benchmark, not a large public dataset. It covers the
intents that matter for the PRD demo: Safety, Memory, Companion, caregiver
request, clinical deflection, and self-harm escalation.

Outputs:
  data/nlu_eval_report.json
  data/nlu_eval_predictions.csv
  data/nlu_eval_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CRITICAL_INTENTS = {
    "safety_confirmation",
    "clinical_question",
    "self_harm_risk",
    "emergency_help",
}


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def evaluate(path: Path) -> dict:
    from core import nlu as nlu_mod

    rows = _load_jsonl(path)
    labels = sorted({r["intent"] for r in rows})
    tp = Counter()
    fp = Counter()
    fn = Counter()
    lang_total = Counter()
    lang_correct = Counter()
    priority_total = Counter()
    priority_correct = Counter()
    confusion: dict[str, Counter] = defaultdict(Counter)
    predictions: list[dict] = []

    correct = 0
    for r in rows:
        res = nlu_mod.analyze(r["text"])
        gold = r["intent"]
        pred = res.intent
        ok = pred == gold
        correct += int(ok)
        if ok:
            tp[gold] += 1
        else:
            fp[pred] += 1
            fn[gold] += 1
        confusion[gold][pred] += 1
        lang = r.get("language", "unknown")
        pri = r.get("priority", "routine")
        lang_total[lang] += 1
        lang_correct[lang] += int(ok)
        priority_total[pri] += 1
        priority_correct[pri] += int(ok)
        predictions.append({
            "id": r.get("id", ""),
            "text": r["text"],
            "language": lang,
            "priority": pri,
            "gold_intent": gold,
            "pred_intent": pred,
            "correct": ok,
            "intent_confidence": res.intent_confidence,
            "emotion": res.emotion,
            "emotion_confidence": res.emotion_confidence,
            "medicine_name": res.slots.medicine_name or "",
            "time": res.slots.time or "",
            "person": res.slots.person or "",
            "routine": res.slots.routine or "",
        })

    per_intent: dict[str, dict] = {}
    f1_values = []
    for lab in labels:
        p = _safe_div(tp[lab], tp[lab] + fp[lab])
        r = _safe_div(tp[lab], tp[lab] + fn[lab])
        f1 = _safe_div(2 * p * r, p + r)
        f1_values.append(f1)
        per_intent[lab] = {
            "support": sum(1 for x in rows if x["intent"] == lab),
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(f1, 3),
            "tp": tp[lab],
            "fp": fp[lab],
            "fn": fn[lab],
        }

    critical_support = sum(1 for r in rows if r["intent"] in CRITICAL_INTENTS)
    critical_correct = sum(1 for p in predictions if p["gold_intent"] in CRITICAL_INTENTS and p["correct"])
    critical_recall = _safe_div(critical_correct, critical_support)

    report = {
        "dataset": str(path),
        "n": len(rows),
        "n_intents": len(labels),
        "labels": labels,
        "accuracy": round(_safe_div(correct, len(rows)), 3),
        "macro_f1": round(sum(f1_values) / max(1, len(f1_values)), 3),
        "critical_intents": sorted(CRITICAL_INTENTS),
        "critical_recall": round(critical_recall, 3),
        "per_intent": per_intent,
        "per_language_accuracy": {
            lang: round(_safe_div(lang_correct[lang], lang_total[lang]), 3)
            for lang in sorted(lang_total)
        },
        "per_priority_accuracy": {
            pri: round(_safe_div(priority_correct[pri], priority_total[pri]), 3)
            for pri in sorted(priority_total)
        },
        "confusion_matrix": {g: dict(preds) for g, preds in confusion.items()},
        "errors": [p for p in predictions if not p["correct"]],
        "predictions": predictions,
    }
    return report


def write_outputs(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "nlu_eval_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    pred_csv = out_dir / "nlu_eval_predictions.csv"
    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(report["predictions"][0].keys()) if report["predictions"] else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["predictions"])

    summary_csv = out_dir / "nlu_eval_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["intent", "support", "precision", "recall", "f1", "tp", "fp", "fn"])
        writer.writeheader()
        for intent, vals in sorted(report["per_intent"].items()):
            writer.writerow({"intent": intent, **vals})


def print_report(report: dict) -> None:
    print("=" * 64)
    print("Sahaay Voice NLU benchmark")
    print("=" * 64)
    print(f"Examples: {report['n']} | intents: {report['n_intents']}")
    print(f"Accuracy: {report['accuracy']:.3f}")
    print(f"Macro-F1: {report['macro_f1']:.3f}")
    print(f"Critical-intent recall: {report['critical_recall']:.3f}")
    print("\nPer-intent performance:")
    for intent, vals in sorted(report["per_intent"].items()):
        print(f"  {intent:30s} support={vals['support']:2d} P={vals['precision']:.3f} R={vals['recall']:.3f} F1={vals['f1']:.3f}")
    if report["errors"]:
        print("\nErrors:")
        for e in report["errors"][:20]:
            print(f"  [{e['id']}] gold={e['gold_intent']} pred={e['pred_intent']} text={e['text']!r}")
    else:
        print("\nNo misclassifications on this controlled benchmark.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "nlu_eval.jsonl")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    report = evaluate(args.data)
    write_outputs(report, args.out_dir)
    print_report(report)
    print(f"\nReports written to: {args.out_dir}")
    return 0 if not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
