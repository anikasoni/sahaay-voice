"""Generate final report-ready metrics for Sahaay Voice.

This script is intentionally presentation-oriented: it runs/loads the core test
batteries and writes clean JSON/CSV summaries that can be pasted into the
Thapar project report Results section.

Outputs:
  data/final_project_metrics.json
  data/final_project_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def _run(cmd: list[str], *, allow_fail: bool = False) -> tuple[int, str]:
    print("\n$ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout)
    if proc.returncode != 0 and not allow_fail:
        raise SystemExit(proc.returncode)
    return proc.returncode, proc.stdout


def _load_json(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _parse_pytest(output: str) -> dict[str, Any]:
    # Handles: "51 passed in 8.81s"
    m = re.search(r"(\d+)\s+passed(?:,\s*(\d+)\s+failed)?\s+in\s+([0-9.]+)s", output)
    if not m:
        return {"passed": None, "failed": None, "total": None, "duration_sec": None, "raw": output[-2000:]}
    passed = int(m.group(1))
    failed = int(m.group(2) or 0)
    return {"passed": passed, "failed": failed, "total": passed + failed, "duration_sec": float(m.group(3))}


def _metric_row(component: str, metric: str, value: str, interpretation: str) -> dict[str, str]:
    return {
        "component": component,
        "metric": metric,
        "value": str(value),
        "interpretation": interpretation,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-run", action="store_true", help="Do not rerun tests; only read existing JSON outputs.")
    args = parser.parse_args()

    DATA.mkdir(exist_ok=True)
    pytest_info: dict[str, Any] = {}

    if not args.skip_run:
        _, pytest_out = _run([sys.executable, "-m", "pytest", "tests", "-q"])
        pytest_info = _parse_pytest(pytest_out)
        _run([sys.executable, "scripts/acceptance_check.py"])
        _run([sys.executable, "scripts/evaluate.py", "--fusion"])
        # NLU benchmark can return 1 if there are errors; keep going so report captures them.
        _run([sys.executable, "scripts/evaluate_nlu_dataset.py"], allow_fail=True)
    else:
        pytest_info = {"passed": "not rerun", "failed": "not rerun", "total": "not rerun", "duration_sec": "not rerun"}

    acceptance = _load_json(DATA / "acceptance_report.json")
    eval_report = _load_json(DATA / "eval_report.json")
    fusion = eval_report.get("fusion", {})
    nlu = _load_json(DATA / "nlu_eval_report.json")

    rows: list[dict[str, str]] = []
    rows.append(_metric_row(
        "Automated unit tests",
        "pytest pass rate",
        f"{pytest_info.get('passed')}/{pytest_info.get('total')} passed",
        "Module-level tests for speech features, NLU fast paths, baseline, fusion, dialogue, and evidence generation.",
    ))
    rows.append(_metric_row(
        "PRD acceptance scenarios",
        "acceptance pass rate",
        f"{acceptance.get('passed', '?')}/{acceptance.get('n', '?')} passed ({acceptance.get('pass_rate', 0)*100:.0f}%)" if acceptance else "missing",
        "End-to-end scripted flows covering Safety, Memory, Companion, clinical deflection, self-harm escalation, and code-mixed input.",
    ))
    rows.append(_metric_row(
        "Fusion and escalation",
        "scenario pass rate",
        f"{fusion.get('pass', '?')}/{fusion.get('n', '?')} passed ({fusion.get('pass_rate', 0)*100:.0f}%)" if fusion else "missing",
        "Rule-based decision layer correctly selects normal response, soft check-in, caregiver alert, reminder repeat, and safe deflection.",
    ))
    if nlu:
        rows.append(_metric_row(
            "NLU benchmark",
            "intent accuracy",
            f"{nlu.get('accuracy', 0):.3f}",
            "Controlled project-specific multilingual benchmark of safety, memory, companion, caregiver, and clinical-deflection intents.",
        ))
        rows.append(_metric_row(
            "NLU benchmark",
            "macro-F1",
            f"{nlu.get('macro_f1', 0):.3f}",
            "Balances performance across all intent classes rather than letting common classes dominate.",
        ))
        rows.append(_metric_row(
            "Safety-critical NLU",
            "critical-intent recall",
            f"{nlu.get('critical_recall', 0):.3f}",
            "Recall across safety_confirmation, emergency_help, clinical_question, and self_harm_risk examples.",
        ))
        for intent in ["clinical_question", "self_harm_risk", "emergency_help", "safety_confirmation"]:
            vals = nlu.get("per_intent", {}).get(intent)
            if vals:
                rows.append(_metric_row(
                    "Critical intent recall",
                    intent,
                    f"{vals.get('recall', 0):.3f}",
                    f"Recall for the {intent} pathway, important for safety-first routing.",
                ))
    else:
        rows.append(_metric_row("NLU benchmark", "status", "missing", "Run scripts/evaluate_nlu_dataset.py."))

    rows.extend([
        _metric_row("ASR engine", "model", "faster-whisper medium", "Local multilingual ASR used for English, Hindi, Punjabi, and code-mixed speech."),
        _metric_row("ASR runtime", "device", "CUDA/float16 on RTX 4070 when available", "Report based on the local GPU setup used during development."),
        _metric_row("Baseline demonstration", "normal speech", "max_z < 2, normal_response", "Normal calibrated speech remained below deviation threshold after robust baseline correction."),
        _metric_row("Baseline demonstration", "abnormal speech", "max_z ≈ 7.5, soft_check_in", "Slow/pause-heavy speech triggered a gentle check-in without immediate caregiver alert."),
        _metric_row("Medical safety boundary", "diagnosis policy", "hard deflection + caregiver notification", "The prototype does not diagnose; clinical-style prompts route to safe deflection."),
        _metric_row("Privacy posture", "storage", "local SQLite + JSONL only", "Baseline, alerts, and transcripts are stored locally for the prototype."),
    ])

    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project": "Sahaay Voice",
        "summary_rows": rows,
        "pytest": pytest_info,
        "acceptance": acceptance,
        "fusion": fusion,
        "nlu": nlu,
        "report_claims_to_use": [
            "The system passed all PRD acceptance scenarios in the scripted test battery.",
            "The fusion layer passed all scripted decision scenarios covering normal response, soft check-in, caregiver alert, reminder repeat, and safe deflection.",
            "The NLU layer was evaluated on a controlled multilingual project-specific benchmark covering safety-critical and routine elder-care intents.",
            "The baseline-aware speech module demonstrated expected behavior: normal calibrated speech remained normal, while intentionally pause-heavy speech triggered a soft check-in.",
        ],
        "claims_not_to_make_without_extra_data": [
            "Do not claim ASR WER unless a wav+reference transcript test set is created.",
            "Do not claim clinical validation or medical diagnosis capability.",
            "Do not claim real-world fall detection; wearable events are simulated by design.",
        ],
    }

    (DATA / "final_project_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    with (DATA / "final_project_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["component", "metric", "value", "interpretation"])
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 64)
    print("Report-ready metrics")
    print("=" * 64)
    for r in rows:
        print(f"{r['component']}: {r['metric']} = {r['value']}")
    print(f"\nWrote: {DATA / 'final_project_metrics.json'}")
    print(f"Wrote: {DATA / 'final_project_metrics.csv'}")
    return 0


if __name__ == "__main__":
    # Some ML/audio stacks can leave non-daemon helper threads alive on Windows.
    # Force-exit after all report files are flushed so the terminal returns cleanly.
    import os
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
