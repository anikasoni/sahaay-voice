"""Create an auditable pipeline evidence report from recent JSONL logs.

Usage:
    python scripts/pipeline_report.py --n 25

Outputs:
    data/pipeline_evidence_report.json
    data/pipeline_evidence_report.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import DATA_DIR  # noqa: E402
from core.evidence import evidence_rows, turn_to_evidence  # noqa: E402
from core.logger import read_recent_turns  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25, help="number of recent turns to include")
    args = ap.parse_args()

    turns = read_recent_turns(args.n)
    nested = [turn_to_evidence(t) for t in turns]
    rows = evidence_rows(turns)

    json_path = DATA_DIR / "pipeline_evidence_report.json"
    csv_path = DATA_DIR / "pipeline_evidence_report.csv"
    json_path.write_text(json.dumps(nested, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    print(f"Wrote {len(rows)} turns")
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")


if __name__ == "__main__":
    main()
