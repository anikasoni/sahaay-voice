"""Structured JSON-Lines logger.

Per TRD §4.11: every utterance produces a structured entry covering ASR,
features, NLU, baseline deviation, fusion outcome, response, mode.
Per TRD §9: stored locally only. Operator can purge.
"""
from __future__ import annotations

import json
import logging
import sys
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import LOG_DIR

# Human-readable console logger (separate from the structured JSONL turn log)
_console = logging.getLogger("sahaay")
if not _console.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    _console.addHandler(h)
    _console.setLevel(logging.INFO)


def log(msg: str, level: str = "info", **extra: Any) -> None:
    """Console log with optional structured extras."""
    if extra:
        msg = f"{msg} | {json.dumps(extra, default=str)}"
    getattr(_console, level.lower(), _console.info)(msg)


def _serialize(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def new_turn_id() -> str:
    return uuid.uuid4().hex[:12]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_turn(entry: dict[str, Any]) -> None:
    """Append one conversation turn to today's JSONL log."""
    entry = {"turn_id": entry.get("turn_id") or new_turn_id(),
             "timestamp": entry.get("timestamp") or now_iso(),
             **entry}
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = LOG_DIR / f"turns-{day}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_serialize(entry), ensure_ascii=False) + "\n")


def read_recent_turns(n: int = 20) -> list[dict[str, Any]]:
    """Read up to N most recent turns across all log files."""
    files = sorted(LOG_DIR.glob("turns-*.jsonl"))
    out: list[dict[str, Any]] = []
    for path in reversed(files):
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in reversed(lines):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(out) >= n:
                return list(reversed(out))
    return list(reversed(out))


def purge_all() -> int:
    """Operator action — delete all conversation logs. Returns count removed."""
    count = 0
    for path in LOG_DIR.glob("turns-*.jsonl"):
        path.unlink()
        count += 1
    return count
