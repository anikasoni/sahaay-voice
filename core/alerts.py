"""Caregiver alert store.

Per TRD §5.3 and PRD FR-35: alerts are displayed in the caregiver panel
with timestamp, reason, severity, and recent transcript context.

In-memory + JSONL persistence so the panel survives a Streamlit rerun.
"""
from __future__ import annotations

import json
import threading
import uuid
from typing import Optional

from core.config import LOG_DIR
from core.logger import log, now_iso
from core.schemas import CaregiverAlert

_ALERTS_PATH = LOG_DIR / "caregiver_alerts.jsonl"
_lock = threading.Lock()
_alerts: list[CaregiverAlert] = []
_loaded = False


def _load_once() -> None:
    global _loaded
    if _loaded:
        return
    if _ALERTS_PATH.exists():
        with _ALERTS_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    _alerts.append(CaregiverAlert(**d))
                except Exception:
                    continue
    _loaded = True


def raise_alert(
    reason: str,
    severity: str = "warning",
    recent_transcript: Optional[list[str]] = None,
    related_turn_ids: Optional[list[str]] = None,
) -> CaregiverAlert:
    _load_once()
    a = CaregiverAlert(
        alert_id=uuid.uuid4().hex[:10],
        timestamp=now_iso(),
        reason=reason,
        severity=severity,                 # type: ignore[arg-type]
        recent_transcript=recent_transcript or [],
        related_turn_ids=related_turn_ids or [],
    )
    with _lock:
        _alerts.append(a)
        with _ALERTS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(a.__dict__, ensure_ascii=False) + "\n")
    log("CAREGIVER ALERT", level="warning",
        reason=reason, severity=severity)
    return a


def list_alerts(limit: int = 50) -> list[CaregiverAlert]:
    _load_once()
    with _lock:
        return list(reversed(_alerts[-limit:]))


def clear_all() -> int:
    """Wipe alerts. Operator action."""
    global _alerts
    _load_once()
    with _lock:
        n = len(_alerts)
        _alerts = []
        if _ALERTS_PATH.exists():
            _ALERTS_PATH.unlink()
    log("caregiver alerts cleared", count=n)
    return n
