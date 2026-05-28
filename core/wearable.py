"""Simulated wearable feed.

Per TRD §4.6: a programmatic publisher of categorical events. Real hardware
is out of scope (PRD §1.2). The fusion layer reads the most recent event
within a configurable time window.

Thread-safe. Used by the operator panel to trigger demo scenarios.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Optional

from core.config import LOG_DIR, thresholds
from core.logger import log
from core.schemas import WearableEvent

_lock = threading.Lock()
_latest: Optional[WearableEvent] = None
_EVENT_LOG = LOG_DIR / "wearable_events.jsonl"


def publish(event_type: str, severity: str = "high") -> WearableEvent:
    """Emit a wearable event. Persists to log and updates the latest pointer."""
    global _latest
    ev = WearableEvent(
        type=event_type,           # type: ignore[arg-type]
        severity=severity,         # type: ignore[arg-type]
        timestamp_ms=int(time.time() * 1000),
        age_ms=0,
    )
    with _lock:
        _latest = ev
    with _EVENT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "type": ev.type, "severity": ev.severity,
            "timestamp_ms": ev.timestamp_ms,
        }) + "\n")
    log("wearable event published", type=ev.type, severity=ev.severity)
    return ev


def latest() -> Optional[WearableEvent]:
    """Return the most recent event if it's still within the window, else None."""
    cfg = thresholds()["fusion"]
    win = cfg["wearable_event_window_ms"]
    with _lock:
        ev = _latest
    if ev is None:
        return None
    age = int(time.time() * 1000) - ev.timestamp_ms
    if age > win:
        return None
    return WearableEvent(
        type=ev.type, severity=ev.severity,
        timestamp_ms=ev.timestamp_ms, age_ms=age,
    )


def clear() -> None:
    """Forget the latest event. Used by the operator panel."""
    global _latest
    with _lock:
        _latest = None
