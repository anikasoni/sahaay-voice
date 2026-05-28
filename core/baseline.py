"""Personal baseline store and comparator.

Per TRD §4.5: rolling mean+std per user across a small set of speech features,
plus z-score deviation. Storage is SQLite, single user for the prototype.

Uses Welford's online algorithm so we never store raw samples — privacy-aware
and constant memory regardless of history length (FR-11..FR-14).
"""
from __future__ import annotations

import math
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Iterator, Optional

from core.config import DB_PATH, thresholds
from core.logger import log
from core.schemas import BaselineDeviation, SpeechFeatures

# Features tracked in the baseline (must match SpeechFeatures.as_baseline_features)
TRACKED_FEATURES = (
    "latency_ms",
    "speech_rate_cps",
    "pause_total_ms",
    "clarity_score",
    "repetition_score",
)

DEFAULT_USER_ID = "elder_01"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_baseline (
    user_id          TEXT PRIMARY KEY,
    language_default TEXT,
    sample_count     INTEGER NOT NULL DEFAULT 0,
    -- Welford running aggregates: mean, M2 (=sum of squared deltas)
    latency_ms_mean       REAL NOT NULL DEFAULT 0,
    latency_ms_m2         REAL NOT NULL DEFAULT 0,
    speech_rate_cps_mean  REAL NOT NULL DEFAULT 0,
    speech_rate_cps_m2    REAL NOT NULL DEFAULT 0,
    pause_total_ms_mean   REAL NOT NULL DEFAULT 0,
    pause_total_ms_m2     REAL NOT NULL DEFAULT 0,
    clarity_score_mean    REAL NOT NULL DEFAULT 0,
    clarity_score_m2      REAL NOT NULL DEFAULT 0,
    repetition_score_mean REAL NOT NULL DEFAULT 0,
    repetition_score_m2   REAL NOT NULL DEFAULT 0,
    updated_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

_lock = threading.Lock()


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    c = sqlite3.connect(str(DB_PATH))
    c.execute(_SCHEMA)
    try:
        yield c
        c.commit()
    finally:
        c.close()


def _ensure_user(user_id: str, language: str = "en") -> None:
    with _conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO user_baseline(user_id, language_default) VALUES (?, ?)",
            (user_id, language),
        )


def update(
    features: SpeechFeatures,
    abnormal: bool = False,
    user_id: str = DEFAULT_USER_ID,
) -> None:
    """Incrementally update baseline using Welford's algorithm.

    Per FR-12 we DO NOT include abnormal utterances in the rolling stats.
    """
    if abnormal:
        return
    _ensure_user(user_id)
    fvals = features.as_baseline_features()

    with _lock, _conn() as c:
        row = c.execute(
            "SELECT sample_count, "
            "latency_ms_mean, latency_ms_m2, "
            "speech_rate_cps_mean, speech_rate_cps_m2, "
            "pause_total_ms_mean, pause_total_ms_m2, "
            "clarity_score_mean, clarity_score_m2, "
            "repetition_score_mean, repetition_score_m2 "
            "FROM user_baseline WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            return
        (n, *flat) = row
        means_m2 = {
            "latency_ms":       [flat[0], flat[1]],
            "speech_rate_cps":  [flat[2], flat[3]],
            "pause_total_ms":   [flat[4], flat[5]],
            "clarity_score":    [flat[6], flat[7]],
            "repetition_score": [flat[8], flat[9]],
        }
        n = int(n) + 1
        for f in TRACKED_FEATURES:
            x = fvals[f]
            mean, m2 = means_m2[f]
            delta = x - mean
            mean = mean + delta / n
            delta2 = x - mean
            m2 = m2 + delta * delta2
            means_m2[f] = [mean, m2]

        c.execute(
            "UPDATE user_baseline SET sample_count=?, "
            "latency_ms_mean=?, latency_ms_m2=?, "
            "speech_rate_cps_mean=?, speech_rate_cps_m2=?, "
            "pause_total_ms_mean=?, pause_total_ms_m2=?, "
            "clarity_score_mean=?, clarity_score_m2=?, "
            "repetition_score_mean=?, repetition_score_m2=?, "
            "updated_at=CURRENT_TIMESTAMP WHERE user_id=?",
            (
                n,
                *means_m2["latency_ms"],
                *means_m2["speech_rate_cps"],
                *means_m2["pause_total_ms"],
                *means_m2["clarity_score"],
                *means_m2["repetition_score"],
                user_id,
            ),
        )


def snapshot(user_id: str = DEFAULT_USER_ID) -> dict[str, float | int]:
    """Return current baseline stats — for the operator panel."""
    _ensure_user(user_id)
    with _conn() as c:
        row = c.execute(
            "SELECT sample_count, "
            "latency_ms_mean, latency_ms_m2, "
            "speech_rate_cps_mean, speech_rate_cps_m2, "
            "pause_total_ms_mean, pause_total_ms_m2, "
            "clarity_score_mean, clarity_score_m2, "
            "repetition_score_mean, repetition_score_m2 "
            "FROM user_baseline WHERE user_id=?", (user_id,)
        ).fetchone()
    if row is None:
        return {"sample_count": 0}
    n = int(row[0])
    out: dict[str, float | int] = {"sample_count": n}
    fields = list(TRACKED_FEATURES)
    for i, f in enumerate(fields):
        mean = row[1 + 2 * i]
        m2 = row[2 + 2 * i]
        std = math.sqrt(m2 / (n - 1)) if n > 1 else 0.0
        out[f"{f}_mean"] = float(round(mean, 3))
        out[f"{f}_std"] = float(round(std, 3))
    return out


def compare(features: SpeechFeatures, user_id: str = DEFAULT_USER_ID) -> BaselineDeviation:
    """Compute per-feature z-scores and aggregate deviation metrics."""
    cfg = thresholds()["baseline"]
    snap = snapshot(user_id)
    n = int(snap.get("sample_count", 0))

    sufficient = n >= cfg["min_samples_before_flagging"]
    fvals = features.as_baseline_features()
    per_z: dict[str, float] = {}
    std_floor = cfg.get("std_floor", {})
    for f, x in fvals.items():
        # Latency in Streamlit idle/companion mode is often explicitly not measured.
        # Do not let synthetic latency values drive baseline alarms.
        if f == "latency_ms" and cfg.get("ignore_unmeasured_latency", True):
            if getattr(features, "latency_status", "") in {"not_measured", "unknown"} or getattr(features, "latency_source", "") == "not_measured_no_active_prompt":
                per_z[f] = 0.0
                continue

        # Repetition score is meaningful only for Safety Mode repeat prompts. In normal
        # speech it is structurally zero and should not count as an abnormality.
        if f == "repetition_score" and cfg.get("ignore_repetition_outside_safety", True):
            if float(getattr(features, "repetition_score", 0.0)) == 0.0:
                per_z[f] = 0.0
                continue

        mean = float(snap.get(f"{f}_mean", 0.0))
        raw_std = float(snap.get(f"{f}_std", 0.0))
        floor = float(std_floor.get(f, 0.0))
        std = max(raw_std, floor)
        if not sufficient or std <= 1e-6:
            per_z[f] = 0.0
            continue
        per_z[f] = float(round((float(x) - mean) / std, 3))

    abs_zs = [abs(z) for z in per_z.values()]
    max_z = max(abs_zs) if abs_zs else 0.0
    # Count only clearly abnormal features. Using z_low=2.0 avoids false alarms
    # from one borderline feature plus one noisy feature.
    exceed = sum(1 for z in abs_zs if z >= cfg.get("z_low", 2.0))

    return BaselineDeviation(
        max_z=float(round(max_z, 3)),
        exceed_count=exceed,
        per_feature_z=per_z,
        sufficient_history=sufficient,
    )


def reset(user_id: str = DEFAULT_USER_ID) -> None:
    """Wipe baseline for this user. Used by operator panel."""
    with _lock, _conn() as c:
        c.execute("DELETE FROM user_baseline WHERE user_id=?", (user_id,))
    log("baseline reset", user_id=user_id)
