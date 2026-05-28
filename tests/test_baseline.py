"""Tests for core.baseline. Uses a temp SQLite DB."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from core import baseline as baseline_mod
from core.schemas import SpeechFeatures


@pytest.fixture(autouse=True)
def _temp_db(monkeypatch, tmp_path):
    """Redirect the SQLite path so tests don't touch real data/baseline.sqlite."""
    db = tmp_path / "test_baseline.sqlite"
    monkeypatch.setattr(baseline_mod, "DB_PATH", db)
    yield


def _feat(latency=1000, rate=8.0, clarity=0.8, pause=500, rep=0.0) -> SpeechFeatures:
    return SpeechFeatures(
        latency_ms=latency, pause_total_ms=pause,
        speech_rate_cps=rate, clarity_score=clarity,
        repetition_score=rep,
    )


def test_insufficient_history_no_deviation():
    baseline_mod.reset()
    # Single sample — comparator should report not-sufficient
    baseline_mod.update(_feat())
    dev = baseline_mod.compare(_feat(latency=10_000, clarity=0.1))
    assert dev.sufficient_history is False
    # All z's should be zero since std is undefined
    assert dev.max_z == 0.0


def test_welford_running_stats():
    baseline_mod.reset()
    for i in range(10):
        baseline_mod.update(_feat(latency=1000 + i * 10, rate=8.0, clarity=0.8))
    snap = baseline_mod.snapshot()
    assert snap["sample_count"] == 10
    # mean of 1000..1090 step 10 is 1045
    assert snap["latency_ms_mean"] == pytest.approx(1045.0, abs=1.0)
    assert snap["latency_ms_std"] > 0


def test_z_score_detects_outlier():
    baseline_mod.reset()
    for _ in range(20):
        baseline_mod.update(_feat(latency=1000, rate=8.0, clarity=0.8))
    # Need some variance for z-score to mean anything
    for i in range(5):
        baseline_mod.update(_feat(latency=1000 + i * 50, rate=8.0 + i * 0.1,
                                   clarity=0.8 - i * 0.01))
    # Now feed a far outlier
    dev = baseline_mod.compare(_feat(latency=8000, rate=1.0, clarity=0.1))
    assert dev.sufficient_history is True
    assert dev.max_z > 2.0
    # Robust Phase 1.9 ignores unmeasured latency/repetition and uses std floors,
    # so one very abnormal stable feature is enough to prove z-score detection here.
    assert dev.exceed_count >= 1


def test_abnormal_samples_excluded():
    baseline_mod.reset()
    for _ in range(5):
        baseline_mod.update(_feat(latency=1000), abnormal=False)
    snap_before = baseline_mod.snapshot()
    baseline_mod.update(_feat(latency=50_000), abnormal=True)
    snap_after = baseline_mod.snapshot()
    # Sample count unchanged
    assert snap_after["sample_count"] == snap_before["sample_count"]


def test_reset_clears():
    baseline_mod.reset()
    baseline_mod.update(_feat())
    assert baseline_mod.snapshot()["sample_count"] == 1
    baseline_mod.reset()
    assert baseline_mod.snapshot()["sample_count"] == 0
