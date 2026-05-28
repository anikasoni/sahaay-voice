from __future__ import annotations

import pytest

from core import baseline as baseline_mod
from core.schemas import SpeechFeatures


@pytest.fixture(autouse=True)
def _temp_db(monkeypatch, tmp_path):
    db = tmp_path / "test_baseline_robust.sqlite"
    monkeypatch.setattr(baseline_mod, "DB_PATH", db)
    yield


def _feat(rate=14.0, clarity=0.78, pause=400, latency=0):
    return SpeechFeatures(
        latency_ms=latency, latency_source="not_measured_no_active_prompt", latency_status="not_measured",
        speech_rate_cps=rate, clarity_score=clarity, pause_total_ms=pause, repetition_score=0.0
    )


def test_single_clarity_drop_does_not_overtrigger_exceed_count():
    baseline_mod.reset()
    for _ in range(10):
        baseline_mod.update(_feat(rate=14.0, clarity=0.78, pause=400))
    dev = baseline_mod.compare(_feat(rate=17.6, clarity=0.53, pause=320))
    assert dev.sufficient_history is True
    # With std floors, one lower-clarity feature alone should not count as 2 features off.
    assert dev.exceed_count < 2
