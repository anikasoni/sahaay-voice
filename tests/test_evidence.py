from __future__ import annotations

from core.dialogue import TurnResult
from core.evidence import evidence_rows, turn_to_evidence
from core.schemas import AsrResult, BaselineDeviation, FusionDecision, NluResult, SpeechFeatures


def test_turn_to_evidence_contains_all_pipeline_stages():
    t = TurnResult(
        turn_id="abc",
        mode_before="idle",
        mode_after="companion",
        asr=AsrResult(text="hello", language="en", avg_logprob=-0.2),
        features=SpeechFeatures(speech_duration_ms=1000, pause_total_ms=200, pause_count=1, speech_rate_cps=5, clarity_score=0.8),
        nlu=NluResult(intent="casual_chat", intent_confidence=0.9, emotion="calm", emotion_confidence=0.7),
        deviation=BaselineDeviation(max_z=0.5, exceed_count=0, per_feature_z={"clarity_score": 0.5}, sufficient_history=True),
        decision=FusionDecision(outcome="normal_response", reason="ok"),
        response_text="hello",
        response_lang="en",
        response_audio_path=None,
    )
    ev = turn_to_evidence(t)
    assert ev["asr"]["transcript"] == "hello"
    assert ev["speech_features"]["voice_activity_ratio"] > 0
    assert ev["nlu"]["intent"] == "casual_chat"
    assert ev["baseline"]["max_z"] == 0.5
    assert ev["fusion"]["outcome"] == "normal_response"


def test_evidence_rows_flatten():
    row = evidence_rows([{
        "turn_id": "1",
        "timestamp": "now",
        "mode_before": "idle",
        "mode_after": "idle",
        "asr": {"text": "done", "language": "en", "avg_logprob": -0.1},
        "features": {"clarity_score": 0.9, "speech_rate_cps": 4.0},
        "nlu": {"intent": "task_acknowledgement", "intent_confidence": 0.97, "emotion": "calm"},
        "baseline_deviation": {"max_z": 0, "exceed_count": 0},
        "fusion_outcome": "normal_response",
        "fusion_reason": "ok",
    }])[0]
    assert row["intent"] == "task_acknowledgement"
    assert row["outcome"] == "normal_response"


def test_phase15_evidence_exposes_defensible_pause_fields():
    t = TurnResult(
        turn_id="p15",
        mode_before="idle",
        mode_after="companion",
        asr=AsrResult(text="feel lonely", language="en", avg_logprob=-0.4),
        features=SpeechFeatures(
            latency_ms=1200,
            latency_source="prompt_end_to_first_voiced_estimated",
            audio_duration_ms=5000,
            voiced_duration_ms=2000,
            speech_duration_ms=2000,
            unvoiced_total_ms=3000,
            leading_silence_ms=600,
            trailing_silence_ms=1400,
            internal_pause_total_ms=1000,
            internal_pause_count=1,
            mean_internal_pause_ms=1000,
            voice_activity_ratio=0.4,
            speech_rate_cps=5.5,
            clarity_score=0.6,
        ),
        nlu=NluResult(intent="loneliness_expression", intent_confidence=0.97, emotion="sad", emotion_confidence=0.95),
        deviation=BaselineDeviation(max_z=2.1, exceed_count=2, sufficient_history=True),
        decision=FusionDecision(outcome="soft_check_in", reason="speech pattern deviates"),
        response_text="I hear you.",
        response_lang="en",
        response_audio_path=None,
    )
    ev = turn_to_evidence(t)
    sf = ev["speech_features"]
    assert sf["audio_duration_ms"] == 5000
    assert sf["trailing_silence_ms"] == 1400
    assert sf["internal_pause_total_ms"] == 1000
    assert sf["latency_source"] == "prompt_end_to_first_voiced_estimated"
