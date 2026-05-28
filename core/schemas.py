"""Shared dataclass schemas.

Mirror the TRD-specified output shapes so every module agrees on contracts.
See TRD §4.2.2 (Asr), §4.3.2 (Features), §4.4.3 (NLU), §5.x (logs/alerts).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Language = Literal["en", "hi", "pa", "mixed"]
Mode = Literal["idle", "safety", "companion", "memory"]
Emotion = Literal["calm", "sad", "anxious", "confused"]
WearableEventType = Literal["fall_detected", "inactivity_long", "abnormal_motion", "normal"]
WearableSeverity = Literal["low", "medium", "high"]
FusionOutcome = Literal[
    "normal_response",
    "soft_check_in",
    "reminder_repeat",
    "caregiver_alert",
    "safe_deflection",
]


@dataclass
class AsrSegment:
    start: float
    end: float
    text: str
    confidence: float = 0.0


@dataclass
class AsrResult:
    text: str
    language: Language
    segments: list[AsrSegment] = field(default_factory=list)
    avg_logprob: float = 0.0

    def is_empty(self) -> bool:
        return not self.text or not self.text.strip()


@dataclass
class SpeechFeatures:
    # Prompt-response timing. This is estimated in Streamlit push-to-talk mode
    # as: submit_time - prompt_end_time - recording_duration + leading_silence.
    latency_ms: int = 0
    latency_source: str = "not_measured"
    latency_status: str = "unknown"

    # Backward-compatible fields used by the baseline/fusion layer. In Phase 1.5,
    # speech_duration_ms = voiced_duration_ms and pause_total_ms = internal pauses
    # only, not leading/trailing silence.
    speech_duration_ms: int = 0
    pause_total_ms: int = 0
    pause_count: int = 0

    # Auditable speech-processing fields for explaining the above values.
    audio_duration_ms: int = 0
    voiced_duration_ms: int = 0
    unvoiced_total_ms: int = 0
    leading_silence_ms: int = 0
    trailing_silence_ms: int = 0
    internal_pause_total_ms: int = 0
    internal_pause_count: int = 0
    mean_internal_pause_ms: float = 0.0
    voice_activity_ratio: float = 0.0
    pause_definition: str = "internal_unvoiced_between_first_and_last_voiced"

    speech_rate_cps: float = 0.0     # characters/sec over voiced duration
    clarity_score: float = 0.0       # 0..1
    repetition_score: float = 0.0    # 0..1 (Safety Mode only)
    asr_avg_logprob: float = 0.0

    def as_baseline_features(self) -> dict[str, float]:
        """Subset of features tracked in the rolling baseline.

        We intentionally baseline only interpretable, stable features. Total
        leading/trailing silence is excluded because it can be affected by how
        long the user leaves the recorder open.
        """
        return {
            "latency_ms": float(self.latency_ms),
            "speech_rate_cps": float(self.speech_rate_cps),
            "pause_total_ms": float(self.pause_total_ms),
            "clarity_score": float(self.clarity_score),
            "repetition_score": float(self.repetition_score),
        }


@dataclass
class NluSlots:
    medicine_name: Optional[str] = None
    time: Optional[str] = None
    person: Optional[str] = None
    routine: Optional[str] = None


@dataclass
class NluResult:
    intent: str
    intent_confidence: float
    emotion: Emotion
    emotion_confidence: float = 0.0
    slots: NluSlots = field(default_factory=NluSlots)


@dataclass
class WearableEvent:
    type: WearableEventType
    severity: WearableSeverity
    timestamp_ms: int
    age_ms: int = 0


@dataclass
class BaselineDeviation:
    max_z: float = 0.0
    exceed_count: int = 0
    per_feature_z: dict[str, float] = field(default_factory=dict)
    sufficient_history: bool = False


@dataclass
class FusionDecision:
    outcome: FusionOutcome
    reason: str
    severity: Literal["info", "warning", "urgent"] = "info"
    notify_caregiver: bool = False


@dataclass
class CaregiverAlert:
    alert_id: str
    timestamp: str
    reason: str
    severity: Literal["info", "warning", "urgent"]
    recent_transcript: list[str] = field(default_factory=list)
    related_turn_ids: list[str] = field(default_factory=list)
