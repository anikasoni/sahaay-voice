"""Natural Language Understanding.

Zero-shot multilingual classification using XLM-R/NLI by default — works
out-of-the-box, no labeled data needed (TRD §3.2 lists this as the fallback
embedding-model path; we promote it to default given limited time).

A fine-tuning pipeline for MuRIL/IndicBERT is provided in scripts/finetune_muril.py.
Once you have a trained checkpoint at models/muril_intent/, flip
`USE_FINETUNED=True` in this file or set the env var SAHAAY_FINETUNED=1.

Slot filling: regex-based, sufficient for the prototype's controlled domain.
"""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Optional

from core.config import intents
from core.logger import log
from core.schemas import Emotion, NluResult, NluSlots

USE_FINETUNED = os.getenv("SAHAAY_FINETUNED", "0") == "1"
FINETUNED_PATH = Path(__file__).resolve().parent.parent / "models" / "muril_intent"


# --- zero-shot classifier ---------------------------------------------------

class _ZeroShotEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pipe = None
        self._intent_labels: list[str] = []
        self._intent_hypotheses: dict[str, str] = {}
        self._emotion_labels: list[str] = []
        self._emotion_hypotheses: dict[str, str] = {}
        self._build_label_space()

    def _build_label_space(self) -> None:
        cfg = intents()
        for name, spec in cfg["intents"].items():
            self._intent_labels.append(name)
            # English hypothesis works across languages with XLM-R NLI models
            self._intent_hypotheses[name] = f"This text means: {spec['description']}."
        for name, desc in cfg["emotions"].items():
            self._emotion_labels.append(name)
            self._emotion_hypotheses[name] = desc

    def _load(self):
        from transformers import pipeline  # heavy import
        log("loading zero-shot NLI model (XLM-R)…")
        # joeddav/xlm-roberta-large-xnli — multilingual, supports en/hi/pa
        return pipeline(
            "zero-shot-classification",
            model="joeddav/xlm-roberta-large-xnli",
            device=0 if _cuda_ok() else -1,
        )

    def get(self):
        if self._pipe is None:
            with self._lock:
                if self._pipe is None:
                    self._pipe = self._load()
        return self._pipe

    def classify_intent(self, text: str) -> tuple[str, float]:
        if not text.strip():
            return "casual_chat", 0.0
        pipe = self.get()
        out = pipe(
            text,
            candidate_labels=list(self._intent_hypotheses.values()),
            multi_label=False,
        )
        # map best hypothesis back to intent name
        best_hyp = out["labels"][0]
        score = float(out["scores"][0])
        inv = {v: k for k, v in self._intent_hypotheses.items()}
        return inv.get(best_hyp, "casual_chat"), score

    def classify_emotion(self, text: str) -> tuple[Emotion, float]:
        if not text.strip():
            return "calm", 0.0
        pipe = self.get()
        out = pipe(
            text,
            candidate_labels=list(self._emotion_hypotheses.values()),
            multi_label=False,
        )
        best_hyp = out["labels"][0]
        score = float(out["scores"][0])
        inv = {v: k for k, v in self._emotion_hypotheses.items()}
        return inv.get(best_hyp, "calm"), score  # type: ignore[return-value]


def _cuda_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# --- fine-tuned MuRIL classifier (loaded only if checkpoint exists) ---------

class _FinetunedEngine:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._pipe = None
        self._labels: list[str] = []

    def _load(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        log(f"loading fine-tuned MuRIL from {self.path}…")
        tok = AutoTokenizer.from_pretrained(str(self.path))
        mdl = AutoModelForSequenceClassification.from_pretrained(str(self.path))
        return pipeline("text-classification", model=mdl, tokenizer=tok,
                        device=0 if _cuda_ok() else -1, return_all_scores=False)

    def get(self):
        if self._pipe is None:
            with self._lock:
                if self._pipe is None:
                    self._pipe = self._load()
        return self._pipe

    def classify_intent(self, text: str) -> tuple[str, float]:
        if not text.strip():
            return "casual_chat", 0.0
        out = self.get()(text)[0]
        return out["label"], float(out["score"])


_zero_shot = _ZeroShotEngine()
_finetuned: Optional[_FinetunedEngine] = None
if USE_FINETUNED and FINETUNED_PATH.exists():
    _finetuned = _FinetunedEngine(FINETUNED_PATH)


def warm_up() -> None:
    _zero_shot.get()
    if _finetuned:
        _finetuned.get()
    log("NLU warm-up complete")


# --- deterministic domain shortcuts ----------------------------------------
# These fast paths cover the PRD acceptance flows and reduce demo latency.
# The zero-shot model remains the fallback for free-form utterances.

_KEYWORD_PATTERNS = {
    "self_harm_risk": [
        r"\bwant to die\b", r"\bkill myself\b", r"\bend my life\b", r"\bsuicide\b",
        r"jeena nahi", r"marna chaht", r"marna chaund", r"khud ko maar", r"apne aap nu maar",
    ],
    "clinical_question": [
        r"having a stroke", r"am i having.*stroke", r"heart attack", r"do i have (cancer|diabetes)",
        r"stroke ho", r"stroke ho raha", r"stroke ho riha", r"heart attack hai",
        r"mujhe (kya|ky[au]) (bimari|disease)", r"diagnos", r"medical advice",
    ],
    "emergency_help": [
        r"\bhelp\b", r"\bbachao\b", r"\bfall(en)?\b", r"gir gay[ai]", r"dig gay[ai]",
        r"can't move", r"cannot move", r"hil nahi", r"hill nahi", r"madad", r"call.*ambulance",
    ],
    "safety_confirmation": [
        r"\bi am okay\b", r"\bi'?m okay\b", r"\bi am fine\b", r"\bi am alright\b",
        r"main theek hoon", r"main thik hoon", r"main theek haan", r"main thik haan",
        r"sab theek", r"sab thik", r"\bok+\b", r"\ball good\b",
    ],
    "task_acknowledgement": [
        r"\bdone\b", r"\bfinished\b", r"\btaken it\b", r"i took", r"already took",
        r"ho gaya", r"kar diya", r"kar ditta", r"le li", r"lai layi", r"le layi", r"dawai le", r"medicine le",
    ],
    "loneliness_expression": [
        r"lonely", r"alone", r"miss my", r"no one (comes|visits|talks)",
        r"akela", r"akeli", r"ikalla", r"ikalli", r"yaad aati", r"yaad aundi",
    ],
    "confusion_or_disorientation": [
        r"where am i", r"what day", r"who are you", r"i forgot", r"confused",
        r"kahan hoon", r"kithe haan", r"kaunsa din", r"kehra din", r"bhool gaya", r"bhul gaya",
    ],
    "caregiver_request": [
        r"call my", r"speak to my", r"where is my (son|daughter|family|wife|husband)",
        r"bete ko", r"beti se", r"parivaar", r"putt nu", r"dhee naal", r"family ko",
    ],
    "medicine_query": [
        r"medicine", r"pill", r"tablet", r"bp", r"blood pressure", r"insulin", r"diabetes",
        r"dawai", r"goli", r"sugar ki", r"sugar di",
    ],
    "reminder_request": [
        r"remind me", r"set a reminder", r"yaad dila", r"yaad kara", r"reminder",
    ],
    "casual_chat": [
        r"^hello+$", r"^hi+$", r"good morning", r"how are you", r"namaste", r"sat sri akal",
    ],
}

_INTENT_PRIORITY = [
    "self_harm_risk",
    "clinical_question",
    "emergency_help",
    "safety_confirmation",
    "task_acknowledgement",
    "caregiver_request",
    "loneliness_expression",
    "confusion_or_disorientation",
    "medicine_query",
    "reminder_request",
    "casual_chat",
]


def _keyword_intent(text: str) -> Optional[str]:
    t = text.lower().strip()
    for intent_name in _INTENT_PRIORITY:
        for pat in _KEYWORD_PATTERNS.get(intent_name, []):
            if re.search(pat, t):
                return intent_name
    return None


def is_task_acknowledgement(text: str) -> bool:
    """Shared helper for Memory Mode acknowledgement checks."""
    t = text.lower().strip()
    return any(re.search(pat, t) for pat in _KEYWORD_PATTERNS["task_acknowledgement"] + _KEYWORD_PATTERNS["safety_confirmation"])


def contains_self_harm_language(text: str) -> bool:
    t = text.lower().strip()
    return any(re.search(pat, t) for pat in _KEYWORD_PATTERNS["self_harm_risk"])


# --- slot filling -----------------------------------------------------------

_TIME_RE = re.compile(
    r"\b(?:(\d{1,2})\s*(?:[:.]\s*(\d{2}))?\s*(am|pm|baje|vaje)?)\b",
    re.IGNORECASE,
)
_MEDS = [
    "blood pressure", "bp", "sugar", "diabetes", "insulin", "thyroid",
    "dawai", "goli", "tablet", "pill", "medicine",
]
_PEOPLE = [
    "son", "daughter", "wife", "husband", "doctor", "beta", "beti", "putt",
    "dhee", "patni", "pati", "pita", "maa",
]


def _extract_slots(text: str) -> NluSlots:
    t = text.lower()
    slots = NluSlots()
    m = _TIME_RE.search(t)
    if m:
        slots.time = m.group(0).strip()
    for med in _MEDS:
        if med in t:
            slots.medicine_name = med
            break
    for person in _PEOPLE:
        if re.search(rf"\b{person}\b", t):
            slots.person = person
            break
    return slots


# --- public API -------------------------------------------------------------

def analyze(text: str) -> NluResult:
    """Run intent + emotion + slot extraction on a transcript."""
    text = (text or "").strip()
    if not text:
        return NluResult(intent="casual_chat", intent_confidence=0.0, emotion="calm")

    # 1) deterministic fast path for domain-critical intents.
    # This keeps acceptance scenarios reliable and avoids using the heavy NLI
    # model for obvious safety/reminder phrases.
    kw_intent = _keyword_intent(text)
    if kw_intent is not None:
        intent, conf = kw_intent, 0.97
        # Fast-path emotion avoids loading the NLI model for obvious demo flows.
        if intent in ("self_harm_risk", "loneliness_expression"):
            emotion, emo_conf = "sad", 0.95
        elif intent in ("clinical_question", "emergency_help"):
            emotion, emo_conf = "anxious", 0.95
        elif intent == "confusion_or_disorientation":
            emotion, emo_conf = "confused", 0.95
        else:
            emotion, emo_conf = "calm", 0.90
    elif _finetuned is not None:
        intent, conf = _finetuned.classify_intent(text)
        emotion, emo_conf = _zero_shot.classify_emotion(text)
    else:
        intent, conf = _zero_shot.classify_intent(text)
        emotion, emo_conf = _zero_shot.classify_emotion(text)
    slots = _extract_slots(text)

    log("NLU", intent=intent, conf=round(conf, 3), emotion=emotion)
    return NluResult(
        intent=intent,
        intent_confidence=float(round(conf, 3)),
        emotion=emotion,
        emotion_confidence=float(round(emo_conf, 3)),
        slots=slots,
    )


def runtime_info() -> dict[str, object]:
    """Display-only NLU runtime metadata for the AI evidence panel."""
    engine = "fine-tuned MuRIL/IndicBERT" if _finetuned is not None else "domain rules + zero-shot XLM-R/XNLI"
    return {
        "stage": "NLU",
        "engine": engine,
        "keyword_fast_paths": True,
        "zero_shot_model": "joeddav/xlm-roberta-large-xnli",
        "zero_shot_loaded": _zero_shot._pipe is not None,
        "finetuned_requested": USE_FINETUNED,
        "finetuned_checkpoint": str(FINETUNED_PATH),
        "finetuned_loaded": _finetuned is not None,
        "intent_count": len(_zero_shot._intent_labels),
        "emotion_count": len(_zero_shot._emotion_labels),
        "slot_filling": "regex: medicine/time/person/routine placeholders",
    }
