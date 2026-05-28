"""Synthetic PRD acceptance checker for Sahaay Voice.

This does NOT load Whisper or the zero-shot NLI model. It patches ASR/NLU/TTS so
that the dialogue manager, fusion rules, mode transitions, caregiver alerts,
non-diagnostic boundary, and memory/safety loops can be tested quickly.

Run:
    python scripts/acceptance_check.py

Output:
    data/acceptance_report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.audio_capture import SAMPLE_RATE, Utterance
from core.dialogue import DialogueManager
from core.schemas import AsrResult, NluResult
from core import baseline as baseline_mod, alerts as alerts_mod, wearable as wearable_mod


def _utt() -> Utterance:
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    audio = (0.25 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return Utterance(audio=audio, sample_rate=SAMPLE_RATE, start_ms=0, end_ms=1000)


def _asr(text: str, lang: str = "en", logprob: float = -0.2):
    return AsrResult(text=text, language=lang, avg_logprob=logprob)


def _nlu(intent: str, emotion: str = "calm"):
    return NluResult(intent=intent, intent_confidence=0.97, emotion=emotion)


def run() -> dict:
    alerts_mod.clear_all()
    baseline_mod.reset()
    wearable_mod.clear()
    rows = []

    def record(name: str, ok: bool, got: str, expected: str, note: str = ""):
        rows.append({"name": name, "pass": bool(ok), "got": got, "expected": expected, "note": note})

    # A1: Safety check clear reply
    mgr = DialogueManager()
    with patch("core.dialogue.tts_mod.synthesize", return_value=None):
        mgr.enter_safety_mode("en")
    with patch("core.dialogue.asr_mod.transcribe", return_value=_asr("i am okay")), \
         patch("core.dialogue.nlu_mod.analyze", return_value=_nlu("safety_confirmation")), \
         patch("core.dialogue.tts_mod.synthesize", return_value=None):
        r = mgr.handle_utterance(_utt())
    record("A1 safety clear reply closes loop", r.mode_after == "idle" and r.decision.outcome == "normal_response", f"{r.mode_after}/{r.decision.outcome}", "idle/normal_response")

    # A2: Safety no reply escalates
    mgr = DialogueManager()
    with patch("core.dialogue.tts_mod.synthesize", return_value=None):
        mgr.enter_safety_mode("en")
        r = mgr.handle_no_reply("timeout")
    record("A2 safety no reply escalates", r.decision.outcome == "caregiver_alert", r.decision.outcome, "caregiver_alert")

    # A3: Reminder acknowledged
    mgr = DialogueManager()
    with patch("core.dialogue.tts_mod.synthesize", return_value=None):
        mgr.deliver_reminder("med", "blood pressure medicine", "en")
    with patch("core.dialogue.asr_mod.transcribe", return_value=_asr("done")), \
         patch("core.dialogue.nlu_mod.analyze", return_value=_nlu("task_acknowledgement")), \
         patch("core.dialogue.tts_mod.synthesize", return_value=None):
        r = mgr.handle_utterance(_utt())
    record("A3 reminder acknowledged", r.mode_after == "idle" and mgr.state.active_reminder_id is None, r.mode_after, "idle")

    # A4: Reminder missed repeats then escalates
    mgr = DialogueManager()
    with patch("core.dialogue.tts_mod.synthesize", return_value=None):
        mgr.deliver_reminder("med", "blood pressure medicine", "en")
        r1 = mgr.handle_no_reply("timeout1")
        r2 = mgr.handle_no_reply("timeout2")
        r3 = mgr.handle_no_reply("timeout3")
    ok = r1.decision.outcome == "reminder_repeat" and r2.decision.outcome == "reminder_repeat" and r3.decision.outcome == "caregiver_alert"
    record("A4 reminder repeats then escalates", ok, f"{r1.decision.outcome},{r2.decision.outcome},{r3.decision.outcome}", "reminder_repeat,reminder_repeat,caregiver_alert")

    # A5: Companion loneliness
    mgr = DialogueManager()
    with patch("core.dialogue.asr_mod.transcribe", return_value=_asr("I feel lonely today")), \
         patch("core.dialogue.nlu_mod.analyze", return_value=_nlu("loneliness_expression", "sad")), \
         patch("core.dialogue.tts_mod.synthesize", return_value=None):
        r = mgr.handle_utterance(_utt())
    record("A5 loneliness gives companion response", r.mode_after == "companion" and "alone" in r.response_text.lower(), r.response_text, "empathetic companion reply")

    # A6: Clinical question deflection
    mgr = DialogueManager()
    with patch("core.dialogue.asr_mod.transcribe", return_value=_asr("am I having a stroke")), \
         patch("core.dialogue.nlu_mod.analyze", return_value=_nlu("clinical_question", "anxious")), \
         patch("core.dialogue.tts_mod.synthesize", return_value=None):
        r = mgr.handle_utterance(_utt())
    ok = r.decision.outcome == "safe_deflection" and "stroke" not in r.response_text.lower()
    record("A6 clinical prompt is non-diagnostic", ok, r.response_text, "safe deflection without diagnosis")

    # A8: Self-harm language alerts caregiver
    mgr = DialogueManager()
    with patch("core.dialogue.asr_mod.transcribe", return_value=_asr("I want to die")), \
         patch("core.dialogue.nlu_mod.analyze", return_value=_nlu("self_harm_risk", "sad")), \
         patch("core.dialogue.tts_mod.synthesize", return_value=None):
        r = mgr.handle_utterance(_utt())
    record("A8 self-harm escalates", r.decision.outcome == "caregiver_alert" and "family" in r.response_text.lower(), f"{r.decision.outcome}: {r.response_text}", "caregiver_alert + gentle response")

    # A9: code-mixed input routes normally
    mgr = DialogueManager()
    with patch("core.dialogue.asr_mod.transcribe", return_value=_asr("main bahut tired hoon today", "mixed")), \
         patch("core.dialogue.nlu_mod.analyze", return_value=_nlu("casual_chat", "calm")), \
         patch("core.dialogue.tts_mod.synthesize", return_value=None):
        r = mgr.handle_utterance(_utt())
    record("A9 code-mixed input processed", r.decision.outcome == "normal_response", r.decision.outcome, "normal_response")

    passed = sum(1 for x in rows if x["pass"])
    report = {"n": len(rows), "passed": passed, "failed": len(rows)-passed, "pass_rate": round(passed / len(rows), 3), "rows": rows}
    return report


if __name__ == "__main__":
    report = run()
    out = ROOT / "data" / "acceptance_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for row in report["rows"]:
        print(("✓" if row["pass"] else "✗"), row["name"], "→", row["got"])
    print(f"\n{report['passed']}/{report['n']} passed ({report['pass_rate']*100:.0f}%). Report: {out}")
    raise SystemExit(0 if report["failed"] == 0 else 1)
