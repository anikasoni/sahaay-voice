"""Headless demo runner.

Useful for: smoke-testing on a server without a display, capturing pipeline
outputs for a screencast, running scripted demos for the PRD §10 scenarios.

Usage:
    # Single utterance
    python scripts/run_demo.py samples/clean/hello.wav

    # Full safety-mode scenario: trigger fall, then play a reply
    python scripts/run_demo.py samples/safety_reply.wav --trigger-fall

    # Pre-deliver a reminder, then play the ack
    python scripts/run_demo.py samples/ack.wav --deliver-reminder med-morning-bp

    # Print the report as JSON
    python scripts/run_demo.py samples/hello.wav --json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.audio_capture import utterance_from_wav  # noqa: E402
from core.dialogue import DialogueManager  # noqa: E402


def _pretty(turn) -> str:
    lines = [
        "─" * 60,
        f"turn         : {turn.turn_id}",
        f"mode         : {turn.mode_before} → {turn.mode_after}",
        f"elder said   : {turn.asr.text or '(silent)'}",
        f"detected lang: {turn.asr.language}",
        f"clarity      : {turn.features.clarity_score:.2f}",
        f"speech rate  : {turn.features.speech_rate_cps:.1f} cps",
        f"pauses       : {turn.features.pause_total_ms} ms total ({turn.features.pause_count} pauses)",
        f"nlu          : {turn.nlu.intent} (conf {turn.nlu.intent_confidence:.2f}) | emotion {turn.nlu.emotion}",
        f"deviation    : max_z={turn.deviation.max_z:.2f}, exceed={turn.deviation.exceed_count}, "
        f"sufficient={turn.deviation.sufficient_history}",
        f"fusion       : {turn.decision.outcome} — {turn.decision.reason}",
        f"sahaay says  : {turn.response_text}",
        f"audio reply  : {turn.response_audio_path}",
        "─" * 60,
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path, help="input .wav file (16kHz mono preferred)")
    ap.add_argument("--trigger-fall", action="store_true",
                    help="publish a high-severity fall event before processing")
    ap.add_argument("--deliver-reminder", type=str, default=None,
                    help="reminder_id from config/reminders.yaml to deliver first")
    ap.add_argument("--lang", default=None,
                    help="optional language hint for ASR (en/hi/pa)")
    ap.add_argument("--json", dest="as_json", action="store_true")
    args = ap.parse_args()

    if not args.audio.exists():
        print(f"ERROR: file not found: {args.audio}", file=sys.stderr)
        sys.exit(2)

    mgr = DialogueManager()

    # Optional pre-conditions
    if args.trigger_fall:
        from core import wearable as wearable_mod
        wearable_mod.publish("fall_detected", "high")
        text, _ = mgr.enter_safety_mode(lang=args.lang or "en")  # type: ignore[arg-type]
        if not args.as_json:
            print(f"[setup] wearable=fall_detected high; entered Safety Mode")
            print(f"[setup] sahaay asked: {text}")

    if args.deliver_reminder:
        from core.config import reminders
        rems = reminders().get("reminders", [])
        match = next((r for r in rems if r["reminder_id"] == args.deliver_reminder), None)
        if not match:
            print(f"ERROR: no reminder with id={args.deliver_reminder}", file=sys.stderr)
            sys.exit(2)
        label_field = {"en": "label", "hi": "label_hi", "pa": "label_pa"}[match["language"]]
        label = match.get(label_field) or match["label"]
        text, _ = mgr.deliver_reminder(match["reminder_id"], label, match["language"])
        if not args.as_json:
            print(f"[setup] delivered reminder {match['reminder_id']}")
            print(f"[setup] sahaay said: {text}")

    utt = utterance_from_wav(args.audio)
    turn = mgr.handle_utterance(utt, audio_path=args.audio)

    if args.as_json:
        out = {
            "turn_id": turn.turn_id,
            "mode_before": turn.mode_before,
            "mode_after": turn.mode_after,
            "asr": asdict(turn.asr),
            "features": asdict(turn.features),
            "nlu": asdict(turn.nlu),
            "deviation": asdict(turn.deviation),
            "decision": asdict(turn.decision),
            "response_text": turn.response_text,
            "response_lang": turn.response_lang,
            "response_audio_path": str(turn.response_audio_path) if turn.response_audio_path else None,
        }
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(_pretty(turn))


if __name__ == "__main__":
    main()
