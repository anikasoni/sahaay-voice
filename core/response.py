"""Response generator.

Per TRD §4.9: template-and-rules-based response selection per mode + intent.
A constrained generative fallback is allowed in Companion Mode only — gated
behind config and DISABLED IN SAFETY MODE (TRD §9, hard rule).
"""
from __future__ import annotations

import re
from typing import Optional

import requests

from core.config import get_prompt, thresholds
from core.logger import log
from core.schemas import Language, NluResult

# --- self-harm guardrails ---------------------------------------------------

def contains_self_harm(text: str, lang: Language) -> bool:
    # Combine configurable keyword lists with NLU regex guardrails.
    try:
        from core.nlu import contains_self_harm_language
        if contains_self_harm_language(text):
            return True
    except Exception:
        pass
    cfg = thresholds()["companion_mode"]
    key = {"en": "self_harm_keywords_en",
           "hi": "self_harm_keywords_hi",
           "pa": "self_harm_keywords_pa"}.get(lang, "self_harm_keywords_en")
    kws = cfg.get(key, []) + cfg.get("self_harm_keywords_en", [])
    t = text.lower()
    return any(k.lower() in t for k in kws)


# --- per-mode response selectors -------------------------------------------

def safety_response(stage: str, lang: Language) -> str:
    """stage ∈ {ask, repeat, confirmed, escalated}."""
    mapping = {
        "ask": ("safety", "ask_confirmation"),
        "repeat": ("safety", "ask_repeat"),
        "confirmed": ("safety", "confirmed_ok"),
        "escalated": ("safety", "escalated"),
    }
    cat, key = mapping[stage]
    return get_prompt(cat, key, lang)


def memory_response(stage: str, lang: Language, label: str = "") -> str:
    if stage == "deliver":
        prefix = get_prompt("memory", "reminder_prefix", lang)
        ack = get_prompt("memory", "ask_ack", lang)
        return f"{prefix} {label}. {ack}".strip()
    if stage == "ack":
        return get_prompt("memory", "ack_received", lang)
    if stage == "repeat":
        prefix = get_prompt("memory", "repeat_reminder", lang)
        return f"{prefix} {label}".strip()
    return ""


def clinical_deflection(lang: Language) -> str:
    return get_prompt("clinical_deflection", "", lang) or \
        get_prompt("clinical_deflection", "en", lang)  # robustness


def soft_checkin_response(lang: Language) -> str:
    """Gentle baseline-deviation check-in. May start a safety confirmation loop."""
    return get_prompt("safety", "soft_checkin", lang)


def companion_soft_checkin(lang: Language) -> str:
    """Combined response for loneliness/sadness + speech deviation.

    This preserves dignity: first companionship, then a low-alarm wellness check.
    It does not immediately notify the caregiver and does not diagnose.
    """
    custom = get_prompt("companion", "empathetic_soft_checkin", lang)
    if custom:
        return custom
    return f"{get_prompt('companion', 'empathetic_lonely', lang)} {get_prompt('safety', 'soft_checkin', lang)}"


def asr_error(lang: Language) -> str:
    return get_prompt("error", "asr_failed", lang)


# --- Companion Mode: templates + optional Ollama ---------------------------

def _template_companion(nlu: NluResult, lang: Language) -> str:
    # Map intent + emotion to a soft, dignified template.
    if contains_self_harm(getattr(nlu, "_text", "") or "", lang):
        return get_prompt("companion", "self_harm_response", lang)

    if nlu.intent == "loneliness_expression" or nlu.emotion == "sad":
        return get_prompt("companion", "empathetic_lonely", lang)
    if nlu.intent == "caregiver_request":
        return get_prompt("companion", "acknowledge", lang)
    if nlu.intent == "casual_chat" and nlu.emotion == "calm":
        return get_prompt("companion", "positive_engagement", lang)
    if nlu.intent == "confusion_or_disorientation":
        return get_prompt("companion", "fallback", lang)
    return get_prompt("companion", "acknowledge", lang)


_SYSTEM_PROMPT = (
    "You are Sahaay Voice, a warm, respectful voice companion for an elderly user.\n"
    "RULES (NEVER BREAK):\n"
    "- Reply in ONE or TWO short sentences only.\n"
    "- Match the user's language: English, Hindi, or Punjabi.\n"
    "- NEVER diagnose, NEVER give medical advice or dosages.\n"
    "- NEVER argue with the user, correct memories, or contradict.\n"
    "- If you sense self-harm or severe distress, reply gently and say a family member is being told.\n"
    "- Tone: like a kind, patient grandchild. No emojis. No jargon."
)


def _ollama_companion(user_text: str, lang: Language) -> Optional[str]:
    cfg = thresholds()["companion_mode"]
    if not cfg.get("use_local_llm"):
        return None
    try:
        r = requests.post(
            cfg["llm_endpoint"],
            json={
                "model": cfg["llm_model"],
                "prompt": (f"{_SYSTEM_PROMPT}\n\nUser ({lang}): {user_text}\nAssistant:"),
                "stream": False,
                "options": {"temperature": 0.4, "num_predict": 80},
            },
            timeout=8,
        )
        r.raise_for_status()
        text = (r.json().get("response") or "").strip()
        # strip trailing system-prompt echoes if any
        text = re.split(r"\n(?:User|System):", text, maxsplit=1)[0].strip()
        return text or None
    except Exception as e:
        log(f"Ollama fallback failed: {e}", level="warning")
        return None


def companion_response(nlu: NluResult, user_text: str, lang: Language) -> str:
    # Hard guardrail FIRST — never let an LLM see self-harm content unfiltered
    if contains_self_harm(user_text, lang):
        return get_prompt("companion", "self_harm_response", lang)

    # Try local LLM if enabled
    llm = _ollama_companion(user_text, lang)
    if llm:
        return llm

    # Stash text for the template path
    setattr(nlu, "_text", user_text)
    return _template_companion(nlu, lang)
