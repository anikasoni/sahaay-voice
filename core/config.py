"""Configuration loader.

Loads thresholds, prompts, reminders, and intents from /config YAMLs.
Per TRD §7.4: "Configuration values such as thresholds, baseline window size,
and reminder schedules shall be externalised into configuration files."
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


def _load(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def thresholds() -> dict[str, Any]:
    return _load("thresholds.yaml")


@lru_cache(maxsize=1)
def prompts() -> dict[str, Any]:
    return _load("prompts.yaml")


@lru_cache(maxsize=1)
def reminders() -> dict[str, Any]:
    return _load("reminders.yaml")


@lru_cache(maxsize=1)
def intents() -> dict[str, Any]:
    return _load("intents.yaml")


def reload_all() -> None:
    """Drop cached configs — call after editing YAML files at runtime."""
    thresholds.cache_clear()
    prompts.cache_clear()
    reminders.cache_clear()
    intents.cache_clear()


def get_prompt(category: str, key: str, lang: str = "en") -> str:
    """Look up a localized prompt with safe fallback to English."""
    p = prompts()
    try:
        entry = p[category][key]
    except KeyError:
        return ""
    if isinstance(entry, str):
        return entry
    return entry.get(lang) or entry.get("en") or ""


# Paths used across modules
DATA_DIR = ROOT / "data"
LOG_DIR = DATA_DIR / "logs"
AUDIO_DIR = DATA_DIR / "audio"
DB_PATH = DATA_DIR / "baseline.sqlite"

# Ensure dirs exist at import time
for _d in (DATA_DIR, LOG_DIR, AUDIO_DIR):
    _d.mkdir(parents=True, exist_ok=True)
