"""Helpers for loading prompt text files with safe fallbacks."""

from __future__ import annotations

from pathlib import Path

_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


def load_prompt(filename: str, fallback: str) -> str:
    """Return prompt file contents, or the fallback when the file is missing."""
    prompt_path = _PROMPT_DIR / filename
    try:
        prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return fallback.strip()

    return prompt_text or fallback.strip()
