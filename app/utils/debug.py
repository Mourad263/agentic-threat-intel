"""Shared debug helpers for pipeline verbosity."""

from __future__ import annotations

from app.utils.app_mode import is_demo_mode


def is_debug_enabled() -> bool:
    """Return True only when demo-mode walkthrough output is enabled."""
    return is_demo_mode()


def debug_print(*args, **kwargs) -> None:
    """Print pipeline progress immediately during execution."""
    if is_debug_enabled():
        print(*args, **kwargs)
