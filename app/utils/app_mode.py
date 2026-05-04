"""Helpers for resolving the app output mode."""

from __future__ import annotations

import os

DEMO_MODE = "demo"


def get_app_mode() -> str:
    """Return the configured mode, defaulting to the quiet normal mode."""
    return os.getenv("APP_MODE", "").strip().lower()


def is_demo_mode() -> bool:
    """Return True when the app should render the readable demo walkthrough."""
    return get_app_mode() == DEMO_MODE


def is_normal_mode() -> bool:
    """Return True when only final output should be shown."""
    return not is_demo_mode()
