"""Backward-compatible reviser module."""

from app.nodes.reviser_node import repair_recent_examples, reviser_node


def run_reviser(state):
    """Backward-compatible wrapper that mutates and returns full state."""
    state.update(reviser_node(state))
    return state
