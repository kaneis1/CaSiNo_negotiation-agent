"""Hypothesis space: the 3 items and 6 strict orderings over them."""

from __future__ import annotations

from itertools import permutations
from typing import List, Tuple

ITEMS: Tuple[str, str, str] = ("Food", "Water", "Firewood")

HYPOTHESES: List[Tuple[str, str, str]] = list(permutations(ITEMS))


def hypothesis_label(idx: int) -> str:
    """Return ``"H{idx+1}"`` for use in prompts (1-indexed for readability)."""
    return f"H{idx + 1}"


def hypothesis_text(h: Tuple[str, str, str]) -> str:
    """Render a hypothesis as ``"Food > Water > Firewood"``."""
    return " > ".join(h)


def list_hypotheses_for_prompt() -> str:
    """Render the 6 hypotheses as numbered lines for an LLM prompt."""
    return "\n".join(
        f"{hypothesis_label(i)}: {hypothesis_text(h)}"
        for i, h in enumerate(HYPOTHESES)
    )
