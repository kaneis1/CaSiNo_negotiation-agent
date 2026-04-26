"""Big Five traits -> student style token.

Train-locked deterministic branch rule for the Day 9 style smoke. Trait names
follow John & Srivastava (1999), with CaSiNo's exported key aliases accepted.
"""

from __future__ import annotations

from typing import Mapping

COOPERATIVE_AGREEABLENESS_MIN = 6.5
COMPETITIVE_AGREEABLENESS_MAX = 4.5
COMPETITIVE_EXTRAVERSION_MIN = 6.5


def bigfive_to_style(big_five: Mapping[str, float] | None) -> str:
    traits = {str(k).lower(): float(v) for k, v in (big_five or {}).items()}
    agree = traits.get("agreeableness", 4.0)
    extra = traits.get("extraversion", 4.0)
    if agree >= COOPERATIVE_AGREEABLENESS_MIN:
        return "cooperative"
    if (
        agree <= COMPETITIVE_AGREEABLENESS_MAX
        or extra >= COMPETITIVE_EXTRAVERSION_MIN
    ):
        return "competitive"
    return "balanced"


__all__ = [
    "COOPERATIVE_AGREEABLENESS_MIN",
    "COMPETITIVE_AGREEABLENESS_MAX",
    "COMPETITIVE_EXTRAVERSION_MIN",
    "bigfive_to_style",
]
