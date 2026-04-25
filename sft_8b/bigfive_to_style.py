"""Big Five traits -> student style token.

Deterministic branch rule for tomorrow's threading smoke; trait names follow
John & Srivastava (1999), with CaSiNo's exported key aliases accepted.
"""

from __future__ import annotations

from typing import Mapping


def bigfive_to_style(big_five: Mapping[str, float] | None) -> str:
    traits = {str(k).lower(): float(v) for k, v in (big_five or {}).items()}
    agree = traits.get("agreeableness", 4.0)
    extra = traits.get("extraversion", 4.0)
    if agree >= 5.0:
        return "cooperative"
    if agree <= 3.0 or extra >= 6.0:
        return "competitive"
    return "balanced"


__all__ = ["bigfive_to_style"]
