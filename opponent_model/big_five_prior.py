"""Big Five → Dirichlet prior over priority hypotheses.

This is a STUB for Contribution 1 (offline-fit Dirichlet from a Big Five
regression). It exposes the same interface the real fitted model will:

    log_prior = big_five_prior(big_five_dict)  # shape (6,), log space

so the rest of the pipeline can be developed against it.

Until the regression is trained, ``big_five_prior`` returns a uniform prior
unless explicit Big Five → item bumps are configured. The shape and contract
should not change once the real model is plugged in.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np

from opponent_model.hypotheses import HYPOTHESES, ITEMS

BIG_FIVE_TRAITS = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)


def _validate_big_five(big_five: Mapping[str, float]) -> Dict[str, float]:
    """Normalize Big Five dict keys (case-insensitive) and check coverage."""
    normalized = {k.lower(): float(v) for k, v in big_five.items()}
    missing = [t for t in BIG_FIVE_TRAITS if t not in normalized]
    if missing:
        raise ValueError(
            f"Missing Big Five traits: {missing}. Expected: {BIG_FIVE_TRAITS}."
        )
    return normalized


def big_five_prior(
    big_five: Optional[Mapping[str, float]] = None,
    *,
    concentration: float = 6.0,
    bumps: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> np.ndarray:
    """Return a log-space prior over the 6 hypotheses.

    Args:
        big_five: Optional dict with keys openness, conscientiousness,
            extraversion, agreeableness, neuroticism (case-insensitive).
            Pass ``None`` for a uniform prior.
        concentration: Total pseudo-count of the underlying Dirichlet.
            Larger ``concentration`` ⇒ stronger prior. Default 6.0 keeps the
            stub gentle (one pseudo-count per hypothesis on uniform).
        bumps: Optional override mapping ``trait → {item → weight}``. The
            real regression will populate this from offline-fit coefficients.
            For now, ``None`` falls back to a uniform Dirichlet.

    Returns:
        np.ndarray of shape (6,) in log space, normalized to sum-exp = 1.
    """
    if big_five is None or bumps is None:
        return np.log(np.ones(len(HYPOTHESES)) / len(HYPOTHESES))

    traits = _validate_big_five(big_five)

    item_score: Dict[str, float] = {item: 0.0 for item in ITEMS}
    for trait, item_weights in bumps.items():
        trait_value = traits.get(trait.lower())
        if trait_value is None:
            continue
        for item, weight in item_weights.items():
            if item not in item_score:
                continue
            item_score[item] += float(weight) * trait_value

    item_alpha = {
        item: max(np.exp(score), 1e-3) for item, score in item_score.items()
    }
    total = sum(item_alpha.values())
    item_alpha = {k: v / total for k, v in item_alpha.items()}

    pseudo_counts = np.zeros(len(HYPOTHESES))
    for i, hypothesis in enumerate(HYPOTHESES):
        weight = (
            3.0 * item_alpha[hypothesis[0]]
            + 2.0 * item_alpha[hypothesis[1]]
            + 1.0 * item_alpha[hypothesis[2]]
        )
        pseudo_counts[i] = weight

    pseudo_counts *= concentration / pseudo_counts.sum()

    prior = pseudo_counts / pseudo_counts.sum()
    return np.log(prior + 1e-12)


__all__ = ["big_five_prior", "BIG_FIVE_TRAITS"]
