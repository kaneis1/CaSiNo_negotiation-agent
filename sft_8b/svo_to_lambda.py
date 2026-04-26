"""SVO category -> Bayesian menu lambda.

Categories follow the SVO Slider angle bins from Murphy et al. (2011).
CaSiNo's coarse ``proself`` label is treated as individualistic.

The menu score is additive, not convex:

    score(pi) = U_self(pi) + lambda * E[U_opp(pi | theta)]

So lambda is an opponent-utility weight relative to self utility. The default
``rescaled`` mapping uses moderate human-SVO weights. The
``legacy_boundary`` mapping preserves the original Day-9/Day-10 job with
lambda in {0, 1, 2}; keep it for ablations and apples-to-apples comparison,
but treat lambda=2 as an altruistic boundary condition rather than a realistic
human setting. ``*_swapped`` modes are counterfactual controls for the
match-vs-mismatch SVO analysis: proself/individualistic receives the
prosocial weight, and prosocial/altruistic receives the proself weight.
"""

from __future__ import annotations

RESCALED_SVO_LAMBDA = {
    "altruistic": 0.8,
    "prosocial": 0.6,
    "individualistic": 0.2,
    "competitive": 0.0,
    "unclassified": 0.2,
}

LEGACY_BOUNDARY_SVO_LAMBDA = {
    "altruistic": 2.0,
    "prosocial": 2.0,
    "individualistic": 1.0,
    "competitive": 0.0,
    "unclassified": 1.0,
}

RESCALED_SWAPPED_SVO_LAMBDA = {
    "altruistic": 0.2,
    "prosocial": 0.2,
    "individualistic": 0.6,
    "competitive": 0.6,
    "unclassified": 0.2,
}

LEGACY_BOUNDARY_SWAPPED_SVO_LAMBDA = {
    "altruistic": 1.0,
    "prosocial": 1.0,
    "individualistic": 2.0,
    "competitive": 2.0,
    "unclassified": 1.0,
}

SVO_LAMBDA_MODES = {
    "rescaled": RESCALED_SVO_LAMBDA,
    "legacy_boundary": LEGACY_BOUNDARY_SVO_LAMBDA,
    "boundary": LEGACY_BOUNDARY_SVO_LAMBDA,
    "original": LEGACY_BOUNDARY_SVO_LAMBDA,
    "rescaled_swapped": RESCALED_SWAPPED_SVO_LAMBDA,
    "legacy_boundary_swapped": LEGACY_BOUNDARY_SWAPPED_SVO_LAMBDA,
    "boundary_swapped": LEGACY_BOUNDARY_SWAPPED_SVO_LAMBDA,
    "original_swapped": LEGACY_BOUNDARY_SWAPPED_SVO_LAMBDA,
}

# Backward-compatible name for callers that import the default mapping.
SVO_LAMBDA = RESCALED_SVO_LAMBDA


def svo_to_lambda(svo: str | None, *, mode: str = "rescaled") -> float:
    key = str(svo or "unclassified").strip().lower()
    key = "individualistic" if key == "proself" else key
    mode_key = str(mode or "rescaled").strip().lower()
    if mode_key not in SVO_LAMBDA_MODES:
        raise ValueError(
            f"unknown SVO lambda mode {mode!r}; "
            f"use one of {sorted(SVO_LAMBDA_MODES)}"
        )
    mapping = SVO_LAMBDA_MODES[mode_key]
    return mapping.get(key, mapping["unclassified"])


__all__ = [
    "LEGACY_BOUNDARY_SVO_LAMBDA",
    "LEGACY_BOUNDARY_SWAPPED_SVO_LAMBDA",
    "RESCALED_SVO_LAMBDA",
    "RESCALED_SWAPPED_SVO_LAMBDA",
    "SVO_LAMBDA",
    "SVO_LAMBDA_MODES",
    "svo_to_lambda",
]
