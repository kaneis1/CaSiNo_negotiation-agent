"""SVO category -> Bayesian menu lambda.

Categories follow the SVO Slider angle bins from Murphy et al. (2011).
CaSiNo's coarse ``proself`` label is treated as individualistic.
"""

from __future__ import annotations

SVO_LAMBDA = {
    "altruistic": 2.0,
    "prosocial": 2.0,
    "individualistic": 1.0,
    "competitive": 0.0,
    "unclassified": 1.0,
}


def svo_to_lambda(svo: str | None) -> float:
    key = str(svo or "unclassified").strip().lower()
    key = "individualistic" if key == "proself" else key
    return SVO_LAMBDA.get(key, SVO_LAMBDA["unclassified"])


__all__ = ["SVO_LAMBDA", "svo_to_lambda"]
