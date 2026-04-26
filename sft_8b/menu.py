"""Offer-menu generator: score all 64 splits, return the top-K.

    score(π) = U_self(π) + λ · E_θ ~ posterior [ U_opp(π | θ) ]

with the expectation taken over ``sft_8b.posterior.ORDERINGS`` using
whatever length-6 probability vector the caller hands in. The module
is model-agnostic — it knows nothing about SFT 8B, the 70B hybrid,
or any particular posterior source.

Boundary-condition style axis (see ``data/validation_notes.md`` for the
original re-tuning rationale):
    λ = 0    pure self-maximization (competitive)
    λ = 1    equal weight on self and opponent utility (balanced / integrative)
    λ = 2    2× weight on the opponent (altruistic-limit sanity check)

The original plan used λ ∈ {0.2, 0.5, 0.8} to mirror the Day-1 Q-score
``w`` weights. On the eyeball run (5 held-out dialogues × k=1..5, K=16),
that range collapsed: all three λ values returned essentially the same
"take-everything" menu as top-1 because U_self jumps in 3–5 point steps
(priority-weight differences) while ``λ · E[U_opp]`` at λ ≤ 0.8 is
capped below ~12 points, which cannot outrank even a single-item
increment in U_self. Re-tuned boundary ablations used {0, 1, 2}, where
λ=1 reliably surfaces Pareto-efficient integrative splits as the top entry
and λ=2 produces visibly altruistic menus. SVO-conditioned runs use the
moderate relative weights in ``sft_8b.svo_to_lambda`` instead; λ=2 is too
large for behavioral-fidelity experiments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from sft_8b.posterior import N_ORDERINGS, ORDERINGS
from sft_8b.prompts import ITEMS

PRIORITY_POINTS = {"High": 5, "Medium": 4, "Low": 3}
ITEMS_COUNT = 3  # CaSiNo: 3 packages per item type
MAX_SELF_POINTS = sum(PRIORITY_POINTS.values()) * ITEMS_COUNT  # 5+4+3 × 3 = 36


@dataclass
class ScoredSplit:
    """One candidate split over Food/Water/Firewood + its score."""
    self_counts: Dict[str, int]       # {"Food": 2, "Water": 3, "Firewood": 1}
    opp_counts:  Dict[str, int]
    u_self:      int                  # deterministic: your priorities + counts
    exp_u_opp:   float                # E_θ[U_opp(π | θ)] under the posterior
    score:       float                # U_self + λ · exp_u_opp

    def render(self) -> str:
        s, o = self.self_counts, self.opp_counts
        return (
            f"self=(F{s['Food']} W{s['Water']} Fw{s['Firewood']})  "
            f"opp=(F{o['Food']} W{o['Water']} Fw{o['Firewood']})  "
            f"U_self={self.u_self:>2d}  E[U_opp]={self.exp_u_opp:5.2f}  "
            f"score={self.score:6.2f}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Scoring primitives ─────────────────────────────────────────────────────


def points(counts: Mapping[str, int], priorities: Mapping[str, str]) -> int:
    """CaSiNo per-side points: sum(count × priority-weight)."""
    priority_of = {priorities[level]: level for level in ("High", "Medium", "Low")}
    return sum(int(counts[it]) * PRIORITY_POINTS[priority_of[it]] for it in ITEMS)


def ordering_to_priorities(ordering: Tuple[str, str, str]) -> Dict[str, str]:
    """ordering = (top, mid, low) → {'High': top, 'Medium': mid, 'Low': low}."""
    return {"High": ordering[0], "Medium": ordering[1], "Low": ordering[2]}


# ── Menu builder ───────────────────────────────────────────────────────────


def build_menu(
    posterior: Sequence[float],
    self_priorities: Mapping[str, str],
    *,
    lambda_: float,
    top_k: int = 5,
) -> List[ScoredSplit]:
    """Enumerate all 64 (= 4×4×4) splits and return the top ``top_k``.

    The 4 options per issue are (self=0..3, opp=3..0). The score is
    fully determined by the two inputs here (posterior + self_priorities)
    plus λ, so the same function works on any posterior source: SFT 8B,
    70B hybrid, uniform prior, hand-specified beliefs.
    """
    posterior_arr = np.asarray(posterior, dtype=np.float64)
    if posterior_arr.shape != (N_ORDERINGS,):
        raise ValueError(
            f"posterior must have shape ({N_ORDERINGS},); got {posterior_arr.shape}"
        )
    if posterior_arr.min() < 0:
        raise ValueError("posterior has negative entries")
    total = posterior_arr.sum()
    if total <= 0:
        raise ValueError("posterior has zero total mass")
    # Allow slight drift from sampling noise; renormalize silently.
    if abs(total - 1.0) > 1e-6:
        posterior_arr = posterior_arr / total

    opp_priors_by_idx = [ordering_to_priorities(o) for o in ORDERINGS]

    splits: List[ScoredSplit] = []
    for f, w, fw in product(range(ITEMS_COUNT + 1), repeat=3):
        self_counts = {"Food": f, "Water": w, "Firewood": fw}
        opp_counts  = {it: ITEMS_COUNT - self_counts[it] for it in ITEMS}
        u_self = points(self_counts, self_priorities)
        u_opps = np.array(
            [points(opp_counts, prio) for prio in opp_priors_by_idx],
            dtype=np.float64,
        )
        exp_u_opp = float(posterior_arr @ u_opps)
        splits.append(ScoredSplit(
            self_counts=self_counts,
            opp_counts=opp_counts,
            u_self=u_self,
            exp_u_opp=exp_u_opp,
            score=float(u_self) + float(lambda_) * exp_u_opp,
        ))
    splits.sort(key=lambda s: -s.score)
    return splits[: int(top_k)]


def format_menu(menu: Sequence[ScoredSplit], *, indent: str = "  ") -> str:
    return "\n".join(
        f"{indent}#{i + 1}  {s.render()}" for i, s in enumerate(menu)
    )


__all__ = [
    "ScoredSplit",
    "build_menu",
    "format_menu",
    "points",
    "ordering_to_priorities",
    "PRIORITY_POINTS",
    "ITEMS_COUNT",
    "MAX_SELF_POINTS",
]
