"""Scenario-specific DND allocation menu scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from sft_8b.dnd_data import DNDRecord, DND_ITEMS, ordering_to_values_543


@dataclass
class DNDScoredAllocation:
    self_counts: Dict[str, int]
    opp_counts: Dict[str, int]
    u_self: float
    exp_u_opp: float
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def self_tuple(self) -> Tuple[int, int, int]:
        return tuple(int(self.self_counts[it]) for it in DND_ITEMS)  # type: ignore[return-value]

    @property
    def opp_tuple(self) -> Tuple[int, int, int]:
        return tuple(int(self.opp_counts[it]) for it in DND_ITEMS)  # type: ignore[return-value]


def utility(counts: Sequence[int] | Mapping[str, int], values: Sequence[float]) -> float:
    if isinstance(counts, Mapping):
        arr = [int(counts[it]) for it in DND_ITEMS]
    else:
        arr = [int(x) for x in counts]
    return float(sum(a * float(v) for a, v in zip(arr, values)))


def enumerate_allocations(counts: Sequence[int]) -> Iterable[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    c = tuple(int(x) for x in counts)
    for b, h, bl in product(*(range(x + 1) for x in c)):
        self_alloc = (b, h, bl)
        opp_alloc = tuple(c[i] - self_alloc[i] for i in range(3))
        yield self_alloc, opp_alloc  # type: ignore[misc]


def build_value_map_543(orderings: Sequence[Tuple[str, str, str]]) -> Dict[Tuple[str, str, str], Tuple[float, float, float]]:
    return {tuple(o): ordering_to_values_543(o) for o in orderings}


def empirical_value_map(
    records: Sequence[DNDRecord],
    orderings: Sequence[Tuple[str, str, str]],
) -> Dict[Tuple[str, str, str], Tuple[float, float, float]]:
    by_order: Dict[Tuple[str, str, str], List[Tuple[int, int, int]]] = {
        tuple(o): [] for o in orderings
    }
    for rec in records:
        if rec.self_ordering is not None:
            by_order[tuple(rec.self_ordering)].append(tuple(rec.self_values))
        if rec.partner_ordering is not None:
            by_order[tuple(rec.partner_ordering)].append(tuple(rec.partner_values))
    fallback = build_value_map_543(orderings)
    out: Dict[Tuple[str, str, str], Tuple[float, float, float]] = {}
    for ordering in orderings:
        vals = by_order.get(tuple(ordering)) or []
        if vals:
            arr = np.asarray(vals, dtype=float)
            out[tuple(ordering)] = tuple(float(x) for x in arr.mean(axis=0))  # type: ignore[assignment]
        else:
            out[tuple(ordering)] = fallback[tuple(ordering)]
    return out


def build_dnd_menu(
    *,
    posterior: Sequence[float],
    orderings: Sequence[Tuple[str, str, str]],
    counts: Sequence[int],
    self_values: Sequence[float],
    opp_value_map: Mapping[Tuple[str, str, str], Sequence[float]],
    lambda_: float = 1.0,
    top_k: int = 5,
) -> List[DNDScoredAllocation]:
    p = np.asarray(posterior, dtype=float).flatten()
    if p.shape != (len(orderings),):
        raise ValueError(f"posterior shape {p.shape} does not match {len(orderings)} orderings")
    p = np.maximum(np.where(np.isfinite(p), p, 0.0), 0.0)
    total = float(p.sum())
    p = p / total if total > 0 else np.full(len(orderings), 1.0 / len(orderings))

    scored: List[DNDScoredAllocation] = []
    for self_alloc, opp_alloc in enumerate_allocations(counts):
        u_self = utility(self_alloc, self_values)
        opp_utils = np.asarray(
            [utility(opp_alloc, opp_value_map[tuple(ordering)]) for ordering in orderings],
            dtype=float,
        )
        exp_opp = float(p @ opp_utils)
        self_counts = {it: int(self_alloc[i]) for i, it in enumerate(DND_ITEMS)}
        opp_counts = {it: int(opp_alloc[i]) for i, it in enumerate(DND_ITEMS)}
        scored.append(DNDScoredAllocation(
            self_counts=self_counts,
            opp_counts=opp_counts,
            u_self=u_self,
            exp_u_opp=exp_opp,
            score=float(u_self) + float(lambda_) * exp_opp,
        ))
    scored.sort(key=lambda x: (-x.score, -x.u_self, tuple(x.self_counts[it] for it in DND_ITEMS)))
    return scored[: int(top_k)]


def allocation_vector(self_counts: Sequence[int], total_counts: Sequence[int]) -> Tuple[int, int, int, int, int, int]:
    self_t = tuple(int(x) for x in self_counts)
    opp_t = tuple(int(total_counts[i]) - self_t[i] for i in range(3))
    return self_t + opp_t  # type: ignore[return-value]
