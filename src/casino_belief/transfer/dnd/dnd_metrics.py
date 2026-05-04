"""Metrics for DND transfer posterior and allocation diagnostics."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def brier_reference(n_classes: int = 6) -> float:
    # Sum-squared multiclass Brier divided by K-1. For a uniform K-way
    # posterior, this equals 1/K, matching the paper's six-way 0.167 reference.
    return 1.0 / float(n_classes)


def normalized_brier_sum(posterior: Sequence[float], true_index: int) -> float:
    arr = np.asarray(posterior, dtype=float).flatten()
    n = arr.shape[0]
    if n <= 1:
        return float("nan")
    total = float(arr.sum())
    if total > 0:
        arr = arr / total
    one_hot = np.zeros(n, dtype=float)
    one_hot[int(true_index)] = 1.0
    return float(np.sum((arr - one_hot) ** 2) / (n - 1))


def ema(predicted: Sequence[str], true_ordering: Sequence[str]) -> float:
    return 1.0 if tuple(predicted) == tuple(true_ordering) else 0.0


def top1(predicted: Sequence[str], true_ordering: Sequence[str]) -> float:
    return 1.0 if predicted and true_ordering and predicted[0] == true_ordering[0] else 0.0


def ndcg_at_3(predicted: Sequence[str], true_ordering: Sequence[str]) -> float:
    rel = {true_ordering[0]: 5.0, true_ordering[1]: 4.0, true_ordering[2]: 3.0}

    def dcg(order: Sequence[str]) -> float:
        return sum(rel[item] / math.log2(i + 2) for i, item in enumerate(order))

    best = dcg(true_ordering)
    worst = dcg(list(reversed(true_ordering)))
    if best == worst:
        return float("nan")
    return float((dcg(predicted) - worst) / (best - worst))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 0:
        return float("nan")
    return float(np.dot(aa, bb) / denom)


def mean_or_nan(vals: Iterable[Optional[float]]) -> float:
    clean = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    return float(np.mean(clean)) if clean else float("nan")


def kpenalty(
    per_k_means: Mapping[int, Mapping[str, float]],
    *,
    metric: str,
    k_min: int,
    k_max: int,
) -> float:
    ks = list(range(int(k_min), int(k_max) + 1))
    weights = np.array([(k_max + 1 - k) for k in ks], dtype=float)
    weights = weights / float(weights.sum())
    vals = []
    for k in ks:
        value = (per_k_means.get(k) or {}).get(metric)
        if value is None or not np.isfinite(float(value)):
            return float("nan")
        vals.append(float(value))
    return float(np.dot(weights, np.asarray(vals, dtype=float)))


def summarize_snapshot_metrics(records: Sequence[Mapping[str, Any]], *, max_k: int = 5) -> Dict[str, Any]:
    metrics = ("brier", "ema", "top1", "ndcg", "bid_cosine", "utility_norm")
    per_k: Dict[int, Dict[str, float]] = {}
    support: Dict[int, int] = {}
    for k in range(1, max_k + 1):
        rows = [r for r in records if int(r.get("k", -1)) == k]
        support[k] = len(rows)
        per_k[k] = {m: mean_or_nan(r.get(m) for r in rows) for m in metrics}
    overall = {m: mean_or_nan(r.get(m) for r in records) for m in metrics}
    return {
        "support_by_k": {str(k): v for k, v in support.items()},
        "per_k": {str(k): v for k, v in per_k.items()},
        "overall": overall,
        "ema_at2": per_k.get(2, {}).get("ema", float("nan")),
        "kpenalty_1_3": {
            m: kpenalty(per_k, metric=m, k_min=1, k_max=3)
            for m in ("ema", "top1", "ndcg")
        },
        "kpenalty_1_5": {
            m: kpenalty(per_k, metric=m, k_min=1, k_max=5)
            for m in ("ema", "top1", "ndcg")
        },
    }
