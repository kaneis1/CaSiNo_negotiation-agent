"""Satisfaction-prediction metrics for the SFT'd 8B model.

CaSiNo's satisfaction is a 5-class ordinal label
(Extremely dissatisfied ... Extremely satisfied). We report:

    * ``accuracy``  — exact 5-class match rate.
    * ``mae``       — mean absolute error on the ordinal scale (0..4).
    * ``kpenalty``  — same 5/4/3/2/1 weighted average over k=1..5
                      that ``opponent_model.metrics.summarize`` uses for
                      prefs metrics, so the two summaries are directly
                      comparable.

The "predicted" satisfaction can be ``None`` (e.g. the model emitted a
malformed label or no satisfaction key at all). Such rows count as
incorrect for accuracy and contribute the maximum possible MAE (4) so
they actively hurt both metrics rather than being silently dropped.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from sft_8b.prompts import SATISFACTION_LABELS

# Map label -> ordinal index (0 = lowest satisfaction, 4 = highest).
LABEL_TO_INDEX: Dict[str, int] = {
    label: idx for idx, label in enumerate(SATISFACTION_LABELS)
}
N_CLASSES = len(SATISFACTION_LABELS)
MAX_MAE = float(N_CLASSES - 1)  # 4.0


# ── Per-prediction scoring ─────────────────────────────────────────────────


def satisfaction_accuracy(pred: Optional[str], true: str) -> float:
    return 1.0 if pred == true else 0.0


def satisfaction_abs_error(pred: Optional[str], true: str) -> float:
    if pred not in LABEL_TO_INDEX or true not in LABEL_TO_INDEX:
        return MAX_MAE
    return float(abs(LABEL_TO_INDEX[pred] - LABEL_TO_INDEX[true]))


# ── Aggregation over the predictions.jsonl bucket ──────────────────────────


def summarize_satisfaction(
    predictions: Iterable[Mapping[str, Any]],
    *,
    max_k: int = 5,
) -> Dict[str, Any]:
    """Compute per-k satisfaction metrics + k-penalty.

    Each input record must carry ``pred_satisfaction`` and
    ``true_satisfaction`` (records emitted by ``sft_8b.eval_run``). Rows
    missing either are skipped from the count entirely.

    Returned dict:

        {
          "per_k_means":  {"sat_acc": {1: .., 2: ..}, "sat_mae": {...}},
          "per_k_counts": {1: int, ...},
          "kpenalty":     {"sat_acc": float, "sat_mae": float},
          "summary":      {"sat_acc_k1": .., "sat_mae_kpenalty": .., ...},
        }
    """
    per_k: Dict[int, Dict[str, List[float]]] = {
        k: {"sat_acc": [], "sat_mae": []} for k in range(1, max_k + 1)
    }

    for r in predictions:
        if "true_satisfaction" not in r:
            continue
        k = r.get("k")
        if k not in per_k:
            continue
        true = r["true_satisfaction"]
        pred = r.get("pred_satisfaction")
        per_k[k]["sat_acc"].append(satisfaction_accuracy(pred, true))
        per_k[k]["sat_mae"].append(satisfaction_abs_error(pred, true))

    metrics = ("sat_acc", "sat_mae")
    per_k_means: Dict[str, Dict[int, float]] = {m: {} for m in metrics}
    per_k_counts: Dict[int, int] = {}
    summary_flat: Dict[str, float] = {}

    for k in range(1, max_k + 1):
        per_k_counts[k] = len(per_k[k]["sat_acc"])
        for m in metrics:
            vals = per_k[k][m]
            mean = float(np.mean(vals)) if vals else float("nan")
            per_k_means[m][k] = mean
            summary_flat[f"{m}_k{k}"] = mean

    weights = np.array(
        [(max_k + 1 - k) for k in range(1, max_k + 1)], dtype=float,
    )
    weights = weights / weights.sum()
    kpenalty: Dict[str, float] = {}
    for m in metrics:
        scores = np.array(
            [per_k_means[m][k] for k in range(1, max_k + 1)], dtype=float,
        )
        if np.any(np.isnan(scores)):
            kpenalty[m] = float("nan")
        else:
            kpenalty[m] = float(np.dot(weights, scores))
        summary_flat[f"{m}_kpenalty"] = kpenalty[m]

    return {
        "per_k_means":  per_k_means,
        "per_k_counts": per_k_counts,
        "kpenalty":     kpenalty,
        "summary":      summary_flat,
    }


def format_satisfaction_summary(result: Mapping[str, Any]) -> str:
    summary = result["summary"]
    counts = result["per_k_counts"]
    metrics = ("sat_acc", "sat_mae")
    ks = sorted(counts.keys())

    lines: List[str] = []
    header = f"{'metric':<8}" + "".join(f"  k={k:<6}" for k in ks) + "  kpenalty"
    lines.append(header)
    lines.append("-" * len(header))
    for m in metrics:
        row = f"{m:<8}"
        for k in ks:
            row += f"  {summary[f'{m}_k{k}']:>7.3f}"
        row += f"  {summary[f'{m}_kpenalty']:>7.3f}"
        lines.append(row)
    lines.append("")
    lines.append(
        "satisfaction snapshot counts: "
        + ", ".join(f"k={k}:{counts[k]}" for k in ks)
    )
    return "\n".join(lines)


__all__ = [
    "SATISFACTION_LABELS",
    "LABEL_TO_INDEX",
    "satisfaction_accuracy",
    "satisfaction_abs_error",
    "summarize_satisfaction",
    "format_satisfaction_summary",
]
