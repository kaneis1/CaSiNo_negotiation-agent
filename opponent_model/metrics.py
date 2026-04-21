"""Evaluation metrics for opponent-priority prediction in CaSiNo.

Implements three per-prediction metrics against a ground-truth ordering of
the 3 items (Food / Water / Firewood):

    * ``ema``       — Exact Match Accuracy. 1 iff full ordering matches.
    * ``top1``      — 1 iff predicted top item matches true top item.
    * ``ndcg_at_3`` — Normalized DCG@3 with relevance grades (3, 2, 1) for
                      the true (top, mid, low) items, evaluated over the
                      predicted ranking.

Plus the dialogue-level harness ``evaluate_opponent_model`` that walks each
dialogue from both perspectives, snapshots predictions after the k-th
opponent utterance for k=1..5, and aggregates per-k means + a k-penalty
weighted average that rewards correctness from fewer observations.

All functions accept orderings as 3-element lists/tuples in
``[top, mid, low]`` order. Use :func:`get_ordering` to convert from the
CaSiNo ``value2issue`` dict shape.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


# ── Ordering helpers ───────────────────────────────────────────────────────


def get_ordering(priorities: Mapping[str, str]) -> List[str]:
    """Convert ``{"High": "Food", "Medium": "Water", "Low": "Firewood"}`` →
    ``["Food", "Water", "Firewood"]``.
    """
    try:
        return [priorities["High"], priorities["Medium"], priorities["Low"]]
    except KeyError as e:
        raise KeyError(
            f"priorities dict missing key {e}. Expected keys: High, Medium, Low. "
            f"Got: {dict(priorities)}"
        )


# ── Per-prediction metrics ────────────────────────────────────────────────


def ema(predicted: Sequence[str], true_ordering: Sequence[str]) -> float:
    """Exact Match Accuracy: 1.0 if the full ordering matches, else 0.0."""
    return 1.0 if list(predicted) == list(true_ordering) else 0.0


def top1(predicted: Sequence[str], true_ordering: Sequence[str]) -> float:
    """1.0 if the predicted top-priority item matches the true top, else 0.0."""
    return 1.0 if predicted[0] == true_ordering[0] else 0.0


def ndcg_at_3(predicted: Sequence[str], true_ordering: Sequence[str]) -> float:
    rel: Dict[str, int] = {
        true_ordering[0]: 5,  # Chawla spec, matches CaSiNo's 5/4/3 scoring
        true_ordering[1]: 4,
        true_ordering[2]: 3,
    }
    
    def _dcg(order: Sequence[str]) -> float:
        return sum(rel[item] / math.log2(i + 2) for i, item in enumerate(order))

    dcg_pred = _dcg(predicted)
    dcg_best = _dcg(true_ordering)
    dcg_worst = _dcg(list(reversed(true_ordering)))

    return (dcg_pred - dcg_worst) / (dcg_best - dcg_worst)
# ── Aggregation harness ────────────────────────────────────────────────────


# Type alias: a function that takes (partial_chat_logs, perspective_priorities,
#             opp_role_id, my_role_id, my_reasons) and returns an ordering
#             (top, mid, low). The eval harness gives the model_fn everything
#             it could plausibly need to behave like the agent in a real game.
OpponentModelFn = Callable[
    [List[Dict[str, Any]], Dict[str, str], str, str, Dict[str, str]],
    Sequence[str],
]


def _is_deal_action(turn: Mapping[str, Any]) -> bool:
    """CaSiNo non-utterance turns we should not feed to the opponent model."""
    return turn.get("text", "") in {
        "Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away",
    }


def evaluate_opponent_model(
    dialogues: Iterable[Mapping[str, Any]],
    opponent_model_fn: OpponentModelFn,
    *,
    max_k: int = 5,
    on_prediction: Callable[[Dict[str, Any]], None] | None = None,
) -> Dict[str, Any]:
    """Compute EMA / top1 / NDCG@3 at k = 1..max_k opponent utterances.

    For each dialogue × perspective, walk the chat once. After the k-th
    opponent utterance, snapshot the model's predicted ordering and score
    it against the opponent's ground-truth ``value2issue`` ordering.

    Args:
        dialogues: iterable of CaSiNo dialogues with ``chat_logs`` and
            ``participant_info[<role>]["value2issue"]``.
        opponent_model_fn: see :data:`OpponentModelFn`. Called once per
            (dialogue, perspective, k) snapshot. The harness passes the
            full partial conversation up to and including the k-th
            opponent utterance, plus the perspective agent's own
            priorities/reasons (which the agent may need for prompting).
        max_k: maximum number of opponent utterances to score. Default 5
            (matches the CaSiNo opponent-modeling literature).
        on_prediction: optional callback invoked with one record per
            scored snapshot. Useful for streaming logs to disk during a
            long run; the same record is also accumulated in the return.

    Returns:
        Dict with:
            ``per_k_means``    — ``{metric: {k: mean}}``
            ``per_k_counts``   — ``{k: n_snapshots}``
            ``kpenalty``       — ``{metric: weighted_mean}``  (weights 5..1 / 15)
            ``predictions``    — list of per-snapshot records
            ``summary``        — flat ``{f"{m}_k{k}": float, f"{m}_kpenalty": float}``
    """
    bucket: Dict[int, Dict[str, List[float]]] = {
        k: {"ema": [], "top1": [], "ndcg": []} for k in range(1, max_k + 1)
    }
    predictions: List[Dict[str, Any]] = []

    roles = ("mturk_agent_1", "mturk_agent_2")

    for dialogue in dialogues:
        chat_logs = dialogue["chat_logs"]
        pinfo = dialogue["participant_info"]
        dialogue_id = dialogue.get("dialogue_id")

        for perspective in roles:
            opp_role = roles[1] if perspective == roles[0] else roles[0]

            true_ordering = get_ordering(pinfo[opp_role]["value2issue"])
            my_priorities = pinfo[perspective]["value2issue"]
            my_reasons = pinfo[perspective].get("value2reason", {})

            opp_count = 0
            partial: List[Dict[str, Any]] = []

            for turn in chat_logs:
                if _is_deal_action(turn):
                    continue
                partial.append(turn)
                if turn["id"] != opp_role:
                    continue

                opp_count += 1
                if opp_count > max_k:
                    break
                if opp_count not in bucket:
                    continue

                predicted = list(opponent_model_fn(
                    partial, dict(my_priorities), opp_role, perspective, dict(my_reasons),
                ))

                e = ema(predicted, true_ordering)
                t = top1(predicted, true_ordering)
                n = ndcg_at_3(predicted, true_ordering)

                bucket[opp_count]["ema"].append(e)
                bucket[opp_count]["top1"].append(t)
                bucket[opp_count]["ndcg"].append(n)

                record = {
                    "dialogue_id": dialogue_id,
                    "perspective": perspective,
                    "opp_role": opp_role,
                    "k": opp_count,
                    "predicted": predicted,
                    "true": true_ordering,
                    "ema": e,
                    "top1": t,
                    "ndcg": n,
                }
                predictions.append(record)
                if on_prediction is not None:
                    on_prediction(record)

    return summarize(bucket, predictions=predictions, max_k=max_k)


def summarize(
    bucket: Mapping[int, Mapping[str, Sequence[float]]],
    *,
    predictions: List[Dict[str, Any]] | None = None,
    max_k: int = 5,
) -> Dict[str, Any]:
    """Compute per-k means and the k-penalty weighted aggregate."""
    metrics = ("ema", "top1", "ndcg")
    per_k_means: Dict[str, Dict[int, float]] = {m: {} for m in metrics}
    per_k_counts: Dict[int, int] = {}
    summary_flat: Dict[str, float] = {}

    for k in range(1, max_k + 1):
        per_k_counts[k] = len(bucket[k]["ema"])
        for m in metrics:
            vals = bucket[k][m]
            mean = float(np.mean(vals)) if vals else float("nan")
            per_k_means[m][k] = mean
            summary_flat[f"{m}_k{k}"] = mean

    weights = np.array(
        [(max_k + 1 - k) for k in range(1, max_k + 1)], dtype=float,
    )
    weights = weights / weights.sum()  # e.g. [5,4,3,2,1] / 15
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
        "per_k_means": per_k_means,
        "per_k_counts": per_k_counts,
        "kpenalty": kpenalty,
        "predictions": predictions if predictions is not None else [],
        "summary": summary_flat,
    }


def format_summary(result: Mapping[str, Any]) -> str:
    """Pretty single-string render of the per-k + k-penalty table."""
    summary = result["summary"]
    counts = result["per_k_counts"]
    metrics = ("ema", "top1", "ndcg")
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
    lines.append("snapshot counts: " + ", ".join(f"k={k}:{counts[k]}" for k in ks))
    return "\n".join(lines)


__all__ = [
    "ema",
    "top1",
    "ndcg_at_3",
    "get_ordering",
    "evaluate_opponent_model",
    "summarize",
    "format_summary",
    "OpponentModelFn",
]
