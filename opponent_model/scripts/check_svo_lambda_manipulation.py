#!/usr/bin/env python3
"""Directional A/B manipulation check for SVO-conditioned Bayesian lambda.

This is intentionally not called a monotonicity check: CaSiNo's SVO metadata
currently yields two active lambda values in the held-out set. We compare
lambda=1 against lambda=2 on predicted bid self-points and joint-points.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

from opponent_model.hypotheses import ITEMS
from sft_8b.menu import points


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _dialogue_lookup(data_path: Path) -> Dict[str, Mapping[str, Any]]:
    with data_path.open() as f:
        dialogues = json.load(f)
    return {str(d.get("dialogue_id", i)): d for i, d in enumerate(dialogues)}


def _counts_from_bid(bid: Any, offset: int) -> Optional[Dict[str, int]]:
    if not isinstance(bid, (list, tuple)) or len(bid) != 6:
        return None
    try:
        return {item: int(float(bid[offset + i])) for i, item in enumerate(ITEMS)}
    except (TypeError, ValueError):
        return None


def _mean(vals: Iterable[float]) -> float:
    arr = list(vals)
    return float(sum(arr) / len(arr)) if arr else float("nan")


def _std(vals: Iterable[float]) -> float:
    arr = list(vals)
    if len(arr) < 2:
        return float("nan")
    return float(np.std(np.asarray(arr, dtype=float), ddof=1))


def _cohens_d(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s = math.sqrt((float(np.var(a, ddof=1)) + float(np.var(b, ddof=1))) / 2.0)
    if s == 0.0:
        return 0.0
    return float((_mean(b) - _mean(a)) / s)


def _cliffs_delta(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return float("nan")
    greater = lesser = 0
    for av in a:
        for bv in b:
            if bv > av:
                greater += 1
            elif bv < av:
                lesser += 1
    return float((greater - lesser) / (len(a) * len(b)))


def _tests(a: list[float], b: list[float]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "cohens_d_lambda2_minus_lambda1": _cohens_d(a, b),
        "cliffs_delta_lambda2_minus_lambda1": _cliffs_delta(a, b),
        "scipy_available": False,
    }
    try:
        from scipy import stats  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on env
        out["scipy_error"] = repr(exc)
        return out

    out["scipy_available"] = True
    if len(a) >= 2 and len(b) >= 2:
        t_res = stats.ttest_ind(b, a, equal_var=False, nan_policy="omit")
        out["welch_t"] = float(t_res.statistic)
        out["welch_p"] = float(t_res.pvalue)
    if a and b:
        u_res = stats.mannwhitneyu(b, a, alternative="two-sided")
        out["mannwhitney_u"] = float(u_res.statistic)
        out["mannwhitney_p"] = float(u_res.pvalue)
    return out


def _metric_summary(groups: Mapping[str, list[float]], metric: str) -> Dict[str, Any]:
    a = list(groups.get("1", []))
    b = list(groups.get("2", []))
    out: Dict[str, Any] = {
        "metric": metric,
        "lambda_1": {"n": len(a), "mean": _mean(a), "std": _std(a)},
        "lambda_2": {"n": len(b), "mean": _mean(b), "std": _std(b)},
        "mean_diff_lambda2_minus_lambda1": _mean(b) - _mean(a),
    }
    out.update(_tests(a, b))
    return out


def _significant(block: Mapping[str, Any]) -> bool:
    if not block.get("scipy_available"):
        return False
    pvals = [
        block.get("welch_p"),
        block.get("mannwhitney_p"),
    ]
    usable = [float(p) for p in pvals if isinstance(p, (int, float)) and math.isfinite(float(p))]
    return bool(usable) and all(p < 0.05 for p in usable)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/casino_test.json"))
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("opponent_model/results/day9_svo_lambda_manipulation"),
    )
    args = ap.parse_args()

    dialogues = _dialogue_lookup(args.data)
    rows = _load_jsonl(args.records)
    self_groups: Dict[str, list[float]] = defaultdict(list)
    joint_groups: Dict[str, list[float]] = defaultdict(list)
    scored_rows: list[Dict[str, Any]] = []

    for row in rows:
        pred = row.get("pred") or {}
        bid = pred.get("bid")
        lambda_value = pred.get("lambda")
        if bid is None or lambda_value is None:
            continue
        lambda_key = f"{float(lambda_value):g}"
        if lambda_key not in {"1", "2"}:
            continue
        self_counts = _counts_from_bid(bid, 0)
        opp_counts = _counts_from_bid(bid, 3)
        if self_counts is None or opp_counts is None:
            continue
        dialogue = dialogues.get(str(row.get("dialogue_id")))
        if dialogue is None:
            continue
        pinfo = dialogue.get("participant_info") or {}
        perspective = row.get("perspective")
        opp_role = row.get("opp_role")
        try:
            my_priorities = pinfo[perspective]["value2issue"]
            opp_priorities = pinfo[opp_role]["value2issue"]
        except Exception:
            continue
        self_pts = float(points(self_counts, my_priorities))
        opp_pts = float(points(opp_counts, opp_priorities))
        joint_pts = self_pts + opp_pts
        self_groups[lambda_key].append(self_pts)
        joint_groups[lambda_key].append(joint_pts)
        scored_rows.append({
            "dialogue_id": row.get("dialogue_id"),
            "turn_index": row.get("turn_index"),
            "perspective": perspective,
            "lambda": float(lambda_value),
            "self_points": self_pts,
            "opp_points": opp_pts,
            "joint_points": joint_pts,
            "bid": bid,
        })

    self_summary = _metric_summary(self_groups, "self_points")
    joint_summary = _metric_summary(joint_groups, "joint_points")
    direction_ok = (
        self_summary["mean_diff_lambda2_minus_lambda1"] < 0
        and joint_summary["mean_diff_lambda2_minus_lambda1"] > 0
    )
    significance_ok = _significant(self_summary) and _significant(joint_summary)
    if direction_ok and significance_ok:
        gate_status = "pass"
    elif direction_ok:
        gate_status = "inconclusive_direction_only"
    else:
        gate_status = "fail_wrong_direction"

    report = {
        "metadata": {
            "records": str(args.records),
            "data": str(args.data),
            "check": (
                "directional A/B: lambda=2 should lower self-points and "
                "raise joint-points versus lambda=1"
            ),
            "not_monotonicity": True,
        },
        "support": {
            "lambda_1_bid_turns": len(self_groups.get("1", [])),
            "lambda_2_bid_turns": len(self_groups.get("2", [])),
            "scored_bid_turns": len(scored_rows),
        },
        "self_points": self_summary,
        "joint_points": joint_summary,
        "direction_ok": direction_ok,
        "significance_ok": significance_ok,
        "gate_status": gate_status,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    debug_path = args.output_dir / "scored_bid_turns.jsonl"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with debug_path.open("w", encoding="utf-8") as f:
        for row in scored_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {summary_path}")
    print(f"Wrote {debug_path}")
    print("gate_status:", gate_status)
    if gate_status == "fail_wrong_direction":
        return 2
    if gate_status == "inconclusive_direction_only":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
