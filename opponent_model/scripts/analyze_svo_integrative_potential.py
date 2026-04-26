#!/usr/bin/env python3
"""Break down SVO-lambda bid outcomes by dialogue integrative potential.

We derive integrative potential from the two participants' priority orderings:
the maximum joint utility attainable over all legal CaSiNo splits. This is a
transparent proxy for the dialogue-level "room for integrative gains" used in
the dataset analysis.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, Mapping, Optional

from opponent_model.hypotheses import ITEMS
from sft_8b.menu import points


PRIORITY_POINTS = {"High": 5, "Medium": 4, "Low": 3}


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_dialogues(path: Path) -> Dict[str, Mapping[str, Any]]:
    with path.open() as f:
        data = json.load(f)
    return {str(d.get("dialogue_id", i)): d for i, d in enumerate(data)}


def _counts_from_bid(bid: Any, offset: int) -> Optional[Dict[str, int]]:
    if not isinstance(bid, (list, tuple)) or len(bid) != 6:
        return None
    try:
        return {item: int(float(bid[offset + i])) for i, item in enumerate(ITEMS)}
    except (TypeError, ValueError):
        return None


def _max_joint_points(
    my_priorities: Mapping[str, str],
    opp_priorities: Mapping[str, str],
) -> int:
    best = -1
    for split in product(range(4), repeat=3):
        self_counts = {item: int(split[i]) for i, item in enumerate(ITEMS)}
        opp_counts = {item: 3 - self_counts[item] for item in ITEMS}
        joint = points(self_counts, my_priorities) + points(opp_counts, opp_priorities)
        best = max(best, int(joint))
    return best


def _summ(vals: Iterable[float]) -> Dict[str, Any]:
    arr = list(vals)
    return {
        "n": len(arr),
        "mean": float(mean(arr)) if arr else float("nan"),
        "std": float(stdev(arr)) if len(arr) > 1 else float("nan"),
        "min": float(min(arr)) if arr else float("nan"),
        "max": float(max(arr)) if arr else float("nan"),
    }


def _welch(lambda1: list[float], lambda2: list[float]) -> Dict[str, Any]:
    out = {
        "mean_diff_lambda2_minus_lambda1": (
            float(mean(lambda2) - mean(lambda1))
            if lambda1 and lambda2 else float("nan")
        ),
        "scipy_available": False,
    }
    try:
        from scipy import stats  # type: ignore
    except Exception as exc:  # pragma: no cover - env-dependent
        out["scipy_error"] = repr(exc)
        return out
    out["scipy_available"] = True
    if len(lambda1) >= 2 and len(lambda2) >= 2:
        res = stats.ttest_ind(lambda2, lambda1, equal_var=False)
        out["welch_t"] = float(res.statistic)
        out["welch_p"] = float(res.pvalue)
    return out


def _bin(max_joint: int) -> str:
    if max_joint <= 36:
        return "low_ip"
    if max_joint <= 39:
        return "mid_ip"
    return "high_ip"


def _dialogue_ip_table(dialogues: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for did, dialogue in dialogues.items():
        pinfo = dialogue.get("participant_info") or {}
        try:
            mt1_priorities = pinfo["mturk_agent_1"]["value2issue"]
            mt2_priorities = pinfo["mturk_agent_2"]["value2issue"]
        except KeyError:
            continue
        max_joint = _max_joint_points(mt1_priorities, mt2_priorities)
        human_joint = float(
            pinfo["mturk_agent_1"]["outcomes"]["points_scored"]
            + pinfo["mturk_agent_2"]["outcomes"]["points_scored"]
        )
        out[did] = {
            "max_joint_points": max_joint,
            "integrative_gain_over_identical_preferences": max_joint - 36,
            "ip_bin": _bin(max_joint),
            "human_joint_points": human_joint,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--records",
        type=Path,
        default=Path("opponent_model/results/turn_eval_bayesian_svo_lambda_m5_f0.50_full150/turn_records.jsonl"),
    )
    ap.add_argument("--data", type=Path, default=Path("data/casino_test.json"))
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("opponent_model/results/day10_svo_integrative_potential"),
    )
    args = ap.parse_args()

    dialogues = _load_dialogues(args.data)
    ip = _dialogue_ip_table(dialogues)
    rows = _load_jsonl(args.records)
    groups: Dict[str, Dict[str, Dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"self_points": [], "opp_points": [], "joint_points": []})
    )
    debug_rows: list[Dict[str, Any]] = []

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
        did = str(row.get("dialogue_id"))
        dialogue = dialogues.get(did)
        ip_row = ip.get(did)
        if dialogue is None or ip_row is None:
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
        bin_name = str(ip_row["ip_bin"])
        for metric, value in (
            ("self_points", self_pts),
            ("opp_points", opp_pts),
            ("joint_points", joint_pts),
        ):
            groups[bin_name][lambda_key][metric].append(value)
            groups["all"][lambda_key][metric].append(value)
        debug_rows.append({
            "dialogue_id": row.get("dialogue_id"),
            "turn_index": row.get("turn_index"),
            "perspective": perspective,
            "lambda": float(lambda_value),
            "ip_bin": bin_name,
            "max_joint_points": ip_row["max_joint_points"],
            "self_points": self_pts,
            "opp_points": opp_pts,
            "joint_points": joint_pts,
            "bid": bid,
        })

    summary: Dict[str, Any] = {}
    for bin_name in ("low_ip", "mid_ip", "high_ip", "all"):
        summary[bin_name] = {}
        for metric in ("self_points", "opp_points", "joint_points"):
            l1 = groups[bin_name]["1"][metric]
            l2 = groups[bin_name]["2"][metric]
            summary[bin_name][metric] = {
                "lambda_1": _summ(l1),
                "lambda_2": _summ(l2),
                "test": _welch(l1, l2),
            }
    dialogue_bin_counts: Dict[str, int] = defaultdict(int)
    for row in ip.values():
        dialogue_bin_counts[str(row["ip_bin"])] += 1

    report = {
        "metadata": {
            "records": str(args.records),
            "data": str(args.data),
            "ip_definition": (
                "max feasible joint points over all 64 CaSiNo splits, given "
                "the dialogue participants' priority orderings"
            ),
            "ip_bins": {
                "low_ip": "max_joint_points <= 36",
                "mid_ip": "37 <= max_joint_points <= 39",
                "high_ip": "max_joint_points >= 40",
            },
        },
        "dialogue_ip_bin_counts": dict(dialogue_bin_counts),
        "summary": summary,
        "interpretation": {
            "joint_lambda2_minus_lambda1_by_bin": {
                bin_name: summary[bin_name]["joint_points"]["test"]["mean_diff_lambda2_minus_lambda1"]
                for bin_name in ("low_ip", "mid_ip", "high_ip", "all")
            }
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    debug_path = args.output_dir / "scored_bid_turns.jsonl"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with debug_path.open("w", encoding="utf-8") as f:
        for row in debug_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {summary_path}")
    print(f"Wrote {debug_path}")
    print(json.dumps(report["interpretation"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
