"""Posterior-to-action intervention metrics for saved turn-level records."""

from __future__ import annotations

import argparse
import json
import math
import random
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from casino_belief.evaluation.hypotheses import ITEMS
from casino_belief.evaluation.turn_level_metrics import coerce_bid_vector
from casino_belief.diagnostics.ablation.ablation import normalize_posterior
from casino_belief.policy.menu import PRIORITY_POINTS, build_menu, points
from casino_belief.belief.posterior import N_ORDERINGS, ORDERINGS

MODES = {"adversarial", "uniform", "random"}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _dialogue_lookup(path: Path) -> Dict[str, Mapping[str, Any]]:
    data = json.load(path.open())
    return {str(d.get("dialogue_id")): d for d in data}


def _true_ordering(dialogue: Mapping[str, Any], role: str) -> Tuple[str, str, str]:
    pri = dialogue["participant_info"][role]["value2issue"]
    return (pri["High"], pri["Medium"], pri["Low"])


def _intervention_posterior(
    *,
    mode: str,
    record: Mapping[str, Any],
    dialogue: Mapping[str, Any],
    rng: random.Random,
) -> np.ndarray:
    if mode == "uniform":
        return np.full(N_ORDERINGS, 1.0 / N_ORDERINGS, dtype=np.float64)
    if mode == "random":
        return normalize_posterior([rng.expovariate(1.0) for _ in range(N_ORDERINGS)])
    if mode == "adversarial":
        opp_role = str(record["opp_role"])
        top, mid, low = _true_ordering(dialogue, opp_role)
        wrong = (low, mid, top)
        arr = np.zeros(N_ORDERINGS, dtype=np.float64)
        arr[ORDERINGS.index(wrong)] = 1.0
        return arr
    raise ValueError(f"unknown mode {mode!r}")


def _best_menu_bid(
    posterior: Sequence[float],
    self_priorities: Mapping[str, str],
    *,
    lambda_: float,
) -> np.ndarray:
    top = build_menu(posterior, self_priorities, lambda_=lambda_, top_k=1)[0]
    self_arr = np.array([int(top.self_counts[it]) for it in ITEMS], dtype=float)
    opp_arr = np.array([int(top.opp_counts[it]) for it in ITEMS], dtype=float)
    return np.concatenate([self_arr, opp_arr])


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def run_intervention(
    *,
    records: Sequence[Mapping[str, Any]],
    dialogues_by_id: Mapping[str, Mapping[str, Any]],
    mode: str,
    lambda_: float,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    consistency: List[float] = []
    changed: List[float] = []
    drift_from_original_menu: List[float] = []
    drift_from_student: List[float] = []
    support = 0

    for rec in records:
        pred = rec.get("pred") or {}
        emitted = pred.get("posterior")
        student_bid = coerce_bid_vector(pred.get("bid"), target_self=True)
        if emitted is None:
            continue
        dialogue = dialogues_by_id.get(str(rec.get("dialogue_id")))
        if dialogue is None:
            continue
        pinfo = dialogue.get("participant_info", {})
        try:
            my_pr = pinfo[rec["perspective"]]["value2issue"]
        except KeyError:
            continue

        try:
            original_menu_bid = _best_menu_bid(emitted, my_pr, lambda_=lambda_)
            intervened = _intervention_posterior(
                mode=mode,
                record=rec,
                dialogue=dialogue,
                rng=rng,
            )
            intervened_bid = _best_menu_bid(intervened, my_pr, lambda_=lambda_)
        except Exception:
            continue

        support += 1
        same_menu = bool(np.array_equal(original_menu_bid, intervened_bid))
        changed.append(0.0 if same_menu else 1.0)
        drift_from_original_menu.append(_l2(original_menu_bid, intervened_bid))
        if student_bid is not None:
            consistency.append(float(np.array_equal(student_bid, original_menu_bid)))
            drift_from_student.append(_l2(student_bid, intervened_bid))

    return {
        "mode": mode,
        "support": support,
        "belief_action_consistency_rate": (
            float(np.mean(consistency)) if consistency else float("nan")
        ),
        "belief_action_consistency_support": len(consistency),
        "intervention_sensitivity_rate": (
            float(np.mean(changed)) if changed else float("nan")
        ),
        "allocation_drift_l2_mean": (
            float(np.mean(drift_from_original_menu)) if drift_from_original_menu else float("nan")
        ),
        "student_to_intervention_drift_l2_mean": (
            float(np.mean(drift_from_student)) if drift_from_student else float("nan")
        ),
    }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", required=True, type=Path)
    p.add_argument("--data", default="data/casino/casino_test.json", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--mode", choices=sorted(MODES), required=True)
    p.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2026)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    records = _load_jsonl(args.records)
    dialogues_by_id = _dialogue_lookup(args.data)
    summary = run_intervention(
        records=records,
        dialogues_by_id=dialogues_by_id,
        mode=args.mode,
        lambda_=args.lambda_,
        seed=args.seed,
    )
    summary["config"] = {
        "records": str(args.records),
        "data": str(args.data),
        "mode": args.mode,
        "lambda": args.lambda_,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

