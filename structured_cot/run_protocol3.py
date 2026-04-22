"""Protocol 3: score agent traces against human ground truth.

Reads an existing Protocol 1 run directory (``turns.jsonl``), pulls
strategy annotations from ``CaSiNo/data/casino_ann.json``, and computes:

  * Accept F1                 (binary, positive class = "accept")
  * Bid cosine similarity     (6-dim, comparing my-share + opp-share)
  * Strategy macro-F1         (10 CaSiNo tags, multi-label per utterance)

Brier score requires a posterior over priority orderings; the vanilla
StructuredCoT agent doesn't expose one, so that metric is reported as
"not applicable" for this run.

Usage:
    python -m structured_cot.run_protocol3 \
        --run-dir structured_cot/results/protocol1_70b_239057722
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from opponent_model.turn_agents import KeywordStrategyClassifier
from opponent_model.turn_level_metrics import (
    CASINO_STRATEGIES,
    _complete_bid,
    binary_accept_f1,
    build_annotation_lookup,
    cosine_similarity,
    macro_f1_multilabel,
)

logger = logging.getLogger("structured_cot.run_protocol3")

ITEMS = ("Food", "Water", "Firewood")


# ── Decode helpers (logs are already in clean form, so these are thin) ─────


def _agent_accept(parsed_decision: Mapping[str, Any]) -> bool:
    return (parsed_decision or {}).get("action") == "accept"


def _human_accept(gt_decision: Mapping[str, Any]) -> bool:
    return (gt_decision or {}).get("action") == "accept"


def _bid_vec_from_counter_offer(
    co: Optional[Mapping[str, Any]],
) -> Optional[np.ndarray]:
    """6-dim [self_Food, self_Water, self_Firewood, opp_*, opp_*, opp_*]."""
    if not isinstance(co, Mapping):
        return None
    try:
        arr = np.array([int(co[it]) for it in ITEMS], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    return _complete_bid(arr, target_self=True)


# ── Run loader ─────────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _load_annotation_lookups(
    dialogue_ids: List[Any],
    *,
    ann_path: Path,
    data_path: Path,
) -> Dict[Any, Dict[int, List[str]]]:
    """Build ``{dialogue_id -> {chat_logs_index -> [tags]}}`` lookups.

    ``build_annotation_lookup`` needs both the annotations list and the
    chat_logs it aligns against; we read chat_logs from the test data
    file (same source Protocol 1 was run on).
    """
    with ann_path.open() as f:
        ann_all = json.load(f)
    with data_path.open() as f:
        data_all = json.load(f)

    ann_by_id = {d.get("dialogue_id"): d.get("annotations", []) for d in ann_all}
    logs_by_id = {d.get("dialogue_id"): d.get("chat_logs", []) for d in data_all}

    lookups: Dict[Any, Dict[int, List[str]]] = {}
    for did in dialogue_ids:
        anns = ann_by_id.get(did)
        logs = logs_by_id.get(did)
        if anns is None or logs is None:
            lookups[did] = {}
            continue
        lookups[did] = build_annotation_lookup(anns, logs)
    return lookups


# ── Core pass ──────────────────────────────────────────────────────────────


def evaluate(
    turns: List[Mapping[str, Any]],
    ann_lookups: Mapping[Any, Mapping[int, List[str]]],
    *,
    strategy_classifier=None,
) -> Dict[str, Any]:
    strategy_classifier = strategy_classifier or KeywordStrategyClassifier()

    accept_pairs: List[Tuple[bool, bool]] = []
    bid_cosines: List[float] = []
    bid_pair_details: List[Dict[str, Any]] = []
    strat_pred: List[List[str]] = []
    strat_true: List[List[str]] = []
    strat_detail: List[Dict[str, Any]] = []

    per_turn_rows: List[Dict[str, Any]] = []
    n_human_submits = 0
    n_human_accepts = 0
    n_agent_accepts = 0

    for t in turns:
        dialogue_id = t.get("dialogue_id")
        chat_idx    = t.get("turn_index")
        agent_dec   = t.get("parsed_decision") or {}
        gt_dec      = t.get("ground_truth_decision") or {}
        agent_utt   = (t.get("parsed_utterance") or "").strip()
        gt_utt      = (t.get("ground_truth_utterance") or "").strip()

        agent_a = _agent_accept(agent_dec)
        human_a = _human_accept(gt_dec)
        accept_pairs.append((agent_a, human_a))
        if agent_a:
            n_agent_accepts += 1
        if human_a:
            n_human_accepts += 1

        row: Dict[str, Any] = {
            "dialogue_id":            dialogue_id,
            "turn_index":             chat_idx,
            "agent_action":           agent_dec.get("action"),
            "human_action":           gt_dec.get("action"),
            "agent_accept":           agent_a,
            "human_accept":           human_a,
            "agent_counter_offer":    agent_dec.get("counter_offer"),
            "human_counter_offer":    gt_dec.get("counter_offer"),
            "bid_cosine":             None,
            "human_strategies":       None,
            "agent_strategies":       None,
        }

        # ── Bid cosine: both sides proposed a split at this turn position
        agent_bid = _bid_vec_from_counter_offer(agent_dec.get("counter_offer"))
        human_co  = gt_dec.get("counter_offer")
        if human_co is not None:
            n_human_submits += 1
        human_bid = _bid_vec_from_counter_offer(human_co)
        if agent_bid is not None and human_bid is not None:
            cs = cosine_similarity(agent_bid, human_bid)
            bid_cosines.append(cs)
            bid_pair_details.append({
                "dialogue_id": dialogue_id,
                "turn_index":  chat_idx,
                "agent_bid":   agent_bid.tolist(),
                "human_bid":   human_bid.tolist(),
                "cosine":      cs,
            })
            row["bid_cosine"] = cs

        # ── Strategy macro-F1: compare on turns with a human annotation
        lookup = ann_lookups.get(dialogue_id, {})
        human_tags = lookup.get(chat_idx)
        if human_tags is not None:
            filtered_true = [tag for tag in human_tags if tag in CASINO_STRATEGIES]
            agent_tags = list(strategy_classifier(agent_utt, []))
            strat_pred.append(agent_tags)
            strat_true.append(filtered_true)
            strat_detail.append({
                "dialogue_id":  dialogue_id,
                "turn_index":   chat_idx,
                "agent_utt":    agent_utt[:200],
                "human_utt":    gt_utt[:200],
                "pred_tags":    agent_tags,
                "true_tags":    filtered_true,
            })
            row["agent_strategies"] = agent_tags
            row["human_strategies"] = filtered_true

        per_turn_rows.append(row)

    # ── Aggregate
    accept_metrics = binary_accept_f1(accept_pairs)
    if bid_cosines:
        bid_metrics = {
            "mean":   float(np.mean(bid_cosines)),
            "median": float(np.median(bid_cosines)),
            "std":    float(np.std(bid_cosines)),
            "min":    float(np.min(bid_cosines)),
            "max":    float(np.max(bid_cosines)),
            "support": len(bid_cosines),
        }
    else:
        bid_metrics = {
            "mean":    float("nan"),
            "support": 0,
            "note":    "No turn had both an agent counter_offer AND a human Submit-Deal.",
        }
    macro_f1, per_label = macro_f1_multilabel(
        strat_pred, strat_true, label_set=CASINO_STRATEGIES,
    )
    strat_metrics = {
        "macro_f1":    macro_f1,
        "support":     len(strat_pred),
        "per_label":   per_label,
    }

    return {
        "accept":   accept_metrics,
        "bid_cosine": bid_metrics,
        "strategy": strat_metrics,
        "brier":    {
            "note": "Not applicable: StructuredCoT agent does not expose a "
                    "posterior over priority orderings. Plug in the "
                    "HybridTurnAgent for Brier.",
        },
        "counts": {
            "n_agent_turns":   len(turns),
            "n_agent_accepts": n_agent_accepts,
            "n_human_accepts": n_human_accepts,
            "n_human_submits": n_human_submits,
            "n_bid_pairs":     len(bid_cosines),
            "n_strategy_pairs": len(strat_pred),
        },
        "per_turn":       per_turn_rows,
        "bid_details":    bid_pair_details,
        "strategy_detail": strat_detail,
    }


# ── Formatter ──────────────────────────────────────────────────────────────


def _fmt(x: Any, n: int = 4) -> str:
    if isinstance(x, float):
        if np.isnan(x):
            return "n/a"
        return f"{x:.{n}f}"
    return str(x)


def print_summary(result: Mapping[str, Any], run_dir: Path) -> None:
    c = result["counts"]
    a = result["accept"]
    b = result["bid_cosine"]
    s = result["strategy"]

    print("=" * 72)
    print(f"Protocol 3 metrics — {run_dir}")
    print("=" * 72)
    print(f"  agent turns scored       : {c['n_agent_turns']}")
    print(f"  agent 'accept' decisions : {c['n_agent_accepts']}")
    print(f"  human Accept-Deal turns  : {c['n_human_accepts']}")
    print(f"  human Submit-Deal turns  : {c['n_human_submits']}")
    print()

    print("--- Accept F1 (positive = 'accept') ---")
    print(f"  F1        : {_fmt(a['f1'])}")
    print(f"  precision : {_fmt(a['precision'])}     "
          f"recall: {_fmt(a['recall'])}")
    print(f"  accuracy  : {_fmt(a['accuracy'])}     "
          f"support: {a['support']}")
    print(f"  confusion : {a.get('confusion')}")
    print()

    print(f"--- Bid cosine (support = {c['n_bid_pairs']}) ---")
    if c["n_bid_pairs"] == 0:
        print(f"  n/a — {b.get('note', '')}")
    else:
        print(f"  mean={_fmt(b['mean'])}  median={_fmt(b['median'])}  "
              f"std={_fmt(b['std'])}  min={_fmt(b['min'])}  max={_fmt(b['max'])}")
    print()

    print(f"--- Strategy macro-F1 over {len(CASINO_STRATEGIES)} CaSiNo tags "
          f"(support = {c['n_strategy_pairs']}) ---")
    print(f"  macro-F1  : {_fmt(s['macro_f1'])}")
    nonzero = sorted(
        ((lab, d) for lab, d in s["per_label"].items() if d["support"] > 0),
        key=lambda kv: -kv[1]["support"],
    )
    print(f"  per-label (support > 0):")
    print(f"    {'label':<22}{'f1':>8}{'support':>10}{'tp':>6}{'fp':>6}{'fn':>6}")
    for lab, d in nonzero:
        print(f"    {lab:<22}{_fmt(d['f1']):>8}{d['support']:>10}"
              f"{d['tp']:>6}{d['fp']:>6}{d['fn']:>6}")
    print()
    print("--- Brier score ---")
    print(f"  {result['brier']['note']}")
    print("=" * 72)


# ── CLI ────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Protocol 1 run directory (contains turns.jsonl).")
    parser.add_argument("--annotations", type=Path,
                        default=Path("CaSiNo/data/casino_ann.json"),
                        help="Source of human strategy labels.")
    parser.add_argument("--data", type=Path,
                        default=Path("data/casino_test.json"),
                        help="Dialogue source (needed to align annotations).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to write protocol3_metrics.json "
                             "(default: <run-dir>/protocol3_metrics.json).")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    turns_path = args.run_dir / "turns.jsonl"
    if not turns_path.exists():
        print(f"No turns.jsonl at {turns_path}", file=sys.stderr)
        return 2

    turns = _read_jsonl(turns_path)
    logger.info("Loaded %d turn records.", len(turns))

    dialogue_ids = sorted({t.get("dialogue_id") for t in turns})
    ann_lookups = _load_annotation_lookups(
        dialogue_ids, ann_path=args.annotations, data_path=args.data,
    )
    matched = sum(1 for did in dialogue_ids if ann_lookups.get(did))
    logger.info("Loaded annotations for %d / %d dialogues.",
                matched, len(dialogue_ids))

    result = evaluate(turns, ann_lookups)
    print_summary(result, args.run_dir)

    out_path = args.output or (args.run_dir / "protocol3_metrics.json")
    slim = {
        "accept":   result["accept"],
        "bid_cosine": result["bid_cosine"],
        "strategy": {
            "macro_f1":  result["strategy"]["macro_f1"],
            "support":   result["strategy"]["support"],
            "per_label": result["strategy"]["per_label"],
        },
        "brier":    result["brier"],
        "counts":   result["counts"],
    }
    with out_path.open("w") as f:
        json.dump(slim, f, indent=2)
    logger.info("Wrote %s", out_path)

    detail_path = args.run_dir / "protocol3_turn_detail.jsonl"
    with detail_path.open("w") as f:
        for row in result["per_turn"]:
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %s", detail_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
