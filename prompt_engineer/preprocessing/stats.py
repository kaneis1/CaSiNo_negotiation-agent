#!/usr/bin/env python3
"""Compute dataset statistics for CaSiNo dialogues."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from prompt_engineer.preprocessing.scoring import LIKENESS_MAP, SATISFACTION_MAP

# ── Helpers to bridge raw CaSiNo format ────────────────────────────────────

DEAL_ACTIONS = {"Accept-Deal", "Reject-Deal", "Walk-Away", "Submit-Deal"}


def _deal_status(dialogue: Dict[str, Any]) -> str:
    last_text = dialogue["chat_logs"][-1]["text"]
    if last_text == "Accept-Deal":
        return "accepted"
    if last_text == "Walk-Away":
        return "walkedaway"
    return "unknown"


def _utterances(dialogue: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return only human-text turns (exclude deal actions)."""
    return [t for t in dialogue["chat_logs"] if t["text"] not in DEAL_ACTIONS]


def _is_annotated(dialogue: Dict[str, Any]) -> bool:
    return bool(dialogue.get("annotations"))


# ── Main stats function ────────────────────────────────────────────────────


def dataset_stats(dialogues: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics over a list of CaSiNo dialogue dicts."""

    stats: Dict[str, Any] = {
        "total": len(dialogues),
        "accepted": sum(1 for d in dialogues if _deal_status(d) == "accepted"),
        "walkedaway": sum(1 for d in dialogues if _deal_status(d) == "walkedaway"),
        "avg_turns": float(np.mean([len(_utterances(d)) for d in dialogues])),
        "avg_points": float(np.mean([
            info["outcomes"]["points_scored"]
            for d in dialogues
            for info in d["participant_info"].values()
        ])),
        "avg_satisfaction": float(np.mean([
            SATISFACTION_MAP[info["outcomes"]["satisfaction"]]
            for d in dialogues
            for info in d["participant_info"].values()
        ])),
        "avg_likeness": float(np.mean([
            LIKENESS_MAP[info["outcomes"]["opponent_likeness"]]
            for d in dialogues
            for info in d["participant_info"].values()
        ])),
    }

    # Strategy distribution (annotated dialogues only)
    strategy_counts: Dict[str, int] = defaultdict(int)
    total_utterances = 0
    for d in dialogues:
        if not _is_annotated(d):
            continue
        for utt_text, strategies_str in d["annotations"]:
            total_utterances += 1
            for s in strategies_str.split(","):
                strategy_counts[s.strip()] += 1

    if total_utterances > 0:
        stats["strategy_distribution"] = {
            k: round(v / total_utterances, 4)
            for k, v in sorted(strategy_counts.items(), key=lambda x: -x[1])
        }
    else:
        stats["strategy_distribution"] = {}

    stats["annotated_dialogues"] = sum(1 for d in dialogues if _is_annotated(d))
    stats["annotated_utterances"] = total_utterances

    return stats


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print CaSiNo dataset statistics.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent
        / "CaSiNo" / "data" / "casino.json",
        help="Path to a CaSiNo JSON file",
    )
    args = parser.parse_args()

    with args.input.open() as f:
        data = json.load(f)

    s = dataset_stats(data)

    print(f"{'Dialogues':.<30} {s['total']}")
    print(f"{'  Accepted':.<30} {s['accepted']}")
    print(f"{'  Walked away':.<30} {s['walkedaway']}")
    print(f"{'  Annotated':.<30} {s['annotated_dialogues']}")
    print(f"{'Avg turns per dialogue':.<30} {s['avg_turns']:.1f}")
    print(f"{'Avg points (per agent)':.<30} {s['avg_points']:.2f}")
    print(f"{'Avg satisfaction (1-5)':.<30} {s['avg_satisfaction']:.2f}")
    print(f"{'Avg likeness (1-5)':.<30} {s['avg_likeness']:.2f}")
    print(f"{'Annotated utterances':.<30} {s['annotated_utterances']}")
    print()
    print("Strategy distribution (fraction of annotated utterances):")
    for strategy, frac in s["strategy_distribution"].items():
        print(f"  {strategy:.<35} {frac:.4f}")
