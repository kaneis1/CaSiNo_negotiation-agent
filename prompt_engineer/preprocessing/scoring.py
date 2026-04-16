#!/usr/bin/env python3
"""Satisfaction / likeness scale mappings and points calculation for CaSiNo."""

from __future__ import annotations

from typing import Any, Dict

# ── Text → numeric scale mappings ──────────────────────────────────────────

SATISFACTION_MAP = {
    "Extremely dissatisfied": 1,
    "Slightly dissatisfied": 2,
    "Undecided": 3,
    "Slightly satisfied": 4,
    "Extremely satisfied": 5,
}

LIKENESS_MAP = {
    "Extremely dislike": 1,
    "Slightly dislike": 2,
    "Undecided": 3,
    "Slightly like": 4,
    "Extremely like": 5,
}

# ── Points calculation ─────────────────────────────────────────────────────

PRIORITY_POINTS = {"High": 5, "Medium": 4, "Low": 3}

WALK_AWAY_POINTS = 5


def satisfaction_to_numeric(text: str) -> int:
    """Convert satisfaction text to 1-5 numeric scale."""
    return SATISFACTION_MAP[text]


def likeness_to_numeric(text: str) -> int:
    """Convert opponent-likeness text to 1-5 numeric scale."""
    return LIKENESS_MAP[text]


def calc_points(
    agent_id: str,
    deal_youget: Dict[str, str],
    deal_theyget: Dict[str, str],
    participant_info: Dict[str, Any],
    submitter_id: str,
) -> int:
    """Calculate points for a given agent from a deal.

    The submitter's perspective defines 'youget' and 'theyget'.
    If *agent_id* is the submitter, they receive issue2youget;
    otherwise they receive issue2theyget.
    """
    my_allocation = deal_youget if agent_id == submitter_id else deal_theyget

    priorities = participant_info[agent_id]["value2issue"]  # {"High": "Food", ...}
    issue2priority = {v: k for k, v in priorities.items()}  # {"Food": "High", ...}

    total = 0
    for item, count in my_allocation.items():
        priority = issue2priority[item]
        total += int(count) * PRIORITY_POINTS[priority]
    return total


def calc_points_from_dialogue(dialogue: Dict[str, Any]) -> Dict[str, int | None]:
    """Return {"mturk_agent_1": pts, "mturk_agent_2": pts} for a dialogue.

    Returns None values if the dialogue ended with Walk-Away.
    """
    chat_logs = dialogue["chat_logs"]
    participant_info = dialogue["participant_info"]
    last_turn = chat_logs[-1]

    if last_turn["text"] == "Walk-Away":
        return {aid: WALK_AWAY_POINTS for aid in participant_info}

    submit_turn = None
    for turn in reversed(chat_logs):
        if turn["text"] == "Submit-Deal":
            submit_turn = turn
            break

    if submit_turn is None:
        return {aid: None for aid in participant_info}

    deal_youget = submit_turn["task_data"]["issue2youget"]
    deal_theyget = submit_turn["task_data"]["issue2theyget"]
    submitter_id = submit_turn["id"]

    results = {}
    for aid in participant_info:
        if aid == submitter_id:
            my_allocation = deal_youget
        else:
            my_allocation = deal_theyget

        priorities = participant_info[aid]["value2issue"]
        issue2priority = {v: k for k, v in priorities.items()}

        total = 0
        for item, count in my_allocation.items():
            priority = issue2priority[item]
            total += int(count) * PRIORITY_POINTS[priority]
        results[aid] = total

    return results


# ── CLI quick-check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from pathlib import Path

    data_path = (
        Path(__file__).resolve().parents[2] / "CaSiNo" / "data" / "casino.json"
    )
    with data_path.open() as f:
        data = json.load(f)

    mismatches = 0
    for d in data:
        computed = calc_points_from_dialogue(d)
        for aid, pts in computed.items():
            recorded = d["participant_info"][aid]["outcomes"]["points_scored"]
            if pts != recorded:
                mismatches += 1
                print(
                    f"dialogue {d['dialogue_id']} {aid}: "
                    f"computed={pts} vs recorded={recorded}"
                )

    print(f"\nChecked {len(data)} dialogues, {mismatches} mismatches.")
