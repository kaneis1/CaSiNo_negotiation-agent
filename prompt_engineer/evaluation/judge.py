#!/usr/bin/env python3
"""LLM-as-judge module for evaluating negotiation quality.

Predicts the opponent's satisfaction and likeness ratings on CaSiNo's
original 5-point text scales, then converts to numeric for scoring.
Can be validated against ground-truth outcomes in the dataset.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from prompt_engineer.preprocessing.scoring import LIKENESS_MAP, SATISFACTION_MAP

DEAL_ACTIONS = {"Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away"}

# Reverse maps for fuzzy matching
_SAT_KEYS = list(SATISFACTION_MAP.keys())
_LIK_KEYS = list(LIKENESS_MAP.keys())


# ── Response parsing ───────────────────────────────────────────────────────

def _fuzzy_match(text: str, valid_options: List[str], default: str) -> str:
    """Best-effort match of LLM output to one of the valid scale values."""
    text_lower = text.strip().lower()
    for opt in valid_options:
        if opt.lower() in text_lower:
            return opt
    for opt in valid_options:
        if text_lower in opt.lower():
            return opt
    return default


def _parse_judge_response(response: str) -> Dict[str, str]:
    """Parse satisfaction and likeness from LLM judge output."""
    sat_text = "Undecided"
    lik_text = "Undecided"

    for line in response.split("\n"):
        line_stripped = line.strip()
        if line_stripped.lower().startswith("satisfaction"):
            raw = line_stripped.split(":", 1)[-1].strip()
            sat_text = _fuzzy_match(raw, _SAT_KEYS, "Undecided")
        elif line_stripped.lower().startswith("likeness") or \
             line_stripped.lower().startswith("like"):
            raw = line_stripped.split(":", 1)[-1].strip()
            lik_text = _fuzzy_match(raw, _LIK_KEYS, "Undecided")

    return {"satisfaction_text": sat_text, "likeness_text": lik_text}


# ── Judge function ─────────────────────────────────────────────────────────

def judge_satisfaction_and_likeness(
    dialogue_text: str,
    agent_role: str,
    deal_outcome: str,
    llm_client: Any,
) -> Dict[str, Any]:
    """LLM judge that predicts opponent's satisfaction and likeness.

    Args:
        dialogue_text: Formatted conversation transcript.
        agent_role: The agent being evaluated ("mturk_agent_1" or "mturk_agent_2").
        deal_outcome: Description of the final deal.
        llm_client: Object with .generate(prompt: str) -> str.

    Returns:
        {"satisfaction": int (1-5), "satisfaction_text": str,
         "likeness": int (1-5), "likeness_text": str}
    """
    opponent_role = (
        "mturk_agent_2" if agent_role == "mturk_agent_1" else "mturk_agent_1"
    )

    prompt = f"""\
You are evaluating a negotiation from the CaSiNo dataset.

DIALOGUE:
{dialogue_text}

FINAL DEAL: {deal_outcome}

Rate from {opponent_role}'s perspective (the opponent of {agent_role}):

1. Satisfaction with outcome (choose one):
   - Extremely dissatisfied
   - Slightly dissatisfied
   - Undecided
   - Slightly satisfied
   - Extremely satisfied

2. How much they like {agent_role} as a person (choose one):
   - Extremely dislike
   - Slightly dislike
   - Undecided
   - Slightly like
   - Extremely like

Consider: Was {agent_role} respectful? Did they acknowledge the opponent's needs?
Was the deal fair to the opponent? Did {agent_role} use manipulative tactics?

Return ONLY two lines:
Satisfaction: [your choice]
Likeness: [your choice]"""

    response = llm_client.generate(prompt)
    parsed = _parse_judge_response(response)

    return {
        "satisfaction": SATISFACTION_MAP.get(parsed["satisfaction_text"], 3),
        "satisfaction_text": parsed["satisfaction_text"],
        "likeness": LIKENESS_MAP.get(parsed["likeness_text"], 3),
        "likeness_text": parsed["likeness_text"],
    }


# ── Helpers to format dialogues for the judge ──────────────────────────────

def format_dialogue_for_judge(dialogue: Dict[str, Any]) -> str:
    """Format a CaSiNo dialogue as readable text for the judge prompt."""
    lines: List[str] = []
    for turn in dialogue["chat_logs"]:
        if turn["text"] in DEAL_ACTIONS:
            lines.append(f"[{turn['id']}: {turn['text']}]")
        else:
            speaker = turn["id"]
            lines.append(f"{speaker}: {turn['text']}")
    return "\n".join(lines)


def format_deal_outcome(dialogue: Dict[str, Any]) -> str:
    """Extract and format the final deal from a dialogue."""
    chat_logs = dialogue["chat_logs"]
    last = chat_logs[-1]

    if last["text"] == "Walk-Away":
        return "No deal — one participant walked away. Both get 5 points."

    for turn in reversed(chat_logs):
        if turn["text"] == "Submit-Deal":
            td = turn["task_data"]
            you = ", ".join(f"{v} {k}" for k, v in td["issue2youget"].items() if int(v) > 0)
            them = ", ".join(f"{v} {k}" for k, v in td["issue2theyget"].items() if int(v) > 0)
            return (
                f"Deal accepted. {turn['id']} gets: {you}. "
                f"Other gets: {them}."
            )

    return "Deal outcome unclear."


# ── Batch validation against ground truth ──────────────────────────────────

def validate_judge(
    dialogues: List[Dict[str, Any]],
    llm_client: Any,
    max_dialogues: Optional[int] = None,
) -> Dict[str, Any]:
    """Compare LLM judge predictions to actual satisfaction/likeness scores.

    Runs the judge on each dialogue for both roles, then compares to
    the ground-truth outcomes recorded in participant_info.

    Returns:
        Dict with MAE, correlation, exact-match rate, and per-evaluation details.
    """
    if max_dialogues:
        dialogues = dialogues[:max_dialogues]

    results: List[Dict[str, Any]] = []

    for d_idx, dialogue in enumerate(dialogues):
        dialogue_text = format_dialogue_for_judge(dialogue)
        deal_text = format_deal_outcome(dialogue)
        pinfo = dialogue["participant_info"]

        for agent_role in ("mturk_agent_1", "mturk_agent_2"):
            opponent_role = (
                "mturk_agent_2" if agent_role == "mturk_agent_1"
                else "mturk_agent_1"
            )

            gt_outcomes = pinfo[opponent_role]["outcomes"]
            gt_sat = SATISFACTION_MAP[gt_outcomes["satisfaction"]]
            gt_lik = LIKENESS_MAP[gt_outcomes["opponent_likeness"]]

            pred = judge_satisfaction_and_likeness(
                dialogue_text, agent_role, deal_text, llm_client,
            )

            results.append({
                "dialogue_id": dialogue["dialogue_id"],
                "agent_role": agent_role,
                "pred_satisfaction": pred["satisfaction"],
                "gt_satisfaction": gt_sat,
                "pred_likeness": pred["likeness"],
                "gt_likeness": gt_lik,
                "sat_error": abs(pred["satisfaction"] - gt_sat),
                "lik_error": abs(pred["likeness"] - gt_lik),
            })

        if (d_idx + 1) % 10 == 0:
            print(f"  Judged {d_idx + 1}/{len(dialogues)} dialogues")

    sat_errors = [r["sat_error"] for r in results]
    lik_errors = [r["lik_error"] for r in results]
    sat_exact = sum(1 for r in results if r["sat_error"] == 0)
    lik_exact = sum(1 for r in results if r["lik_error"] == 0)
    sat_within1 = sum(1 for r in results if r["sat_error"] <= 1)
    lik_within1 = sum(1 for r in results if r["lik_error"] <= 1)

    pred_sats = [r["pred_satisfaction"] for r in results]
    gt_sats = [r["gt_satisfaction"] for r in results]
    pred_liks = [r["pred_likeness"] for r in results]
    gt_liks = [r["gt_likeness"] for r in results]

    sat_corr = float(np.corrcoef(pred_sats, gt_sats)[0, 1]) if len(set(pred_sats)) > 1 else 0.0
    lik_corr = float(np.corrcoef(pred_liks, gt_liks)[0, 1]) if len(set(pred_liks)) > 1 else 0.0

    n = len(results)
    return {
        "num_evaluations": n,
        "satisfaction": {
            "mae": float(np.mean(sat_errors)),
            "exact_match": sat_exact / n,
            "within_1": sat_within1 / n,
            "correlation": sat_corr,
        },
        "likeness": {
            "mae": float(np.mean(lik_errors)),
            "exact_match": lik_exact / n,
            "within_1": lik_within1 / n,
            "correlation": lik_corr,
        },
        "per_evaluation": results,
    }


def print_judge_results(results: Dict[str, Any]) -> None:
    """Pretty-print judge validation results."""
    print(f"\nLLM Judge Validation ({results['num_evaluations']} evaluations)\n")

    for metric_name, label in [("satisfaction", "Satisfaction"), ("likeness", "Likeness")]:
        m = results[metric_name]
        print(f"{label}:")
        print(f"  {'MAE':.<30} {m['mae']:.3f}")
        print(f"  {'Exact match':.<30} {m['exact_match']:.1%}")
        print(f"  {'Within ±1':.<30} {m['within_1']:.1%}")
        print(f"  {'Correlation':.<30} {m['correlation']:.3f}")
        print()


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data_path = (
        Path(__file__).resolve().parents[2] / "CaSiNo" / "data" / "casino.json"
    )
    with data_path.open() as f:
        data = json.load(f)

    d = data[0]
    print("Example dialogue formatting:\n")
    print(format_dialogue_for_judge(d)[:500])
    print("...\n")
    print(f"Deal outcome: {format_deal_outcome(d)}")

    print("\nGround truth (opponent of mturk_agent_1 = mturk_agent_2):")
    gt = d["participant_info"]["mturk_agent_2"]["outcomes"]
    print(f"  Satisfaction: {gt['satisfaction']} → {SATISFACTION_MAP[gt['satisfaction']]}")
    print(f"  Likeness:     {gt['opponent_likeness']} → {LIKENESS_MAP[gt['opponent_likeness']]}")

    print("\nTo run judge validation:")
    print(
        "  from prompt_engineer.evaluation.judge import "
        "validate_judge, print_judge_results"
    )
    print("  results = validate_judge(dialogues, llm_client, max_dialogues=10)")
    print("  print_judge_results(results)")
