"""Prompt + target templates for the SFT'd 8B opponent model.

Used at *both* train time (sft_8b/data.py builds the chat rows) and
inference time (sft_8b/predict.py renders the same prompt for a fresh
dialogue prefix). Keeping a single source of truth here is critical:
any drift between train and inference distributions silently destroys
held-out accuracy.

Output schema the model is taught to emit:

    {"prefs": ["Firewood", "Food", "Water"],
     "satisfaction": "Extremely satisfied"}

Where:
    * ``prefs`` is a permutation of the three CaSiNo items
      [top, mid, low] from the OPPONENT's perspective.
    * ``satisfaction`` is one of the 5 ordinal CaSiNo categories,
      reflecting the SPEAKER's own end-of-game satisfaction.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Sequence

# CaSiNo's 5-class ordinal satisfaction scale (ordered low → high).
# Ordering matters for the satisfaction MAE metric in
# sft_8b/metrics_satisfaction.py; do not reshuffle.
SATISFACTION_LABELS: List[str] = [
    "Extremely dissatisfied",
    "Slightly dissatisfied",
    "Undecided",
    "Slightly satisfied",
    "Extremely satisfied",
]

# Negotiation control turns CaSiNo records as text but which carry no
# linguistic content; we drop them when building the dialogue prefix
# (matches the convention in opponent_model/metrics.py and
# prompt_engineer/scripts/generate_sft_data.py).
DEAL_ACTIONS = {"Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away"}

ITEMS = ["Food", "Water", "Firewood"]


SYSTEM_PROMPT = """\
You are an opponent-modeling assistant for the CaSiNo negotiation game.
Two players each pick up 3 packages of Food, Water, and Firewood for an
upcoming camping trip. Each player ranks the three items as High, Medium,
or Low priority (a permutation of the three).

You will see one player's perspective: their own priority ranking, their
free-text reasons for those priorities, and the dialogue so far (with
``Me:`` for that player and ``Opponent:`` for the other).

From this you must infer two things:

1. ``prefs``: the OPPONENT's priority ordering as
   ``[top, mid, low]`` over {Food, Water, Firewood}.
2. ``satisfaction``: how satisfied the player you are speaking for will
   be at the end of the negotiation. Pick exactly one of:
   "Extremely dissatisfied", "Slightly dissatisfied", "Undecided",
   "Slightly satisfied", "Extremely satisfied".

Reply with JSON only, no prose, in exactly this shape:

{"prefs": ["<item>", "<item>", "<item>"], "satisfaction": "<label>"}

Items must be a permutation of Food, Water, Firewood (no repeats, no
extras). Satisfaction must be one of the five exact strings above."""


# ── User prompt builder ────────────────────────────────────────────────────


def _format_priorities(value2issue: Mapping[str, str]) -> str:
    return (
        f"  High:   {value2issue.get('High', '?')}\n"
        f"  Medium: {value2issue.get('Medium', '?')}\n"
        f"  Low:    {value2issue.get('Low', '?')}"
    )


def _format_reasons(value2reason: Mapping[str, str]) -> str:
    if not value2reason:
        return "  (no reasons provided)"
    return "\n".join(
        f"  {level}: {value2reason.get(level, '').strip()}"
        for level in ("High", "Medium", "Low")
        if value2reason.get(level)
    )


def _format_chat_prefix(
    partial: Sequence[Mapping[str, Any]],
    me_role: str,
) -> str:
    lines: List[str] = []
    for turn in partial:
        text = turn.get("text", "")
        if not text or text in DEAL_ACTIONS:
            continue
        speaker = "Me" if turn.get("id") == me_role else "Opponent"
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines) if lines else "(conversation not yet started)"


def build_user_prompt(
    *,
    partial: Sequence[Mapping[str, Any]],
    my_priorities: Mapping[str, str],
    my_reasons: Mapping[str, str],
    me_role: str,
) -> str:
    """Render the per-snapshot user prompt.

    Same builder is used at train time (over a known dialogue prefix) and
    at inference time. The model's chat template wraps the SYSTEM_PROMPT
    around this; see sft_8b/predict.py.
    """
    return (
        "Your own priority ranking:\n"
        f"{_format_priorities(my_priorities)}\n\n"
        "Your own reasons for those priorities:\n"
        f"{_format_reasons(my_reasons)}\n\n"
        "Dialogue so far:\n"
        f"{_format_chat_prefix(partial, me_role)}\n\n"
        "Predict the opponent's priority ordering and your own end-of-game "
        "satisfaction. Respond with JSON only."
    )


# ── Target builder ─────────────────────────────────────────────────────────


def build_target_json(
    *,
    opp_value2issue: Mapping[str, str],
    my_satisfaction: str,
) -> str:
    """Build the canonical assistant response string from ground truth.

    The shape mirrors what the trained model is asked to emit, so the
    same parser (parse_response in sft_8b/predict.py) handles both
    training-time validation and inference-time decoding.
    """
    prefs = [
        opp_value2issue["High"],
        opp_value2issue["Medium"],
        opp_value2issue["Low"],
    ]
    return json.dumps(
        {"prefs": prefs, "satisfaction": my_satisfaction},
        ensure_ascii=False,
    )


__all__ = [
    "SYSTEM_PROMPT",
    "SATISFACTION_LABELS",
    "DEAL_ACTIONS",
    "ITEMS",
    "build_user_prompt",
    "build_target_json",
]
