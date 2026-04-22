"""Prompt builder for the Structured-CoT CaSiNo negotiator.

A single ``build_prompt(agent_state, dialogue_history)`` entry point.
System prompt is intentionally kept under ~400 words — Llama-3 follows
short structured instructions more reliably, and the user-turn block
already carries the per-turn context.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence, Tuple

ITEMS = ("Food", "Water", "Firewood")


SYSTEM_PROMPT = """\
You are a negotiator in the CaSiNo camping game. You and another participant
divide 3 packages of Food, 3 of Water, and 3 of Firewood for a camping trip.

Scoring (YOUR points):
- Each item you receive scores (count x priority weight): High = 5 pts,
  Medium = 4 pts, Low = 3 pts.
- Example: 2 Food (High) + 1 Water (Low) + 0 Firewood = 2*5 + 1*3 + 0 = 13.
- Walk-away (no deal) gives you a fallback of 5 pts. Walking away is the
  right call only when the opponent refuses to give any of your top-priority
  items and continuing would score under 10.
- Max per side = 3*5 + 3*4 + 3*3 = 36. Realistic deals score 16-26.

A concrete OFFER is on the table only when the opponent's most recent message
proposes specific quantities (e.g. "you get 2 food and 1 water, I get the
rest"). Vague hints, questions, or preferences are conversational moves, not
offers.

Every turn you MUST emit exactly these five XML-tagged blocks, in order,
with no commentary before or after:

<observation>
What just happened. The opponent's last utterance, any offer they made,
any preferences they revealed, any emotional/rapport signals.
</observation>

<opponent_inference>
What do you believe about the opponent's priority ordering over Food, Water,
Firewood? Which ordering is most likely and why? If uncertain, name the two
most plausible orderings and the evidence that would distinguish them.
</opponent_inference>

<plan>
Given YOUR priorities and your belief about theirs:
- Is there a concrete offer on the table? If so, should you accept it?
  Evaluate the points it would give you vs. the walk-away fallback (5).
- If countering, what split maximizes your points while being plausibly
  acceptable to the opponent given what you inferred above?
- What is your conversational intent (counter-propose, probe priorities,
  argue for your top item, build rapport)?
- Cite your <opponent_inference> in one sentence so the plan stays
  coherent with your stated belief.
</plan>

<utterance>
A natural 1-3 sentence message that executes the plan.
</utterance>

<decision>
JSON only, no prose:
  {"action": "accept" | "reject" | "walkaway",
   "counter_offer": {"Food": 0-3, "Water": 0-3, "Firewood": 0-3} or null}
Rules:
- "accept" only when a concrete opponent offer is on the table AND you take it.
  counter_offer must be null.
- "walkaway" ends the negotiation. counter_offer must be null.
- "reject" means you are NOT accepting (either countering or still talking).
  Use counter_offer = {Food, Water, Firewood} with the counts YOU receive
  (opponent gets 3 minus each) whenever you want to propose a split.
  Use counter_offer = null when you are only making a conversational move.
</decision>
"""


USER_PROMPT_TEMPLATE = """\
## Your side

Your priorities (most to least important) with the reasons you wrote at the
start of the game:

- High priority   : {high_item}  ({high_reason})
- Medium priority : {med_item}  ({med_reason})
- Low priority    : {low_item}  ({low_reason})

Per-item point values for YOU:
- {high_item}: 5 pts each
- {med_item}: 4 pts each
- {low_item}: 3 pts each

## Dialogue so far (turn {turn_index})

{history_block}

## Task

Produce your five tagged blocks for THIS turn, in order, and stop. Do not
restate these instructions. Do not add any text before <observation> or
after </decision>.
"""


def _format_history(
    dialogue_history: Sequence[Tuple[str, str]],
    *,
    max_turns: int = 30,
    me_label: str = "You",
    opp_label: str = "Opponent",
) -> str:
    if not dialogue_history:
        return "(conversation has not started yet — you are speaking first)"

    recent = dialogue_history[-max_turns:]
    lines: List[str] = []
    for speaker, utterance in recent:
        role = (speaker or "").lower()
        if role in ("me", "you", "agent", "self"):
            tag = me_label
        elif role in ("opp", "opponent", "them", "other"):
            tag = opp_label
        else:
            tag = str(speaker)
        lines.append(f"{tag}: {utterance}")
    if len(dialogue_history) > max_turns:
        lines.insert(
            0, f"(... {len(dialogue_history) - max_turns} earlier turns omitted ...)",
        )
    return "\n".join(lines)


def build_prompt(
    agent_state: Mapping[str, Any],
    dialogue_history: Sequence[Tuple[str, str]],
    *,
    include_system: bool = True,
) -> str:
    """Render the full prompt for one turn.

    Args:
        agent_state: dict with keys ``priorities``, ``arguments``, ``turn_index``.
            * ``priorities``: ``{"High": "Food", "Medium": "Water", "Low": "Firewood"}``
            * ``arguments``:  ``{"High": "...", "Medium": "...", "Low": "..."}``
              (free-text justifications from CaSiNo's ``value2reason``).
            * ``turn_index``: 0-indexed count of turns the agent has taken so far.
        dialogue_history: list of ``(speaker, utterance)`` pairs, oldest first.
            Speaker is ``"me"`` / ``"agent"`` for the agent's own prior turns
            and ``"opp"`` / ``"opponent"`` for the opponent's turns. Deal
            actions (e.g. ``"Submit-Deal: {Food:2,...}"``) should be pre-
            rendered as utterances by the caller so the LLM sees the exact
            offer text.
        include_system: prepend the system block (default True). Set False
            if you are already feeding the system prompt via
            ``LlamaClient(system_prompt=...)``.
    """
    priorities = dict(agent_state.get("priorities") or {})
    arguments = dict(agent_state.get("arguments") or {})
    turn_index = int(agent_state.get("turn_index", 0))

    missing = [k for k in ("High", "Medium", "Low") if k not in priorities]
    if missing:
        raise ValueError(
            f"agent_state.priorities is missing keys {missing}; "
            f"expected High/Medium/Low. Got: {priorities}"
        )

    user_block = USER_PROMPT_TEMPLATE.format(
        high_item=priorities["High"],
        high_reason=(arguments.get("High") or "no reason recorded"),
        med_item=priorities["Medium"],
        med_reason=(arguments.get("Medium") or "no reason recorded"),
        low_item=priorities["Low"],
        low_reason=(arguments.get("Low") or "no reason recorded"),
        turn_index=turn_index,
        history_block=_format_history(dialogue_history),
    )

    if include_system:
        return SYSTEM_PROMPT.rstrip() + "\n\n" + user_block
    return user_block


__all__ = [
    "ITEMS",
    "SYSTEM_PROMPT",
    "build_prompt",
]
