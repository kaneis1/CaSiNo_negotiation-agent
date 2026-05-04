"""Prompt templates for the Hybrid Bayesian opponent model.

Two pieces of LLM work live here:

1. ``build_likelihood_prompt`` — score an opponent utterance under each of
   the 6 priority hypotheses (relative evidence only, no priors).
2. ``build_generation_prompt`` — generate the agent's next utterance
   conditioned on a natural-language posterior summary.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# ── Piece 1: LLM likelihood ────────────────────────────────────────────────

LIKELIHOOD_PROMPT_TEMPLATE = """\
You are an evidence model for negotiation opponent modeling in the CaSiNo
camping scenario. Two players divide 3 packages each of Food, Water, Firewood.

Task: Evaluate how consistent the opponent's latest utterance is with each
hypothesis about the opponent's priority ordering.

Rules:
- Do NOT compute a posterior.
- Do NOT use any prior beliefs.
- Evaluate each hypothesis independently.
- Use a 0 to 100 scale. 50 = utterance provides no information for or against
  this hypothesis. Above 50 = evidence for; below 50 = evidence against.
- Avoid near-identical scores across hypotheses UNLESS the utterance genuinely
  provides no priority signal.
- Avoid scores above 85 or below 15 unless the utterance is explicit and
  unambiguous.
- Hypotheses that agree on the top-priority item should receive similar scores
  when the utterance only informs about the top item.

Handling low-signal utterances:
- If the utterance contains no priority-relevant content (small talk, greetings,
  acknowledgments, coordination, empathy, non-negotiation topics), return all
  six scores close to 50 and set short_rationale to "no priority signal".

Dialogue so far (for context only — do not re-evaluate prior utterances):
{history}

Opponent's latest utterance: "{utterance}"

Hypotheses:
H1: Food > Water > Firewood
H2: Food > Firewood > Water
H3: Water > Food > Firewood
H4: Water > Firewood > Food
H5: Firewood > Food > Water
H6: Firewood > Water > Food

Return JSON only:
{{
  "evidence_scores": {{"H1": ..., "H2": ..., "H3": ..., "H4": ..., "H5": ..., "H6": ...}},
  "short_rationale": "..."
}}
"""


def build_likelihood_prompt(
    utterance: str,
    history: Optional[List[Dict[str, str]]] = None,
    max_context_turns: int = 6,
) -> str:
    """Render the likelihood-scoring prompt for a single opponent utterance."""
    history_block = _format_history(history, max_context_turns)
    return LIKELIHOOD_PROMPT_TEMPLATE.format(
        history=history_block,
        utterance=utterance.strip(),
    )


# ── Piece 3: Posterior-conditioned generation ──────────────────────────────

GENERATION_PROMPT_TEMPLATE = """\
You are a participant in the CaSiNo negotiation game. You and your partner
are dividing 3 Food, 3 Water, and 3 Firewood for an upcoming camping trip.

Your own priority ordering (most to least important):
  High:   {high}
  Medium: {medium}
  Low:    {low}

What you currently believe about the opponent's priorities:
{posterior_summary}

Recent dialogue:
{context}

Write the next message you (the negotiator) should send. Guidelines:
- Keep it natural, conversational, 1-3 sentences.
- Use what you know about the opponent's likely priorities to propose
  trades that are good for you AND plausibly good for them.
- If you are still very uncertain about the opponent, ask a brief
  clarifying question instead of locking in a deal.
- Do NOT reveal that you are tracking probabilities or hypotheses.

If you want to propose a concrete split, also fill the offer JSON.
Always return JSON only with this exact shape:

{{
  "utterance": "...",
  "offer": {{"Food": <0-3 or null>,
            "Water": <0-3 or null>,
            "Firewood": <0-3 or null>}}
}}

Set "offer" to null (not an object) if you are not yet proposing a split.
The offer counts are what YOU would keep; your partner gets the remainder.
"""


def build_generation_prompt(
    my_priorities: Dict[str, str],
    posterior_summary: str,
    history: Optional[List[Dict[str, str]]] = None,
    max_context_turns: int = 8,
) -> str:
    """Render the speak-prompt conditioned on the posterior summary."""
    context = _format_history(history, max_context_turns)
    return GENERATION_PROMPT_TEMPLATE.format(
        high=my_priorities.get("High", "?"),
        medium=my_priorities.get("Medium", "?"),
        low=my_priorities.get("Low", "?"),
        posterior_summary=posterior_summary,
        context=context,
    )


# ── Shared helpers ─────────────────────────────────────────────────────────


def _format_history(
    history: Optional[List[Dict[str, str]]],
    max_turns: int,
) -> str:
    """Format the most recent ``max_turns`` turns as ``Speaker: text`` lines."""
    if not history:
        return "(conversation start)"
    recent = history[-max_turns:]
    lines: List[str] = []
    for turn in recent:
        role = turn.get("role") or turn.get("speaker") or ""
        text = turn.get("text", "")
        if role in ("opp", "opponent"):
            speaker = "Opponent"
        elif role in ("me", "agent"):
            speaker = "You"
        else:
            speaker = role.capitalize() or "Speaker"
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)
