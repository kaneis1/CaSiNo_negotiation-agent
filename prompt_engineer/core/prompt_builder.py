#!/usr/bin/env python3
"""Utterance generation prompt builder for CaSiNo negotiation agent.

Constructs the LLM prompt that turns selected strategies, bidding targets,
and opponent model state into a natural-sounding negotiation utterance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ── Strategy → instruction mapping ─────────────────────────────────────────

def _strategy_instruction(
    strategy: str,
    agent_priorities: Dict[str, str],
    agent_reasons: Dict[str, str],
    proposed_offer: Optional[Dict[str, int]],
) -> Optional[str]:
    """Map a strategy label to a concrete prompt instruction."""

    high_item = agent_priorities["High"]
    med_item = agent_priorities["Medium"]
    low_item = agent_priorities["Low"]

    if strategy == "small-talk":
        return (
            "Include friendly small talk (camping excitement, ask how "
            "they're doing, wish them a good trip)."
        )

    if strategy == "self-need":
        return (
            f"Argue for {high_item} using your personal reason: "
            f"'{agent_reasons['High']}'"
        )

    if strategy == "other-need":
        return (
            f"Argue for {high_item} by mentioning others' needs based on: "
            f"'{agent_reasons['High']}'"
        )

    if strategy == "no-need":
        return (
            f"Mention that you don't really need extra {low_item}: "
            f"'{agent_reasons['Low']}'"
        )

    if strategy == "elicit-pref":
        return "Ask the opponent what items they need most or least."

    if strategy == "promote-coordination":
        if proposed_offer:
            offer_parts = ", ".join(
                f"{v} {k}" for k, v in proposed_offer.items() if v > 0
            )
            them_parts = ", ".join(
                f"{3 - v} {k}" for k, v in proposed_offer.items() if 3 - v > 0
            )
            return (
                f"Propose a trade: you get {offer_parts}, "
                f"they get {them_parts}."
            )
        return (
            "Suggest working together to find a mutually beneficial deal."
        )

    if strategy == "vouch-fair":
        return (
            "Politely point out that the current deal isn't balanced for "
            "you. Be constructive, not accusatory."
        )

    if strategy == "showing-empathy":
        return (
            "Acknowledge the opponent's needs or situation with genuine "
            "empathy before making your point."
        )

    if strategy == "uv-part":
        return (
            "Gently suggest the opponent might not need as much of the "
            "item they're claiming. Be tactful."
        )

    return None


# ── Opponent model summary for prompt ──────────────────────────────────────

def _opponent_summary(opponent_model: Any) -> str:
    """Summarize what we know about the opponent for internal context."""
    if opponent_model is None or opponent_model.confidence == "low":
        return "You don't know much about the opponent's priorities yet."

    pred = opponent_model.get_predicted_priorities()
    conf = opponent_model.confidence
    style = opponent_model.style

    lines = [f"Based on the conversation (confidence: {conf}), you believe:"]
    lines.append(f"- Opponent's top priority is probably {pred['High']}")
    lines.append(f"- Opponent's lowest priority is probably {pred['Low']}")
    if style != "unknown":
        lines.append(f"- Opponent seems {style} in their negotiation style")
    lines.append(
        "Use this to find mutually beneficial trades, but don't reveal "
        "that you've figured out their priorities."
    )
    return "\n".join(lines)


# ── Main prompt builder ────────────────────────────────────────────────────

def build_generation_prompt(
    agent_priorities: Dict[str, str],
    agent_reasons: Dict[str, str],
    history: List[Dict[str, str]],
    selected_strategies: List[str],
    proposed_offer: Optional[Dict[str, int]],
    opponent_model: Any = None,
    phase: str = "bargaining",
) -> str:
    """Build the full utterance generation prompt.

    Args:
        agent_priorities: {"High": "Food", "Medium": "Water", "Low": "Firewood"}.
        agent_reasons: {"High": "reason...", "Medium": "reason...", "Low": "reason..."}.
        history: [{"role": "agent"/"opponent", "text": str}, ...].
        selected_strategies: Strategy labels from StrategySelector.
        proposed_offer: {"Food": 2, "Water": 1, "Firewood": 0} — what the
            agent wants for itself this turn (from BiddingStrategy).
        opponent_model: OpponentModel instance (optional).
        phase: Negotiation phase from StrategySelector.

    Returns:
        Complete prompt string ready for LLM generation.
    """
    high_item = agent_priorities["High"]
    med_item = agent_priorities["Medium"]
    low_item = agent_priorities["Low"]

    history_text = "\n".join(
        f"{'You' if h['role'] == 'agent' else 'Opponent'}: {h['text']}"
        for h in history
    ) if history else "(This is the start of the conversation.)"

    strategy_instructions: List[str] = []
    for s in selected_strategies:
        inst = _strategy_instruction(s, agent_priorities, agent_reasons, proposed_offer)
        if inst:
            strategy_instructions.append(inst)

    strategy_block = "\n".join(f"- {inst}" for inst in strategy_instructions)

    opp_summary = _opponent_summary(opponent_model)

    prompt = f"""\
You are negotiating for camping supplies in a friendly conversation.
You and your partner each need to divide 3 packages each of Food, Water, and Firewood.

YOUR PRIORITIES (keep secret — never reveal point values):
- Most important: {high_item} (reason: {agent_reasons['High']})
- Somewhat important: {med_item} (reason: {agent_reasons['Medium']})
- Least important: {low_item} (reason: {agent_reasons['Low']})

WHAT YOU KNOW ABOUT THE OPPONENT:
{opp_summary}

CONVERSATION SO FAR:
{history_text}

YOUR TASK THIS TURN:
Phase: {phase}
{strategy_block}

RULES:
- Write 1-3 natural sentences. Sound like a real person chatting, not a formal agent.
- Use contractions and casual language. Emojis are okay occasionally.
- NEVER reveal your exact priority rankings or point values.
- NEVER be rude or dismissive.
- Reference your personal reasons naturally, don't just state them verbatim.
- If making an offer, state it clearly: "How about I get X food, Y water, Z firewood?"

Respond with ONLY your next message (no labels, no reasoning):"""

    return prompt


# ── Deal proposal / acceptance prompts ─────────────────────────────────────

def build_deal_proposal_prompt(
    agent_priorities: Dict[str, str],
    history: List[Dict[str, str]],
    proposed_offer: Dict[str, int],
) -> str:
    """Build prompt for the agent to phrase a final deal submission."""
    offer_parts = ", ".join(f"{v} {k}" for k, v in proposed_offer.items() if v > 0)
    them_parts = ", ".join(
        f"{3 - v} {k}" for k, v in proposed_offer.items() if 3 - v > 0
    )

    history_text = "\n".join(
        f"{'You' if h['role'] == 'agent' else 'Opponent'}: {h['text']}"
        for h in history[-4:]
    )

    return f"""\
You are about to submit a final deal in a camping supply negotiation.

Recent conversation:
{history_text}

FINAL DEAL TO PROPOSE:
- You get: {offer_parts}
- They get: {them_parts}

Write a brief, friendly message confirming this deal. Be warm and positive.
Keep it to 1-2 sentences. End with something like "Deal?" or "Sound good?"

Respond with ONLY your message:"""


def build_deal_response_prompt(
    accept: bool,
    history: List[Dict[str, str]],
    proposed_offer: Optional[Dict[str, int]] = None,
) -> str:
    """Build prompt for accepting or rejecting a deal."""
    last_turn = history[-1]["text"] if history else ""
    offer_text = ""
    if proposed_offer:
        offer_parts = ", ".join(
            f"{count} {item}" for item, count in proposed_offer.items()
        )
        offer_text = f'\nSTRUCTURED OFFER FOR YOU:\n"{offer_parts}"\n'

    if accept:
        return f"""\
The opponent just proposed a deal:
"{last_turn}"
{offer_text}

You've decided to ACCEPT. Write a brief, warm acceptance message.
Keep it to 1-2 sentences. Be genuinely positive.

Respond with ONLY your message:"""

    return f"""\
The opponent just proposed a deal:
"{last_turn}"
{offer_text}

You've decided this doesn't work for you. Write a brief, polite counter
or explanation of why you'd like to adjust the deal.
Keep it to 1-2 sentences. Don't be rude.

Respond with ONLY your message:"""


# ── CLI demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    priorities = {"High": "Food", "Medium": "Water", "Low": "Firewood"}
    reasons = {
        "High": "We have a large group and need to feed everyone properly.",
        "Medium": "Staying hydrated is important since we'll be hiking a lot.",
        "Low": "We brought some matches and lighter fluid already.",
    }

    history = [
        {"role": "opponent", "text": "Hey there! Excited for camping? 🏕️"},
        {"role": "agent", "text": "Hi! Super excited! What items are most important to you?"},
        {"role": "opponent", "text": "I really need firewood — it gets cold at night and my kids need to stay warm."},
    ]

    from prompt_engineer.core.opponent_model import OpponentModel
    opp = OpponentModel()
    opp.update(
        "I really need firewood — it gets cold at night and my kids need to stay warm.",
        ["other-need", "self-need"],
        ["Firewood"],
    )

    offer = {"Food": 3, "Water": 2, "Firewood": 0}

    strategies = ["showing-empathy", "self-need", "promote-coordination"]

    prompt = build_generation_prompt(
        agent_priorities=priorities,
        agent_reasons=reasons,
        history=history,
        selected_strategies=strategies,
        proposed_offer=offer,
        opponent_model=opp,
        phase="bargaining",
    )

    print("═" * 70)
    print("GENERATION PROMPT")
    print("═" * 70)
    print(prompt)
    print()
    print("═" * 70)
    print("DEAL PROPOSAL PROMPT")
    print("═" * 70)
    print(build_deal_proposal_prompt(priorities, history, offer))
    print()
    print("═" * 70)
    print("DEAL ACCEPT PROMPT")
    print("═" * 70)
    print(build_deal_response_prompt(accept=True, history=history))
