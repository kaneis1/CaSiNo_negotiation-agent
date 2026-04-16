#!/usr/bin/env python3
"""Strategy selection module for CaSiNo negotiation agent.

Selects negotiation strategies based on conversation phase, opponent model
state, and current deal quality.  The selected strategies guide both the
LLM prompt and the bidding module.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ── Negotiation phases ─────────────────────────────────────────────────────

PHASES = ("opening", "exploration", "bargaining", "closing")


class StrategySelector:
    """Select strategies for each turn based on negotiation context.

    Args:
        my_priorities: {"High": "Food", "Medium": "Water", "Low": "Firewood"}.
            If provided, enables priority-aware bargaining decisions.
    """

    def __init__(self, my_priorities: Optional[Dict[str, str]] = None) -> None:
        self.my_priorities = my_priorities
        self.my_high = my_priorities["High"] if my_priorities else None
        self.my_low = my_priorities["Low"] if my_priorities else None

    def select(
        self,
        turn: int,
        max_turns: int,
        opponent_model: Any,
        current_deal_quality: str = "acceptable",
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[List[str], str]:
        """Select strategies for this turn.

        Args:
            turn: Current turn number (0-indexed).
            max_turns: Expected max turns in the negotiation.
            opponent_model: OpponentModel instance with confidence and predictions.
            current_deal_quality: "above_threshold" | "acceptable" | "below_threshold".
            history: Conversation history (for detecting repeated patterns).

        Returns:
            (strategy_list, phase) tuple.
        """
        phase = self._get_phase(turn, max_turns, opponent_model)
        strategies: List[str] = []

        if phase == "opening":
            strategies.append("small-talk")
            strategies.append("elicit-pref")

        elif phase == "exploration":
            if opponent_model.confidence == "low":
                strategies.append("elicit-pref")
            strategies.append("self-need")
            if self.my_low:
                strategies.append("no-need")

        elif phase == "bargaining":
            strategies.extend(
                self._bargaining_strategies(opponent_model, current_deal_quality, history)
            )

        elif phase == "closing":
            strategies.append("promote-coordination")
            strategies.append("small-talk")

        return strategies, phase

    # ── Phase detection ────────────────────────────────────────────────

    def _get_phase(
        self,
        turn: int,
        max_turns: int,
        opponent_model: Any,
    ) -> str:
        if turn <= 1:
            return "opening"
        if turn >= max_turns - 2:
            return "closing"
        if opponent_model.confidence == "low" and turn <= 4:
            return "exploration"
        return "bargaining"

    # ── Bargaining-phase logic ─────────────────────────────────────────

    def _bargaining_strategies(
        self,
        opponent_model: Any,
        deal_quality: str,
        history: Optional[List[Dict[str, str]]],
    ) -> List[str]:
        strategies: List[str] = []

        opp_pred = opponent_model.get_predicted_priorities()

        if self.my_high and opp_pred["High"] != self.my_high:
            strategies.append("promote-coordination")

        if deal_quality == "below_threshold":
            strategies.append("vouch-fair")

        strategies.append("self-need")

        if opponent_model.confidence in ("medium", "high"):
            if self._opponent_seems_empathetic(history):
                strategies.append("other-need")

        if self._opponent_overclaiming(opponent_model):
            strategies.append("uv-part")

        return strategies

    # ── Heuristic helpers ──────────────────────────────────────────────

    @staticmethod
    def _opponent_seems_empathetic(
        history: Optional[List[Dict[str, str]]],
    ) -> bool:
        """Check if opponent has shown empathetic signals."""
        if not history:
            return False
        empathy_cues = (
            "sorry", "understand", "tough", "hope you", "that's hard",
            "oh no", "feel for you",
        )
        for turn in history:
            if turn["role"] == "opponent":
                text_lower = turn["text"].lower()
                if any(cue in text_lower for cue in empathy_cues):
                    return True
        return False

    def _opponent_overclaiming(self, opponent_model: Any) -> bool:
        """Detect if opponent is trying to claim too much of everything."""
        scores = opponent_model.priority_scores
        high_items = sum(1 for v in scores.values() if v >= 2.0)
        return high_items >= 2

    # ── Strategy-to-prompt guidance ────────────────────────────────────

    @staticmethod
    def strategy_guidance(strategies: List[str], phase: str) -> str:
        """Convert selected strategies into natural-language prompt guidance.

        This text can be injected into the LLM system prompt to steer the
        agent's response style.
        """
        guidance_map = {
            "small-talk": "Be friendly and build rapport with casual conversation.",
            "elicit-pref": (
                "Ask what the opponent values most or least to discover "
                "their priorities."
            ),
            "self-need": (
                "Explain why you personally need your high-priority items. "
                "Be specific about your situation."
            ),
            "other-need": (
                "Mention the needs of your family, group, or pets to "
                "strengthen your argument."
            ),
            "no-need": (
                "Mention that you don't really need your low-priority item "
                "to signal willingness to trade it."
            ),
            "promote-coordination": (
                "Propose a concrete trade or express willingness to work "
                "together for a fair deal."
            ),
            "vouch-fair": (
                "Point out that the current proposal is unbalanced and "
                "appeal to fairness."
            ),
            "uv-part": (
                "Gently challenge whether the opponent truly needs the item "
                "they are claiming."
            ),
            "showing-empathy": (
                "Acknowledge the opponent's situation and show understanding."
            ),
        }

        phase_prefix = {
            "opening": "You are opening the negotiation.",
            "exploration": "You are exploring each other's needs.",
            "bargaining": "You are in the main bargaining phase.",
            "closing": "You are closing the negotiation.",
        }

        lines = [phase_prefix.get(phase, "")]
        lines.append("In your response, you should:")
        for s in strategies:
            if s in guidance_map:
                lines.append(f"- {guidance_map[s]}")

        return "\n".join(lines)


# ── CLI demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from prompt_engineer.core.opponent_model import OpponentModel

    priorities = {"High": "Food", "Medium": "Water", "Low": "Firewood"}
    selector = StrategySelector(my_priorities=priorities)
    opp = OpponentModel()

    print(f"My priorities: {priorities}\n")

    header = f"{'Turn':>4}  {'Phase':<13}  {'Strategies'}"
    print(header)
    print("─" * 70)

    for turn in range(11):
        strategies, phase = selector.select(
            turn=turn,
            max_turns=10,
            opponent_model=opp,
            current_deal_quality="acceptable",
        )
        print(f"{turn:>4}  {phase:<13}  {', '.join(strategies)}")

        if turn == 2:
            opp.update("I really need firewood for my kids",
                       ["other-need"], ["Firewood"])
        if turn == 4:
            opp.update("Water is important to stay hydrated",
                       ["self-need"], ["Water"])

    print()
    print("── Prompt guidance at turn 5 (bargaining) ──")
    strategies, phase = selector.select(5, 10, opp, "below_threshold")
    print(selector.strategy_guidance(strategies, phase))
