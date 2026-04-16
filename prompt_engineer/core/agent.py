#!/usr/bin/env python3
"""Integrated LLM-based negotiation agent for CaSiNo.

Wires together strategy classification, opponent modeling, bidding,
strategy selection, and prompt building into a single agent that
implements the evaluation harness interface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from prompt_engineer.core.bidding import BiddingStrategy
from prompt_engineer.core.classify_strategy import classify_strategies, format_context
from prompt_engineer.core.opponent_model import OpponentModel, extract_mentioned_items
from prompt_engineer.core.prompt_builder import (
    build_deal_proposal_prompt,
    build_deal_response_prompt,
    build_generation_prompt,
)
from prompt_engineer.core.strategy_selector import StrategySelector
from prompt_engineer.evaluation.evaluate import NegotiationAgent
from prompt_engineer.preprocessing.scoring import WALK_AWAY_POINTS

DEAL_ACTIONS = {"Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away"}


class CaSiNoAgent(NegotiationAgent):
    """Full LLM-powered negotiation agent for the CaSiNo task.

    Orchestrates the complete negotiation loop:
    1. Receive opponent message → classify strategies → update opponent model
    2. Select strategies for this turn based on phase + opponent model
    3. Generate a bidding target from the Boulware curve
    4. Build a prompt and call the LLM to produce a natural utterance
    5. Handle Submit-Deal / Accept-Deal / Reject-Deal / Walk-Away flow

    Args:
        priorities: {"High": "Food", "Medium": "Water", "Low": "Firewood"}.
        reasons: {"High": "reason...", "Medium": "reason...", "Low": "reason..."}.
        llm_client: Object with .generate(prompt: str) -> str.
        personality: Big-Five scores (optional, for future persona tuning).
        svo: "prosocial" or "proself" (optional).
        beta: Boulware exponent (higher = more stubborn). Default 3.0.
        max_turns: Expected max conversation turns. Default 10.
    """

    def __init__(
        self,
        priorities: Dict[str, str],
        reasons: Dict[str, str],
        llm_client: Any,
        personality: Optional[Dict[str, float]] = None,
        svo: Optional[str] = None,
        beta: float = 3.0,
        max_turns: int = 10,
    ) -> None:
        self.llm_client = llm_client
        self.beta = beta
        self.max_turns = max_turns

        self.priorities: Dict[str, str] = {}
        self.reasons: Dict[str, str] = {}
        self.personality = personality
        self.svo = svo

        self.bidding = BiddingStrategy(priorities, beta=beta, max_turns=max_turns)
        self.opponent_model = OpponentModel()
        self.strategy_selector = StrategySelector(my_priorities=priorities)
        self._sync_profile(
            priorities=priorities,
            reasons=reasons,
            personality=personality,
            svo=svo,
        )

        self.history: List[Dict[str, str]] = []
        self.turn: int = 0
        self._last_phase: str = "opening"
        self._last_strategies: List[str] = []

    # ── NegotiationAgent interface (for evaluate.py) ───────────────────

    def generate(
        self,
        history: List[Dict[str, str]],
        priorities: Dict[str, str],
        reasons: Dict[str, str],
        personality: Optional[Dict[str, float]] = None,
        svo: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Stateless generate for the evaluator harness.

        The evaluator manages history externally, so we sync our internal
        state before generating.
        """
        self._sync_profile(
            priorities=priorities,
            reasons=reasons,
            personality=personality,
            svo=svo,
        )
        self.history = list(history)
        self.turn = sum(1 for h in history if h["role"] == "agent")

        if history and history[-1]["role"] == "opponent":
            self._process_opponent_turn(
                history[-1]["text"],
                context_history=history[:-1],
            )

        return self.generate_response()

    def propose_deal(
        self,
        history: List[Dict[str, str]],
        priorities: Dict[str, str],
    ) -> Optional[Dict[str, int]]:
        """Propose a final deal for the evaluator."""
        if priorities != self.priorities:
            self._sync_profile(
                priorities=priorities,
                reasons=self.reasons,
                personality=self.personality,
                svo=self.svo,
            )
        deal = self.bidding.generate_offer(self.turn, self.opponent_model)
        return deal["me"]

    def get_opponent_model_accuracy(
        self,
        true_opponent_priorities: Dict[str, str],
    ) -> Optional[float]:
        return self.opponent_model.get_accuracy(true_opponent_priorities)

    def reset(self) -> None:
        self.history.clear()
        self.turn = 0
        self._last_phase = "opening"
        self._last_strategies = []
        self.opponent_model.reset()

    # ── Stateful conversation API ──────────────────────────────────────

    def receive_opponent_message(self, text: str) -> Dict[str, Any]:
        """Process an incoming opponent utterance.

        Updates history, classifies strategies, updates opponent model.

        Returns:
            Summary dict with detected strategies, mentioned items, and
            updated opponent model state.
        """
        self.history.append({"role": "opponent", "text": text})
        info = self._process_opponent_turn(text, context_history=self.history[:-1])
        self.turn += 1
        return info

    def generate_response(self) -> Dict[str, Any]:
        """Generate the agent's next utterance.

        Returns:
            {"text": str, "strategies_used": list, "offer": dict|None,
             "phase": str, "deal_score": int}
        """
        deal_quality = self._assess_deal_quality()

        strategies, phase = self.strategy_selector.select(
            turn=self.turn,
            max_turns=self.bidding.max_turns,
            opponent_model=self.opponent_model,
            current_deal_quality=deal_quality,
            history=self.history,
        )

        offer: Optional[Dict[str, int]] = None
        if phase in ("bargaining", "closing"):
            deal = self.bidding.generate_offer(self.turn, self.opponent_model)
            offer = deal["me"]

        prompt = build_generation_prompt(
            agent_priorities=self.priorities,
            agent_reasons=self.reasons,
            history=self.history,
            selected_strategies=strategies,
            proposed_offer=offer,
            opponent_model=self.opponent_model,
            phase=phase,
        )

        response_text = self.llm_client.generate(prompt)
        self.history.append({"role": "agent", "text": response_text})

        self._last_phase = phase
        self._last_strategies = strategies
        self.turn += 1

        result: Dict[str, Any] = {
            "text": response_text,
            "strategies_used": strategies,
            "offer": offer,
            "phase": phase,
        }
        if offer:
            result["deal_score"] = self.bidding.score_allocation(offer)

        return result

    # ── Deal handling ──────────────────────────────────────────────────

    def should_accept_deal(
        self,
        their_offer_for_me: Dict[str, int],
    ) -> bool:
        """Evaluate an opponent's Submit-Deal."""
        return self.bidding.evaluate_offer(their_offer_for_me, self.turn)

    def respond_to_deal(
        self,
        their_offer_for_me: Dict[str, int],
    ) -> Dict[str, Any]:
        """Full deal response: decide + generate natural text.

        Returns:
            {"accept": bool, "text": str, "action": "Accept-Deal"|"Reject-Deal",
             "my_score": int, "target_score": int}
        """
        accept = self.should_accept_deal(their_offer_for_me)
        prompt = build_deal_response_prompt(
            accept=accept,
            history=self.history,
            proposed_offer=their_offer_for_me,
        )
        text = self.llm_client.generate(prompt)

        my_score = self.bidding.score_allocation(their_offer_for_me)
        target = self.bidding.get_target(self.turn)
        target_score = self.bidding.score_allocation(target)

        action = "Accept-Deal" if accept else "Reject-Deal"
        self.history.append({"role": "agent", "text": text})

        return {
            "accept": accept,
            "text": text,
            "action": action,
            "my_score": my_score,
            "target_score": target_score,
        }

    def propose_final_deal(self) -> Dict[str, Any]:
        """Generate a Submit-Deal with natural-language confirmation.

        Returns:
            {"task_data": {"issue2youget": ..., "issue2theyget": ...},
             "text": str, "deal": {"me": ..., "them": ...}, "score": int}
        """
        deal = self.bidding.generate_offer(self.turn, self.opponent_model)

        prompt = build_deal_proposal_prompt(
            agent_priorities=self.priorities,
            history=self.history,
            proposed_offer=deal["me"],
        )
        text = self.llm_client.generate(prompt)

        task_data = self.bidding.format_for_submit_deal(deal)
        score = self.bidding.score_allocation(deal["me"])

        self.history.append({"role": "agent", "text": text})

        return {
            "task_data": task_data,
            "text": text,
            "deal": deal,
            "score": score,
        }

    def should_walk_away(self) -> bool:
        """Decide whether to walk away instead of accepting a bad deal.

        Walk away if the best offer we can get at this point scores
        below the walk-away threshold.
        """
        target = self.bidding.get_target(self.turn)
        target_score = self.bidding.score_allocation(target)
        return target_score <= WALK_AWAY_POINTS

    # ── Internal helpers ───────────────────────────────────────────────

    def _sync_profile(
        self,
        priorities: Dict[str, str],
        reasons: Dict[str, str],
        personality: Optional[Dict[str, float]] = None,
        svo: Optional[str] = None,
    ) -> None:
        """Synchronize dialogue-specific profile info for the current speaker."""
        if priorities != self.priorities:
            self.priorities = dict(priorities)
            self.bidding = BiddingStrategy(
                self.priorities,
                beta=self.beta,
                max_turns=self.max_turns,
            )
            self.strategy_selector = StrategySelector(my_priorities=self.priorities)

        self.reasons = dict(reasons)
        self.personality = personality
        self.svo = svo

    def _process_opponent_turn(
        self,
        text: str,
        context_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Classify strategies and update opponent model."""
        context = self._format_history_for_context(context_history)
        strategies = classify_strategies(text, context, self.llm_client)
        items = extract_mentioned_items(text)
        self.opponent_model.update(text, strategies, items)

        return {
            "strategies_detected": strategies,
            "items_mentioned": items,
            "opponent_confidence": self.opponent_model.confidence,
            "opponent_predicted_priorities": self.opponent_model.get_predicted_priorities(),
        }

    def _format_history_for_context(
        self,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Format recent history as context for strategy classification."""
        source = self.history if history is None else history
        recent = source[-5:]
        lines: List[str] = []
        for h in recent:
            speaker = "You" if h["role"] == "agent" else "Opponent"
            lines.append(f"{speaker}: {h['text']}")
        return "\n".join(lines) if lines else "(conversation start)"

    def _assess_deal_quality(self) -> str:
        """Assess current deal quality relative to Boulware target."""
        target = self.bidding.get_target(self.turn)
        target_score = self.bidding.score_allocation(target)
        aspiration_score = self.bidding.score_allocation(self.bidding.aspiration)
        reservation_score = self.bidding.score_allocation(self.bidding.reservation)

        mid = (aspiration_score + reservation_score) / 2
        if target_score >= mid:
            return "above_threshold"
        if target_score >= reservation_score:
            return "acceptable"
        return "below_threshold"

    def summary(self) -> Dict[str, Any]:
        """Full agent state snapshot for debugging."""
        return {
            "turn": self.turn,
            "phase": self._last_phase,
            "last_strategies": self._last_strategies,
            "opponent_model": self.opponent_model.summary(),
            "bidding": self.bidding.summary(self.turn),
            "history_length": len(self.history),
        }


# ── CLI demo (dry run without LLM) ────────────────────────────────────────

class _DummyLLM:
    """Placeholder LLM for testing the agent pipeline without a real model."""

    def generate(self, prompt: str) -> str:
        if "ACCEPT" in prompt:
            return "Sounds like a great deal! Let's go with that! 🎉"
        if "FINAL DEAL" in prompt:
            return "How about I take the food and water, you take the firewood? Deal? 🤝"
        return (
            "That makes sense! I really need the food for my group — we've got "
            "a lot of mouths to feed. I could let you have the firewood though. "
            "How about I get 3 food and 2 water, and you take 3 firewood and 1 water?"
        )


if __name__ == "__main__":
    priorities = {"High": "Food", "Medium": "Water", "Low": "Firewood"}
    reasons = {
        "High": "We have a large group and need to feed everyone properly.",
        "Medium": "Staying hydrated is important since we'll be hiking a lot.",
        "Low": "We brought some matches and lighter fluid already.",
    }

    agent = CaSiNoAgent(
        priorities=priorities,
        reasons=reasons,
        llm_client=_DummyLLM(),
        beta=3.0,
        max_turns=10,
    )

    print("CaSiNo Agent — Dry Run")
    print("=" * 60)

    opponent_msgs = [
        "Hey! Excited for the camping trip? 🏕️",
        "I really need firewood — my kids get cold at night.",
        "Sure, I could give up some food. How about 2 food for 2 firewood?",
    ]

    for msg in opponent_msgs:
        print(f"\nOpponent: {msg}")
        info = agent.receive_opponent_message(msg)
        print(f"  [strategies: {info['strategies_detected']}]")
        print(f"  [opponent confidence: {info['opponent_confidence']}]")

        response = agent.generate_response()
        print(f"Agent: {response['text']}")
        print(f"  [phase: {response['phase']}, strategies: {response['strategies_used']}]")
        if response.get("offer"):
            print(f"  [offer: {response['offer']}, score: {response.get('deal_score')}]")

    print(f"\n{'─' * 60}")
    print("Final deal proposal:")
    deal = agent.propose_final_deal()
    print(f"  Text: {deal['text']}")
    print(f"  task_data: {deal['task_data']}")
    print(f"  Score: {deal['score']}")

    print(f"\n{'─' * 60}")
    print("Agent state:")
    import json
    state = agent.summary()
    state.pop("opponent_model")  # compact output
    print(json.dumps(state, indent=2))
