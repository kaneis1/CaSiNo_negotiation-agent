#!/usr/bin/env python3
"""Evaluation harness for CaSiNo negotiation agents.

Replays recorded dialogues with one participant replaced by an agent,
then scores the agent's negotiation performance.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from prompt_engineer.preprocessing.scoring import (
    LIKENESS_MAP,
    PRIORITY_POINTS,
    SATISFACTION_MAP,
    WALK_AWAY_POINTS,
)

# ── Deal actions that are not regular conversation turns ────────────────────

DEAL_ACTIONS = {"Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away"}

# ── Agent interface ────────────────────────────────────────────────────────


class NegotiationAgent(ABC):
    """Base class that any agent must implement to be evaluated."""

    @abstractmethod
    def generate(
        self,
        history: List[Dict[str, str]],
        priorities: Dict[str, str],
        reasons: Dict[str, str],
        personality: Optional[Dict[str, float]] = None,
        svo: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate the next conversational response.

        Args:
            history: List of {"role": "agent"|"opponent", "text": str}.
            priorities: value2issue mapping, e.g. {"High": "Food", ...}.
            reasons: value2reason mapping.
            personality: Big-Five scores (optional).
            svo: Social Value Orientation — "prosocial" or "proself" (optional).

        Returns:
            {"text": str, "strategies": List[str] (optional), ...}
        """

    @abstractmethod
    def propose_deal(
        self,
        history: List[Dict[str, str]],
        priorities: Dict[str, str],
    ) -> Optional[Dict[str, int]]:
        """Propose a final deal after conversation ends.

        Returns:
            {"Food": int, "Water": int, "Firewood": int} — quantities the
            agent claims for itself (each 0-3, totals across both sides = 3).
            Return None to walk away.
        """

    def get_opponent_model_accuracy(
        self,
        true_opponent_priorities: Dict[str, str],
    ) -> Optional[float]:
        """Return accuracy of agent's internal model of opponent priorities.

        Override if the agent tracks an opponent model. Default returns None.
        """
        return None

    def reset(self) -> None:
        """Reset agent state between dialogues. Override if needed."""


# ── Evaluator ──────────────────────────────────────────────────────────────


class NegotiationEvaluator:
    """Replay CaSiNo dialogues with an agent replacing one participant."""

    def __init__(self, dialogues: List[Dict[str, Any]]):
        self.dialogues = dialogues

    # ── Single-dialogue evaluation ─────────────────────────────────────

    def evaluate_agent_on_dialogue(
        self,
        agent: NegotiationAgent,
        dialogue: Dict[str, Any],
        agent_role: str = "mturk_agent_1",
    ) -> Dict[str, Any]:
        """Run *agent* as one participant; replay opponent turns from data.

        Args:
            agent: The agent to evaluate.
            dialogue: A single CaSiNo dialogue dict.
            agent_role: "mturk_agent_1" or "mturk_agent_2".

        Returns:
            Dict with points_scored, deal_reached, strategy_distribution,
            num_turns, opponent_model_accuracy.
        """
        opponent_role = (
            "mturk_agent_2" if agent_role == "mturk_agent_1" else "mturk_agent_1"
        )

        pinfo = dialogue["participant_info"]
        agent_info = pinfo[agent_role]

        priorities = agent_info["value2issue"]
        reasons = agent_info["value2reason"]
        personality = agent_info.get("personality", {}).get("big-five")
        svo = agent_info.get("personality", {}).get("svo")

        agent.reset()

        history: List[Dict[str, str]] = []
        agent_utterances: List[Dict[str, Any]] = []

        for turn in dialogue["chat_logs"]:
            if turn["text"] in DEAL_ACTIONS:
                continue

            if turn["id"] == opponent_role:
                history.append({"role": "opponent", "text": turn["text"]})
            else:
                response = agent.generate(
                    history=history,
                    priorities=priorities,
                    reasons=reasons,
                    personality=personality,
                    svo=svo,
                )
                history.append({"role": "agent", "text": response["text"]})
                agent_utterances.append(response)

        final_deal = agent.propose_deal(history, priorities)

        opponent_priorities = pinfo[opponent_role]["value2issue"]

        return {
            "dialogue_id": dialogue["dialogue_id"],
            "agent_role": agent_role,
            "points_scored": self._calc_points(final_deal, priorities),
            "deal_reached": final_deal is not None,
            "strategy_distribution": self._count_strategies(agent_utterances),
            "num_turns": len(agent_utterances),
            "opponent_model_accuracy": agent.get_opponent_model_accuracy(
                opponent_priorities
            ),
        }

    # ── Batch evaluation ───────────────────────────────────────────────

    def evaluate_all(
        self,
        agent: NegotiationAgent,
        agent_role: str = "mturk_agent_1",
        max_dialogues: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluate agent across all dialogues and return aggregate metrics."""

        subset = self.dialogues[:max_dialogues] if max_dialogues else self.dialogues
        results = []

        for dialogue in subset:
            r = self.evaluate_agent_on_dialogue(agent, dialogue, agent_role)
            results.append(r)

        points = [r["points_scored"] for r in results]
        deals = [r["deal_reached"] for r in results]
        turns = [r["num_turns"] for r in results]
        opp_acc = [
            r["opponent_model_accuracy"]
            for r in results
            if r["opponent_model_accuracy"] is not None
        ]

        agg_strategies: Dict[str, int] = defaultdict(int)
        total_utt = 0
        for r in results:
            for s, cnt in r["strategy_distribution"].items():
                agg_strategies[s] += cnt
                total_utt += cnt

        return {
            "num_dialogues": len(results),
            "avg_points": float(np.mean(points)),
            "std_points": float(np.std(points)),
            "deal_rate": float(np.mean(deals)),
            "avg_turns": float(np.mean(turns)),
            "avg_opponent_model_accuracy": float(np.mean(opp_acc)) if opp_acc else None,
            "strategy_distribution": (
                {k: v / total_utt for k, v in agg_strategies.items()}
                if total_utt > 0
                else {}
            ),
            "per_dialogue": results,
        }

    # ── Human baseline ─────────────────────────────────────────────────

    def human_baseline(self, agent_role: str = "mturk_agent_1") -> Dict[str, Any]:
        """Compute metrics for the actual human participant as a baseline."""

        points = []
        satisfactions = []
        likenesses = []

        for d in self.dialogues:
            info = d["participant_info"][agent_role]
            points.append(info["outcomes"]["points_scored"])
            satisfactions.append(SATISFACTION_MAP[info["outcomes"]["satisfaction"]])
            likenesses.append(LIKENESS_MAP[info["outcomes"]["opponent_likeness"]])

        return {
            "num_dialogues": len(self.dialogues),
            "avg_points": float(np.mean(points)),
            "std_points": float(np.std(points)),
            "avg_satisfaction": float(np.mean(satisfactions)),
            "avg_likeness": float(np.mean(likenesses)),
        }

    # ── Internal helpers ───────────────────────────────────────────────

    @staticmethod
    def _calc_points(
        deal: Optional[Dict[str, int]],
        priorities: Dict[str, str],
    ) -> int:
        """Calculate points from an agent's proposed deal allocation."""
        if deal is None:
            return WALK_AWAY_POINTS
        issue2priority = {v: k for k, v in priorities.items()}
        total = 0
        for item, count in deal.items():
            priority = issue2priority.get(item, "Low")
            total += count * PRIORITY_POINTS[priority]
        return total

    @staticmethod
    def _count_strategies(
        agent_utterances: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """Count strategy labels across agent utterances."""
        counts: Dict[str, int] = defaultdict(int)
        for utt in agent_utterances:
            strategies = utt.get("strategies")
            if strategies is None:
                strategies = utt.get("strategies_used", [])
            for s in strategies:
                counts[s.strip()] += 1
        return dict(counts)


# ── CLI demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data_path = (
        Path(__file__).resolve().parents[2] / "CaSiNo" / "data" / "casino.json"
    )
    with data_path.open() as f:
        data = json.load(f)

    evaluator = NegotiationEvaluator(data)
    baseline = evaluator.human_baseline("mturk_agent_1")

    print("Human baseline (mturk_agent_1):")
    print(f"  Dialogues:        {baseline['num_dialogues']}")
    print(f"  Avg points:       {baseline['avg_points']:.2f} ± {baseline['std_points']:.2f}")
    print(f"  Avg satisfaction: {baseline['avg_satisfaction']:.2f}")
    print(f"  Avg likeness:     {baseline['avg_likeness']:.2f}")
