"""Hybrid Bayesian opponent model for CaSiNo negotiation.

The package exposes:
    - ``HybridAgent``: full LLM + Bayesian opponent-modeling agent.
    - ``ITEMS`` / ``HYPOTHESES``: the 3 items and 6 priority permutations.
    - ``big_five_prior``: stub mapping a Big Five vector to a Dirichlet prior.

Usage:
    from opponent_model import HybridAgent

    agent = HybridAgent(my_priorities={"High": "Food",
                                       "Medium": "Water",
                                       "Low": "Firewood"},
                        llm_client=client)
    agent.observe("We really need water for the hike.")
    text, offer = agent.speak()
"""

from opponent_model.cache import CachedLLM, DiskCache
from opponent_model.hypotheses import HYPOTHESES, ITEMS
from opponent_model.hybrid_agent import HybridAgent, MalformedLikelihoodError
from opponent_model.turn_level_metrics import (
    CASINO_STRATEGIES,
    TurnLevelAgent,
    TurnRecord,
    aggregate_turn_metrics,
    format_turn_level_summary,
    turn_level_eval,
)

__all__ = [
    "HybridAgent",
    "MalformedLikelihoodError",
    "CachedLLM",
    "DiskCache",
    "ITEMS",
    "HYPOTHESES",
    "TurnLevelAgent",
    "TurnRecord",
    "CASINO_STRATEGIES",
    "turn_level_eval",
    "aggregate_turn_metrics",
    "format_turn_level_summary",
]
