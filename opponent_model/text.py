"""Dry-run demo for the Hybrid Bayesian opponent model.

Runs ``HybridAgent`` against a tiny scripted opponent using a *dummy* LLM
client that returns canned JSON. Useful for quickly sanity-checking the
posterior update + generation plumbing without loading a real model.

Usage:
    python -m opponent_model.text
"""

from __future__ import annotations

import json
from typing import Any, Dict

from opponent_model.hybrid_agent import HybridAgent
from opponent_model.hypotheses import HYPOTHESES, hypothesis_label


class _DummyLLM:
    """Deterministic LLM stand-in: pattern-matches the prompt to return JSON.

    The real client (LlamaClient in prompt_engineer.llm.client) returns a
    raw string; HybridAgent calls _safe_json_loads on it. We mimic that.
    """

    def generate(self, prompt: str) -> str:
        if "evidence model" in prompt.lower():
            return self._fake_likelihood(prompt)
        if "casino negotiation" in prompt.lower():
            return self._fake_generation(prompt)
        return "{}"

    def _fake_likelihood(self, prompt: str) -> str:
        utt_lower = prompt.lower()
        scores: Dict[str, float] = {hypothesis_label(i): 50.0 for i in range(6)}
        rationale = "no priority signal"
        if "really need water" in utt_lower:
            rationale = "Utterance explicitly states water need."
            for i, h in enumerate(HYPOTHESES):
                if h[0] == "Water":
                    scores[hypothesis_label(i)] = 80.0
                elif h[1] == "Water":
                    scores[hypothesis_label(i)] = 45.0
                else:
                    scores[hypothesis_label(i)] = 20.0
                if "give you more food" in utt_lower and h[2] == "Food":
                    scores[hypothesis_label(i)] = min(85.0, scores[hypothesis_label(i)] + 5.0)
        return json.dumps({
            "evidence_scores": scores,
            "short_rationale": rationale,
        })

    def _fake_generation(self, prompt: str) -> str:
        return json.dumps({
            "utterance": (
                "Sounds like water matters most to you. I'd happily take "
                "more food in exchange for letting you have most of the water."
            ),
            "offer": {"Food": 3, "Water": 1, "Firewood": 2},
        })


def main() -> None:
    agent = HybridAgent(
        my_priorities={"High": "Food", "Medium": "Firewood", "Low": "Water"},
        llm_client=_DummyLLM(),
    )

    print("Initial state:")
    _print_state(agent.state())

    opp_msgs = [
        "Hi! Excited for the trip.",
        "We really need water because we are hiking tomorrow, "
        "but I can give you more food.",
    ]

    for msg in opp_msgs:
        print(f"\nOpponent: {msg}")
        info = agent.observe(msg)
        print(f"  rationale: {info['rationale']}")
        _print_state(agent.state())

    print("\nAgent speaks:")
    text, offer = agent.speak()
    print(f"  utterance: {text}")
    print(f"  offer:     {offer}")


def _print_state(state: Dict[str, Any]) -> None:
    print(f"  certainty: {state['certainty']:.3f}")
    print(f"  top marginals: " + ", ".join(
        f"{k}={v:.2f}" for k, v in state["top_marginals"].items()
    ))
    print(f"  predicted: {state['predicted_priorities']}")


if __name__ == "__main__":
    main()
