"""Live Structured-CoT agent for Protocol 3 (gold-history) evaluation.

Unlike :class:`structured_cot.replay_turn_agent.StructuredCoTReplayAgent`, which
replays a finished Protocol-1 *self-play* trace, this adapter calls the 70B
(or any) LLM at **every** gold ``chat_logs`` turn for the perspective role,
with dialogue history built the same way as :func:`structured_cot.run_protocol1.replay_protocol1`.

That yields **matched support** with :func:`opponent_model.turn_level_metrics.turn_level_eval`
(n≈87 accept-decision turns and n≈86 Submit-Deal turns for ``mturk_agent_1`` on
the 150-dialogue held-out split), eliminating the early-termination bias of
Protocol 1.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from structured_cot.agent import StructuredCoTAgent
from structured_cot.run_protocol1 import (
    _coerce_counts,
    _get_arguments,
    _get_priorities,
    counter_offer_is_legal,
    render_agent_counter_offer,
    render_opponent_deal_action,
)
from opponent_model.turn_agents import KeywordStrategyClassifier

logger = logging.getLogger("structured_cot.live_turn_agent")


def _dialogue_id_from_history(history: Sequence[Mapping[str, Any]]) -> Any:
    for turn in history:
        if "dialogue_id" in turn:
            return turn["dialogue_id"]
    return None


def chat_logs_to_cot_pairs(
    chat_log_turns: Sequence[Mapping[str, Any]],
    *,
    my_role: str,
    opp_role: str,
) -> List[Tuple[str, str]]:
    """Mirror ``run_protocol1`` history strings from raw ``chat_logs`` dicts."""
    pairs: List[Tuple[str, str]] = []
    for turn in chat_log_turns:
        speaker = turn.get("id")
        text = (turn.get("text") or "").strip()
        if speaker == opp_role:
            pairs.append(("opp", render_opponent_deal_action(turn)))
        elif speaker == my_role:
            if text.startswith("Submit-Deal"):
                td = turn.get("task_data") or {}
                you = _coerce_counts(td.get("issue2youget", {}))
                pairs.append(("me", render_agent_counter_offer(you)))
            elif text.startswith("Accept-Deal"):
                pairs.append((
                    "me",
                    "Accept-Deal — I accept the proposal on the table.",
                ))
            elif text.startswith("Reject-Deal"):
                pairs.append(("me", "Reject-Deal — I reject the last proposal."))
            elif text.startswith("Walk-Away"):
                pairs.append((
                    "me",
                    "Walk-Away — I am walking away from the negotiation.",
                ))
            else:
                pairs.append(("me", text))
    return pairs


def pending_counts_for_cot(
    pending: Optional[Mapping[str, Any]],
    my_role: str,
) -> Optional[Dict[str, int]]:
    """Same semantics as Protocol 1's ``pending`` dict (agent-receive counts)."""
    if pending is None:
        return None
    td = pending.get("task_data") or {}
    proposer = pending.get("proposer")
    if proposer == my_role:
        return _coerce_counts(td.get("issue2youget", {}))
    return _coerce_counts(td.get("issue2theyget", {}))


class StructuredCoTLiveTurnAgent:
    """``TurnLevelAgent`` that runs :class:`StructuredCoTAgent` on gold history."""

    def __init__(
        self,
        llm_client: Any,
        *,
        max_tokens: int = 800,
        temperature: float = 0.3,
        parse_log_path: Optional[Path] = None,
        strategy_classifier: Optional[Any] = None,
    ) -> None:
        self._llm = llm_client
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
        self._parse_log_path = Path(parse_log_path) if parse_log_path else None
        self.strategy_classifier = (
            strategy_classifier
            if strategy_classifier is not None
            else KeywordStrategyClassifier()
        )
        self._cot: Optional[StructuredCoTAgent] = None
        self._last_dialogue_id: Any = object()  # sentinel

    def _ensure_cot(
        self,
        my_priorities: Mapping[str, str],
        my_reasons: Mapping[str, str],
    ) -> StructuredCoTAgent:
        pri = _get_priorities({"value2issue": dict(my_priorities)})
        args = _get_arguments({"value2reason": dict(my_reasons)})
        return StructuredCoTAgent(
            pri,
            args,
            self._llm,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            parse_log_path=self._parse_log_path,
        )

    def predict_turn(
        self,
        *,
        history: List[Mapping[str, Any]],
        my_role: str,
        opp_role: str,
        my_priorities: Mapping[str, str],
        my_reasons: Mapping[str, str],
        pending_offer: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        did = _dialogue_id_from_history(history)
        if did is not None and did != self._last_dialogue_id:
            self._last_dialogue_id = did
            self._cot = self._ensure_cot(my_priorities, my_reasons)
        elif self._cot is None:
            self._cot = self._ensure_cot(my_priorities, my_reasons)
            self._last_dialogue_id = did

        prior_agent_turns = sum(1 for t in history if t.get("id") == my_role)
        if self._cot is None:
            self._cot = self._ensure_cot(my_priorities, my_reasons)
        self._cot.reset()
        self._cot.turn_count = prior_agent_turns

        dialogue_pairs = chat_logs_to_cot_pairs(
            history, my_role=my_role, opp_role=opp_role,
        )
        pending_simple = pending_counts_for_cot(pending_offer, my_role)

        result = self._cot.act(dialogue_pairs, pending_offer=pending_simple)
        parsed = result.parsed
        decision = parsed.get("decision") or {}
        action = decision.get("action")
        counter = decision.get("counter_offer")

        if action == "accept":
            accept: Optional[bool] = True
            bid = None
        elif action == "walkaway":
            accept = False
            bid = None
        elif action == "reject":
            accept = False
            if isinstance(counter, Mapping) and counter_offer_is_legal(counter):
                bid = _coerce_counts(counter)
            else:
                bid = None
        else:
            accept, bid = None, None

        utterance = (parsed.get("utterance") or "").strip()
        strategy: Optional[List[str]] = None
        if utterance:
            try:
                tags = list(self.strategy_classifier(utterance, list(history)))
                strategy = tags or None
            except Exception:
                logger.exception("strategy classifier failed on live CoT utterance.")
                strategy = None

        return {
            "accept": accept,
            "bid": bid,
            "strategy": strategy,
            "posterior": None,
        }


__all__ = [
    "StructuredCoTLiveTurnAgent",
    "chat_logs_to_cot_pairs",
    "pending_counts_for_cot",
]
