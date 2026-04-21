"""Adapters that present existing CaSiNo agents as ``TurnLevelAgent``.

The new ``turn_level_eval`` harness expects a single
``predict_turn(...)`` entry point that returns a dict with optional
``accept`` / ``bid`` / ``strategy`` / ``posterior`` keys. None of the
existing agents have that shape natively:

    * :class:`opponent_model.HybridAgent` exposes a 6-class posterior and
      can ``speak()`` (utterance + offer) but doesn't classify strategy
      or decide accept/reject directly.

    * :class:`sft_8b.predict.SftModelFn` predicts an ordering + a
      satisfaction label only.

Each adapter below is intentionally small and clearly marked: it fills
in what its underlying agent *does* know, and leaves the rest as
``None`` so the metric simply abstains for that turn rather than
collapsing the run.

For strategy classification we accept an optional pluggable
``StrategyClassifier`` callable; the bundled ``KeywordStrategyClassifier``
is a deterministic, dependency-free fallback useful for smoke tests.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from opponent_model.hybrid_agent import HybridAgent
from opponent_model.hypotheses import HYPOTHESES, ITEMS
from opponent_model.turn_level_metrics import (
    CASINO_STRATEGIES,
    coerce_bid_vector,
)

logger = logging.getLogger("opponent_model.turn_agents")


# ── Strategy classifier protocol + simple impls ────────────────────────────


StrategyClassifier = Callable[[str, List[Mapping[str, Any]]], Sequence[str]]


def _keyword_classifier(text: str, history: List[Mapping[str, Any]]) -> List[str]:
    """Cheap keyword-based CaSiNo strategy tagger.

    NOT meant as a final classifier — its only job is to make the
    pipeline runnable end-to-end with a deterministic, dependency-free
    fallback. Replace with an LLM- or BERT-based tagger before publishing
    real numbers.
    """
    t = (text or "").lower()
    tags: List[str] = []
    if any(w in t for w in (
        "hi ", "hello", "how are you", "have a good", "lol", "haha",
        "thank you", "thanks", "nice to",
    )):
        tags.append("small-talk")
    if any(w in t for w in (
        "i need", "i want", "i'd like", "i would like", "we need", "i'm",
        "i am ", "my kids", "my family", "my dog", "my grandma",
    )):
        tags.append("self-need")
    if any(w in t for w in (
        "you need", "you want", "your kids", "your family", "your dog",
        "for you", "what do you", "your situation",
    )):
        tags.append("other-need")
    if any(w in t for w in (
        "don't need", "don't want", "i'm fine", "i'm good with",
        "no need for", "make do without",
    )):
        tags.append("no-need")
    if any(w in t for w in (
        "what are you", "what do you", "your priorities", "most interested",
        "least interested", "what's your", "preference",
    )):
        tags.append("elicit-pref")
    if any(w in t for w in (
        "let's", "shall we", "we both", "both of us", "work together",
        "compromise", "split", "fair",
    )):
        tags.append("promote-coordination")
    if "fair" in t or "even" in t or "equally" in t:
        tags.append("vouch-fair")
    if any(w in t for w in (
        "i understand", "i see", "i hear you", "sorry to hear", "feel for",
    )):
        tags.append("showing-empathy")
    if not tags:
        tags.append("non-strategic")
    return tags


class KeywordStrategyClassifier:
    """Object-form of :func:`_keyword_classifier` for easy injection."""

    def __init__(self, label_set: Sequence[str] = CASINO_STRATEGIES) -> None:
        self.label_set = set(label_set)

    def __call__(
        self, text: str, history: List[Mapping[str, Any]],
    ) -> List[str]:
        tags = _keyword_classifier(text, history)
        return [t for t in tags if t in self.label_set] or ["non-strategic"]


# ── HybridAgent adapter ────────────────────────────────────────────────────


def _replay_history_into_agent(
    agent: HybridAgent,
    history: Sequence[Mapping[str, Any]],
    *,
    my_role: str,
    opp_role: str,
) -> None:
    """Feed a list of chat_logs turns into a fresh HybridAgent.

    Opponent utterances trigger the LLM-likelihood + Bayes update; agent
    utterances are pushed onto the internal history so prompts have
    proper context. Action turns (Submit-Deal, Accept-Deal, etc.) do not
    contribute to the posterior but are recorded as agent context.
    """
    for turn in history:
        text = (turn.get("text") or "").strip()
        if not text:
            continue
        role = turn.get("id")
        if text in {"Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away"}:
            tag = "me" if role == my_role else "opp"
            agent.history.append({"role": tag, "text": text})
            continue
        if role == opp_role:
            try:
                agent.observe(text)
            except Exception:
                logger.warning(
                    "HybridAgent.observe failed on text %r; skipping turn.",
                    text[:80],
                )
                agent.history.append({"role": "opp", "text": text})
        else:
            agent.history.append({"role": "me", "text": text})


def _accept_decision(
    agent: HybridAgent,
    pending_offer: Mapping[str, Any],
    my_priorities: Mapping[str, str],
) -> bool:
    """Heuristic accept/reject decision used by the HybridAgent adapter.

    Accept iff the deal would give the agent at least 60% of its Pareto
    maximum (with the standard 5/4/3 point map). This is a stand-in
    until the LLM is asked to decide explicitly; the harness lets you
    swap it via ``HybridTurnAgent(accept_fn=...)``.
    """
    points_map = {"High": 5, "Medium": 4, "Low": 3}
    issue2priority = {v: k for k, v in dict(my_priorities).items()}
    td = pending_offer["task_data"]
    proposer = pending_offer["proposer"]

    if proposer == pending_offer.get("perspective"):
        my_share = td.get("issue2youget", {})
    else:
        my_share = td.get("issue2theyget", {})

    try:
        pts = sum(
            int(my_share.get(item, 0)) * points_map[issue2priority[item]]
            for item in ITEMS
        )
    except Exception:
        return False

    max_pts = sum(3 * points_map[p] for p in ("High", "Medium", "Low"))
    return (pts / max_pts) >= 0.60


class HybridTurnAgent:
    """Wrap :class:`HybridAgent` for :func:`turn_level_eval`.

    The hybrid agent produces:
        * ``posterior``: the 6-vector exposed by ``HybridAgent.posterior()``
        * ``bid``      : derived from the agent's ``speak()`` offer
                         (None when the LLM declines to propose)
        * ``accept``   : a simple Pareto-share heuristic by default
                         (override via ``accept_fn``)
        * ``strategy`` : applied to the agent's *own* generated utterance
                         via the configured ``StrategyClassifier``

    Construction is cheap; the heavy work (LLM calls, Bayes updates) is
    deferred to ``predict_turn``. We rebuild the underlying HybridAgent
    from scratch on every call so the harness stays stateless and
    re-orderings of dialogues don't leak posterior state across turns.
    """

    def __init__(
        self,
        llm_client: Any,
        *,
        strategy_classifier: Optional[StrategyClassifier] = None,
        accept_fn: Optional[Callable[..., bool]] = None,
        propose_bid: bool = True,
        likelihood_temperature: float = 25.0,
        likelihood_clip: tuple = (-3.0, 3.0),
        strict_likelihood: bool = False,
    ) -> None:
        self.llm_client = llm_client
        self.strategy_classifier = strategy_classifier
        self.accept_fn = accept_fn or _accept_decision
        self.propose_bid = propose_bid
        self.likelihood_temperature = likelihood_temperature
        self.likelihood_clip = likelihood_clip
        self.strict_likelihood = strict_likelihood

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
        agent = HybridAgent(
            my_priorities=my_priorities,
            llm_client=self.llm_client,
            likelihood_temperature=self.likelihood_temperature,
            likelihood_clip=self.likelihood_clip,
            strict_likelihood=self.strict_likelihood,
        )
        _replay_history_into_agent(
            agent, history, my_role=my_role, opp_role=opp_role,
        )

        posterior = agent.posterior().tolist()

        accept: Optional[bool] = None
        if pending_offer is not None and pending_offer.get("to_perspective"):
            try:
                accept = bool(self.accept_fn(
                    agent,
                    {**dict(pending_offer), "perspective": my_role},
                    my_priorities,
                ))
            except Exception:
                logger.exception("accept_fn failed; reporting None.")
                accept = None

        bid: Optional[Dict[str, int]] = None
        utterance: str = ""
        if self.propose_bid and (accept is None or not accept):
            try:
                utterance, offer = agent.speak()
                if isinstance(offer, Mapping):
                    bid = {it: int(offer.get(it, 0)) for it in ITEMS}
            except Exception:
                logger.exception("HybridAgent.speak failed; bid=None.")
                utterance = ""
                bid = None

        strategy: Optional[List[str]] = None
        if self.strategy_classifier is not None and utterance:
            try:
                tags = list(self.strategy_classifier(utterance, list(history)))
                strategy = tags or None
            except Exception:
                logger.exception("strategy classifier failed; strategy=None.")
                strategy = None

        return {
            "accept": accept,
            "bid": bid,
            "strategy": strategy,
            "posterior": posterior,
        }


# ── SFT 8B adapter ─────────────────────────────────────────────────────────


class SftTurnAgent:
    """Wrap :class:`sft_8b.predict.SftModelFn` for :func:`turn_level_eval`.

    The SFT model only predicts a priority ordering (and a satisfaction
    label, which the turn-level harness ignores). We turn the ordering
    into a one-hot posterior so Brier is well-defined; ``accept``,
    ``bid`` and ``strategy`` are reported as ``None`` (the metrics will
    abstain rather than score the model on signals it doesn't produce).

    Pass an explicit ``strategy_classifier`` if you want the SFT adapter
    to also produce strategy labels — the harness has no way of
    extracting them from the SFT model itself.
    """

    def __init__(
        self,
        sft_model: Any,
        *,
        strategy_classifier: Optional[StrategyClassifier] = None,
    ) -> None:
        self.sft_model = sft_model
        self.strategy_classifier = strategy_classifier

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
        try:
            ordering = list(self.sft_model(
                list(history), dict(my_priorities), opp_role, my_role,
                dict(my_reasons),
            ))
        except Exception:
            logger.exception("SftModelFn call failed; posterior=None.")
            ordering = []

        posterior: Optional[List[float]] = None
        if len(ordering) == 3:
            target = tuple(ordering)
            arr = np.zeros(len(HYPOTHESES), dtype=float)
            for i, h in enumerate(HYPOTHESES):
                if tuple(h) == target:
                    arr[i] = 1.0
                    break
            if arr.sum() > 0:
                posterior = arr.tolist()

        strategy: Optional[List[str]] = None
        if self.strategy_classifier is not None:
            last_utterance = ""
            for turn in reversed(history):
                text = (turn.get("text") or "").strip()
                if text and text not in (
                    "Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away",
                ):
                    last_utterance = text
                    break
            if last_utterance:
                try:
                    tags = list(self.strategy_classifier(
                        last_utterance, list(history),
                    ))
                    strategy = tags or None
                except Exception:
                    logger.exception("strategy classifier failed.")

        return {
            "accept": None,
            "bid": None,
            "strategy": strategy,
            "posterior": posterior,
        }


# ── Trivial test agent ─────────────────────────────────────────────────────


class UniformTurnAgent:
    """Uniform-posterior, never-accepts, no-bid baseline.

    Useful as a dependency-free smoke-test sanity check (gives us
    well-defined Brier and lets us verify the harness wiring without
    needing a GPU).
    """

    def __init__(
        self,
        *,
        strategy_classifier: Optional[StrategyClassifier] = None,
    ) -> None:
        self.strategy_classifier = strategy_classifier or KeywordStrategyClassifier()

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
        posterior = [1.0 / len(HYPOTHESES)] * len(HYPOTHESES)

        accept: Optional[bool] = None
        if pending_offer is not None and pending_offer.get("to_perspective"):
            accept = False

        bid: Dict[str, int] = {it: 1 for it in ITEMS}

        last_utterance = ""
        for turn in reversed(history):
            text = (turn.get("text") or "").strip()
            if text and text not in (
                "Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away",
            ):
                last_utterance = text
                break
        strategy = None
        if last_utterance:
            try:
                strategy = list(self.strategy_classifier(
                    last_utterance, list(history),
                ))
            except Exception:
                strategy = None

        return {
            "accept": accept,
            "bid": bid,
            "strategy": strategy,
            "posterior": posterior,
        }


__all__ = [
    "StrategyClassifier",
    "KeywordStrategyClassifier",
    "HybridTurnAgent",
    "SftTurnAgent",
    "UniformTurnAgent",
]
