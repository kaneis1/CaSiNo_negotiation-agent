"""Replay adapter: scores the finished Protocol 1 run through turn_level_eval.

The Structured-CoT agent is expensive (70B, ~10 h for 150 dialogues) and
we don't want to pay the same wallclock tax twice when we're comparing
against a new agent. This module takes a completed Protocol-1 run
directory (containing ``turns.jsonl``) and exposes its per-turn
decisions as a stateless :class:`TurnLevelAgent` — so the baseline and
any new contender are scored by *the exact same harness* on *the exact
same turns*, with zero chance of a definitional gap in accept F1, bid
cosine or strategy macro-F1.

Scope.
    The structured-CoT agent doesn't emit a posterior; the replay adapter
    therefore always returns ``posterior=None`` and the harness abstains
    on Brier for this agent — which is the headline "automatic win" for
    any Bayesian-style competitor.

Alignment.
    Protocol 1's ``turns.jsonl`` is keyed by
    ``(dialogue_id, turn_index, agent_role)``, where ``turn_index`` is
    the *chat_logs* index (verified — see commit notes). ``turn_level_eval``
    iterates the same chat_logs, so a direct lookup works. Perspectives
    not covered by the Protocol-1 run (typically ``mturk_agent_2``) get
    abstentions rather than wrong answers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from opponent_model.hypotheses import ITEMS
from opponent_model.turn_agents import KeywordStrategyClassifier

logger = logging.getLogger("structured_cot.replay_turn_agent")


# ── JSONL loader ──────────────────────────────────────────────────────────


def _coerce_counts(maybe_counts: Any) -> Optional[Dict[str, int]]:
    """Turn a parsed counter_offer dict into a clean {Food,Water,Firewood}."""
    if not isinstance(maybe_counts, Mapping):
        return None
    try:
        return {it: int(maybe_counts[it]) for it in ITEMS}
    except (KeyError, TypeError, ValueError):
        return None


def _load_turns_jsonl(
    turns_path: Path,
) -> Dict[Tuple[Any, int, str], Dict[str, Any]]:
    """Load a Protocol-1 ``turns.jsonl`` into a ``(did, t, role) -> row`` map."""
    if not turns_path.exists():
        raise FileNotFoundError(f"turns.jsonl not found: {turns_path}")
    lookup: Dict[Tuple[Any, int, str], Dict[str, Any]] = {}
    with turns_path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "skipping malformed line %d in %s", line_no, turns_path,
                )
                continue
            did = row.get("dialogue_id")
            t = row.get("turn_index")
            role = row.get("agent_role")
            if did is None or t is None or not role:
                continue
            lookup[(did, int(t), str(role))] = row
    logger.info("Loaded %d structured-CoT turn records from %s",
                len(lookup), turns_path)
    return lookup


# ── Agent class ───────────────────────────────────────────────────────────


class StructuredCoTReplayAgent:
    """TurnLevelAgent wrapping a completed Protocol-1 ``turns.jsonl``.

    Decision mapping (matches how run_protocol3 scores things):
        parsed_decision.action == "accept"   → {accept=True,  bid=None}
        parsed_decision.action == "reject"   → {accept=False, bid=counter_offer or None}
        parsed_decision.action == "walkaway" → {accept=False, bid=None}
        parsed_decision missing / malformed  → all-None abstention

    Utterance → strategy is fed through the same ``KeywordStrategyClassifier``
    the Bayesian agent uses, so the two agents' strategy F1 numbers share
    a classifier and can be compared without bias.
    """

    def __init__(
        self,
        turns_path: Path,
        *,
        strategy_classifier: Optional[Callable] = None,
    ) -> None:
        self._lookup = _load_turns_jsonl(Path(turns_path))
        self.strategy_classifier = (
            strategy_classifier if strategy_classifier is not None
            else KeywordStrategyClassifier()
        )
        self._miss_count = 0
        self._hit_count = 0

    @property
    def summary(self) -> Dict[str, int]:
        return {"hits": self._hit_count, "misses": self._miss_count,
                "records_loaded": len(self._lookup)}

    # ── TurnLevelAgent protocol ────────────────────────────────────────────

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
        # turn_level_eval doesn't pass dialogue_id / turn_index directly;
        # infer from history length (== number of prior turns in chat_logs)
        # and dialogue from the most recent turn's dialogue_id if present.
        # Fallback: scan for any row whose key matches (did, len(history), my_role).
        turn_index = len(history)

        # Find did by scanning pending_offer or history for any 'dialogue_id'
        # embedded by the caller — CaSiNo's chat_logs entries themselves
        # don't carry a dialogue_id, so we rely on the harness wrapper
        # threading it through `history`. As a fallback, iterate over the
        # loaded lookup to find the unique row matching this (t, role).
        did = None
        for turn in history:
            if "dialogue_id" in turn:
                did = turn["dialogue_id"]
                break

        if did is None:
            # No dialogue_id in history → match by (turn_index, role) only.
            # This is O(N) in # records per turn, but N ~ 800 so it's fine.
            candidates = [
                r for (d, t, role), r in self._lookup.items()
                if t == turn_index and role == my_role
            ]
            if len(candidates) != 1:
                self._miss_count += 1
                return {"accept": None, "bid": None, "strategy": None,
                        "posterior": None}
            row = candidates[0]
        else:
            row = self._lookup.get((did, turn_index, my_role))
            if row is None:
                self._miss_count += 1
                return {"accept": None, "bid": None, "strategy": None,
                        "posterior": None}

        self._hit_count += 1

        parsed = row.get("parsed_decision") or {}
        action = parsed.get("action")
        counter = _coerce_counts(parsed.get("counter_offer"))

        if action == "accept":
            accept, bid = True, None
        elif action in ("reject", "walkaway"):
            accept = False
            bid = counter if action == "reject" else None
        else:
            accept, bid = None, None

        utterance = row.get("parsed_utterance") or ""
        strategy: Optional[List[str]] = None
        if utterance:
            try:
                tags = list(self.strategy_classifier(utterance, list(history)))
                strategy = tags or None
            except Exception:
                logger.exception("strategy classifier failed on replay row.")
                strategy = None

        return {
            "accept":    accept,
            "bid":       bid,
            "strategy":  strategy,
            "posterior": None,   # Structured CoT does not expose a posterior.
        }


# ── Convenience: inject dialogue_id into the dialogue before eval ─────────


def attach_dialogue_ids(dialogues: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Walk dialogues and stamp every chat_logs turn with its dialogue_id.

    ``turn_level_eval`` passes raw chat_logs turns to ``predict_turn``
    via ``history``. The replay agent needs to know which dialogue each
    history slice belongs to; copying ``dialogue_id`` onto each turn
    makes the lookup keyless and robust to reordering.
    """
    out: List[Dict[str, Any]] = []
    for d in dialogues:
        did = d.get("dialogue_id")
        new_chat_logs = []
        for turn in d.get("chat_logs", []):
            t = dict(turn)
            if did is not None and "dialogue_id" not in t:
                t["dialogue_id"] = did
            new_chat_logs.append(t)
        nd = dict(d)
        nd["chat_logs"] = new_chat_logs
        out.append(nd)
    return out


__all__ = [
    "StructuredCoTReplayAgent",
    "attach_dialogue_ids",
]
