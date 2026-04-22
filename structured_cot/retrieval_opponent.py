"""Retrieval-based opponent for Protocol 2.

At each opponent turn, look up the most similar context in the CaSiNo
training corpus (bucketed by the opponent's priority ordering, so the
retrieved opponent is strategically comparable to our scenario's
opponent), then replay that historical opponent's next utterance /
deal action as the response.

Design choices:
  * TF-IDF + cosine, numpy only. No GPU / no torch import. Index builds
    from ``data/casino_train.json`` in a few seconds.
  * One TF-IDF index per priority bucket (6 total). Keeps retrieval
    strategically consistent with the scenario's opponent priorities.
  * Context window = last K utterances, joined with speaker tags. K=8
    by default (covers ~3 exchange rounds).
  * Ties / near-matches are sampled with a temperature over cosine
    similarity so re-runs aren't deterministic. Set ``temperature=0``
    for argmax.
  * The retrieved speaker-turn's ``text`` + ``task_data`` are returned
    verbatim; the driver re-interprets Accept-Deal against our scenario's
    last pending offer and Submit-Deal's quantities are used as-is.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("structured_cot.retrieval_opponent")

ITEMS = ("Food", "Water", "Firewood")
PRIORITY_WEIGHTS = {"High": 5, "Medium": 4, "Low": 3}
WALKAWAY_POINTS = 5
DEAL_ACTIONS = {"Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away"}


# ── TF-IDF ─────────────────────────────────────────────────────────────────


_TOK_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> List[str]:
    return _TOK_RE.findall((text or "").lower())


class TfidfIndex:
    """Minimal TF-IDF + cosine similarity index; numpy only."""

    def __init__(self, documents: Sequence[str]) -> None:
        self.n_docs = len(documents)
        tokenized = [_tokenize(d) for d in documents]

        df = Counter()
        for toks in tokenized:
            for term in set(toks):
                df[term] += 1

        self.vocab: Dict[str, int] = {t: i for i, t in enumerate(sorted(df))}
        self.idf = np.zeros(len(self.vocab), dtype=np.float32)
        for t, i in self.vocab.items():
            self.idf[i] = math.log((self.n_docs + 1) / (df[t] + 1)) + 1.0

        self.matrix = np.zeros((self.n_docs, len(self.vocab)), dtype=np.float32)
        for row, toks in enumerate(tokenized):
            self._fill_row(row, toks)
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix /= norms

    def _fill_row(self, row: int, toks: Sequence[str]) -> None:
        tf = Counter(toks)
        for t, c in tf.items():
            i = self.vocab.get(t)
            if i is None:
                continue
            self.matrix[row, i] = c * self.idf[i]

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        tf = Counter(_tokenize(text))
        for t, c in tf.items():
            i = self.vocab.get(t)
            if i is None:
                continue
            vec[i] = c * self.idf[i]
        n = np.linalg.norm(vec)
        if n > 0:
            vec /= n
        return vec

    def topk(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        q = self.embed(query)
        sims = self.matrix @ q
        if k >= self.n_docs:
            order = np.argsort(-sims)
        else:
            part = np.argpartition(-sims, k)[:k]
            order = part[np.argsort(-sims[part])]
        return [(int(i), float(sims[i])) for i in order]


# ── Retrieval pool ─────────────────────────────────────────────────────────


@dataclass
class RetrievedTurn:
    """One training example: (context before speaker turn, speaker turn)."""

    dialogue_id:         Any
    speaker_role:        str
    speaker_priorities:  Tuple[str, str, str]   # (High, Medium, Low)
    context_text:        str                    # last K turns rendered as text
    response_text:       str                    # the speaker's raw text
    response_task_data:  Dict[str, Any] = field(default_factory=dict)


def _render_context(
    chat_logs: Sequence[Mapping[str, Any]],
    *,
    upto_index: int,
    speaker_role: str,
    max_turns: int = 8,
) -> str:
    recent = chat_logs[max(0, upto_index - max_turns):upto_index]
    parts: List[str] = []
    for t in recent:
        role = t.get("id")
        tag = "SPEAKER" if role == speaker_role else "OTHER"
        text = (t.get("text") or "").strip()
        if text in DEAL_ACTIONS:
            td = t.get("task_data") or {}
            if text == "Submit-Deal" and isinstance(td, dict):
                yg = {it: td.get("issue2youget", {}).get(it, "?") for it in ITEMS}
                tg = {it: td.get("issue2theyget", {}).get(it, "?") for it in ITEMS}
                text = (
                    f"Submit-Deal (proposer gets Food={yg['Food']},"
                    f" Water={yg['Water']}, Firewood={yg['Firewood']};"
                    f" other gets Food={tg['Food']}, Water={tg['Water']},"
                    f" Firewood={tg['Firewood']})"
                )
        if text:
            parts.append(f"{tag}: {text}")
    return "\n".join(parts) or "(empty context)"


def _normalize_priorities(info: Mapping[str, Any]) -> Optional[Tuple[str, str, str]]:
    p = info.get("value2issue") or {}
    if not set(p.keys()) >= {"High", "Medium", "Low"}:
        return None
    return (p["High"], p["Medium"], p["Low"])


def build_retrieval_pool(
    training_dialogues: Sequence[Mapping[str, Any]],
    *,
    context_turns: int = 8,
) -> Dict[Tuple[str, str, str], List[RetrievedTurn]]:
    """Walk every speaker turn in the corpus and collect ``(context, response)``.

    Bucketed by the speaker's priority ordering so that at query time
    we only compare against opponents whose strategic incentives match
    our scenario's opponent.
    """
    buckets: Dict[Tuple[str, str, str], List[RetrievedTurn]] = {}
    for d in training_dialogues:
        did = d.get("dialogue_id")
        chat_logs = d.get("chat_logs") or []
        info = d.get("participant_info") or {}
        for role in ("mturk_agent_1", "mturk_agent_2"):
            pri = _normalize_priorities(info.get(role) or {})
            if pri is None:
                continue
            for i, turn in enumerate(chat_logs):
                if turn.get("id") != role:
                    continue
                ctx = _render_context(
                    chat_logs, upto_index=i, speaker_role=role,
                    max_turns=context_turns,
                )
                buckets.setdefault(pri, []).append(RetrievedTurn(
                    dialogue_id=did,
                    speaker_role=role,
                    speaker_priorities=pri,
                    context_text=ctx,
                    response_text=(turn.get("text") or "").strip(),
                    response_task_data=dict(turn.get("task_data") or {}),
                ))
    return buckets


# ── The opponent ───────────────────────────────────────────────────────────


class RetrievalOpponent:
    """Plays the opponent role by TF-IDF kNN over the training corpus.

    Constructor caches a per-bucket TF-IDF index (built once, reused
    across dialogues) so running on 150 test dialogues does not pay the
    index-build cost repeatedly.
    """

    def __init__(
        self,
        pool: Mapping[Tuple[str, str, str], Sequence[RetrievedTurn]],
        *,
        context_turns: int = 8,
        top_k: int = 5,
        temperature: float = 0.7,
        seed: Optional[int] = 2024,
    ) -> None:
        self.context_turns = int(context_turns)
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self._rng = random.Random(seed)
        self._buckets: Dict[Tuple[str, str, str], List[RetrievedTurn]] = {
            k: list(v) for k, v in pool.items()
        }
        self._indices: Dict[Tuple[str, str, str], TfidfIndex] = {}

    def _index_for(self, priorities: Tuple[str, str, str]) -> Tuple[List[RetrievedTurn], TfidfIndex]:
        if priorities not in self._indices:
            entries = self._buckets.get(priorities) or []
            if not entries:
                # fall back to whole corpus if bucket is empty
                all_entries: List[RetrievedTurn] = []
                for bucket in self._buckets.values():
                    all_entries.extend(bucket)
                entries = all_entries
                logger.warning("Empty bucket %s; falling back to full corpus.",
                               priorities)
            index = TfidfIndex([e.context_text for e in entries])
            self._indices[priorities] = index
            self._buckets[priorities] = entries
        return self._buckets[priorities], self._indices[priorities]

    def _sample_top(self, sims: List[Tuple[int, float]]) -> int:
        if not sims:
            raise RuntimeError("Empty retrieval result.")
        if self.temperature <= 0 or len(sims) == 1:
            return sims[0][0]
        vals = np.array([s for _, s in sims], dtype=np.float64)
        logits = vals / self.temperature
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        choice = self._rng.random()
        cum = 0.0
        for (idx, _), p in zip(sims, probs):
            cum += float(p)
            if choice <= cum:
                return idx
        return sims[-1][0]

    def respond(
        self,
        *,
        priorities: Tuple[str, str, str],
        context: Sequence[Tuple[str, str]],
    ) -> Dict[str, Any]:
        """Retrieve a response for the given priorities + history.

        ``context`` is ``[(speaker_tag, utterance), ...]`` with ``me`` for
        the opponent itself and ``opp`` for the agent, oldest first. Internally
        we render the last K turns with the same SPEAKER/OTHER convention
        the pool was indexed with (SPEAKER = the retrieval opponent).
        """
        entries, index = self._index_for(priorities)
        context = list(context)[-self.context_turns:]
        lines = []
        for tag, utt in context:
            who = "SPEAKER" if tag in ("me", "opp_to_agent", "opponent") else "OTHER"
            lines.append(f"{who}: {utt}")
        query = "\n".join(lines) or "(empty context)"

        candidates = index.topk(query, k=self.top_k)
        chosen_idx = self._sample_top(candidates)
        chosen = entries[chosen_idx]

        return {
            "text":          chosen.response_text,
            "task_data":     dict(chosen.response_task_data),
            "retrieved_from": {
                "dialogue_id":  chosen.dialogue_id,
                "speaker_role": chosen.speaker_role,
                "similarity":   float(
                    next((s for i, s in candidates if i == chosen_idx), 0.0)
                ),
                "topk":         [{"idx": i, "sim": s} for i, s in candidates],
            },
        }


# ── Scoring helpers (shared with the driver) ───────────────────────────────


def points_for(
    quantities: Mapping[str, int],
    priorities: Tuple[str, str, str],
) -> int:
    """CaSiNo points for one side: sum(count * priority-weight)."""
    priority_of = {priorities[0]: "High", priorities[1]: "Medium", priorities[2]: "Low"}
    return sum(
        int(quantities.get(it, 0)) * PRIORITY_WEIGHTS[priority_of[it]]
        for it in ITEMS
    )


def pareto_max_self(
    priorities_self: Tuple[str, str, str],
    priorities_opp:  Tuple[str, str, str],
    *,
    opp_floor: int = 15,
) -> int:
    """Max attainable self-score over all splits where opp_points >= floor."""
    best = 0
    for f in range(4):
        for w in range(4):
            for fw in range(4):
                self_deal = {"Food": f, "Water": w, "Firewood": fw}
                opp_deal  = {"Food": 3 - f, "Water": 3 - w, "Firewood": 3 - fw}
                s_self = points_for(self_deal, priorities_self)
                s_opp  = points_for(opp_deal,  priorities_opp)
                if s_opp >= opp_floor and s_self > best:
                    best = s_self
    return best or 1


def load_training_corpus(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        return json.load(f)


__all__ = [
    "ITEMS",
    "PRIORITY_WEIGHTS",
    "WALKAWAY_POINTS",
    "DEAL_ACTIONS",
    "TfidfIndex",
    "RetrievedTurn",
    "RetrievalOpponent",
    "build_retrieval_pool",
    "load_training_corpus",
    "points_for",
    "pareto_max_self",
]
