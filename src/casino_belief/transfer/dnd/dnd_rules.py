"""DND-specific rule likelihoods based on transactional offer language."""

from __future__ import annotations

import math
import re
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from casino_belief.transfer.dnd.dnd_data import DND_ITEMS, canonical_item

NUMBER_WORDS = {
    "zero": 0,
    "none": 0,
    "no": 0,
    "one": 1,
    "a": 1,
    "an": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
}

ITEM_RE = r"books?|hats?|balls?|food|water|fire\s*wood|firewood|wood"


def _softmax(scores: Sequence[float]) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    arr = arr - float(np.max(arr))
    exp = np.exp(arr)
    total = float(exp.sum())
    if total <= 0:
        return np.full(len(arr), 1.0 / len(arr), dtype=np.float64)
    return exp / total


def _quantity(raw: Optional[str], *, item: str, counts: Mapping[str, int]) -> float:
    if raw is None or not str(raw).strip():
        return 1.0
    text = str(raw).strip().lower()
    if "all" in text:
        return float(counts.get(item, 1) or 1)
    if text.isdigit():
        return float(int(text))
    return float(NUMBER_WORDS.get(text, 1))


def _add_share_evidence(
    evidence: Dict[str, float],
    *,
    item: str,
    qty: float,
    owner: str,
    counts: Mapping[str, int],
) -> None:
    denom = max(float(counts.get(item, 1) or 1), 1.0)
    strength = max(0.25, min(2.5, qty / denom * 2.0))
    if owner == "opp":
        evidence[item] += strength
    elif owner == "self":
        evidence[item] -= strength


def score_utterance(text: str, *, counts: Mapping[str, int]) -> Dict[str, float]:
    """Return item evidence from the opponent's utterance.

    Positive values mean the opponent seems to value/keep the item; negative
    values mean they are giving it away or declaring low value.
    """
    low = " ".join(str(text).lower().replace(",", " , ").split())
    evidence = {item: 0.0 for item in DND_ITEMS}

    # Explicit value/need statements.
    for m in re.finditer(rf"\b(?:i|we)\s+(?:really\s+)?(?:need|want|prefer|like)\s+(?P<item>{ITEM_RE})", low):
        item = canonical_item(m.group("item"))
        if item:
            evidence[item] += 2.0
    for m in re.finditer(rf"\b(?P<item>{ITEM_RE})\s+(?:is|are)\s+(?:important|valuable|worth)", low):
        item = canonical_item(m.group("item"))
        if item:
            evidence[item] += 1.5
    for m in re.finditer(rf"\b(?P<item>{ITEM_RE})\s+(?:has|have)\s+no\s+value\s+for\s+me", low):
        item = canonical_item(m.group("item"))
        if item:
            evidence[item] -= 2.5
    for m in re.finditer(rf"\b(?:i|we)\s+(?:do\s+not|don't|dont)\s+need\s+(?P<item>{ITEM_RE})", low):
        item = canonical_item(m.group("item"))
        if item:
            evidence[item] -= 2.0

    # "I get/take/have/keep all/2 books" means opponent keeps the item.
    keep_patterns = [
        rf"\b(?:i|we)\s+(?:get|take|have|keep|receive)\s+(?P<qty>all|[0-7]|zero|one|two|three|four|five|six|seven|a|an)?\s*(?:the\s+)?(?P<item>{ITEM_RE})",
        rf"\b(?P<qty>all|[0-7]|zero|one|two|three|four|five|six|seven|a|an)\s+(?P<item>{ITEM_RE})\s+(?:for\s+)?(?:me|us)\b",
    ]
    for pat in keep_patterns:
        for m in re.finditer(pat, low):
            item = canonical_item(m.group("item"))
            if item:
                _add_share_evidence(
                    evidence,
                    item=item,
                    qty=_quantity(m.groupdict().get("qty"), item=item, counts=counts),
                    owner="opp",
                    counts=counts,
                )

    # "You get/can have all/2 books" means opponent gives the item away.
    give_patterns = [
        rf"\b(?:you|u)\s+(?:can\s+|may\s+)?(?:get|take|have|keep|receive)\s+(?P<qty>all|[0-7]|zero|one|two|three|four|five|six|seven|a|an)?\s*(?:the\s+)?(?P<item>{ITEM_RE})",
        rf"\b(?P<qty>all|[0-7]|zero|one|two|three|four|five|six|seven|a|an)\s+(?P<item>{ITEM_RE})\s+(?:for\s+)?you\b",
    ]
    for pat in give_patterns:
        for m in re.finditer(pat, low):
            item = canonical_item(m.group("item"))
            if item:
                _add_share_evidence(
                    evidence,
                    item=item,
                    qty=_quantity(m.groupdict().get("qty"), item=item, counts=counts),
                    owner="self",
                    counts=counts,
                )

    return evidence


def posterior_from_evidence(
    evidence: Mapping[str, float],
    orderings: Sequence[Tuple[str, str, str]],
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    if not evidence or all(abs(float(v)) < 1e-9 for v in evidence.values()):
        return np.full(len(orderings), 1.0 / len(orderings), dtype=np.float64)
    rank_weight = {0: 1.0, 1: 0.0, 2: -1.0}
    scores = []
    for ordering in orderings:
        s = 0.0
        for rank, item in enumerate(ordering):
            s += rank_weight[rank] * float(evidence.get(item, 0.0))
        scores.append(s / max(float(temperature), 1e-6))
    return _softmax(scores)


def combine_evidence(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    out = {item: 0.0 for item in DND_ITEMS}
    for row in rows:
        for item in DND_ITEMS:
            out[item] += float(row.get(item, 0.0))
    return out
