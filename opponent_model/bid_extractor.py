"""Rule-based bid extraction from CaSiNo free-text negotiation utterances.

The goal is high precision, not maximum recall. We only return a bid when the
utterance states one unique legal CaSiNo split, either directly or via a
single unambiguous complement cue such as "you take the rest".
"""

from __future__ import annotations

import re
from typing import Dict, Optional

from opponent_model.hypotheses import ITEMS


_WORD_COUNTS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
}

_WORD_COUNT_RE = re.compile(
    r"\b(" + "|".join(sorted(_WORD_COUNTS, key=len, reverse=True)) + r")\b"
)

_DECIMAL_RE = re.compile(r"\b\d+\.\d+\b")
_HALF_RE = re.compile(
    r"\b(?:half|halves|one and a half|two and a half|three and a half)\b"
)
_RANGE_RE = re.compile(
    r"\b(?:\d+|zero|one|two|three)\s*(?:or|-|to)\s*(?:\d+|zero|one|two|three)\b"
)
_ALTERNATIVE_RE = re.compile(
    r"\b(?:or|alternatively)\b.*\b(?:another|else|other option|other options|adjust|different)\b"
)

_SIDE_CUE_RE = re.compile(
    r"(?P<self>"
    r"\bi propose\b|"
    r"\bi propose i take\b|"
    r"\bi would like to take\b|"
    r"\bi would take\b|"
    r"\bi'll take\b|"
    r"\bi will take\b|"
    r"\bi can take\b|"
    r"\bi could take\b|"
    r"\bi'll get\b|"
    r"\bi will get\b|"
    r"\bi could get\b|"
    r"\bi get\b|"
    r"\bi take\b|"
    r"\bme getting\b|"
    r"\bme get\b|"
    r"\bfor me to get\b|"
    r"\bgive me\b|"
    r"\bgives me\b|"
    r"\byou'll give me\b|"
    r"\byou will give me\b|"
    r"\byou give me\b"
    r")|(?P<opp>"
    r"\bfor you to get\b|"
    r"\byou can have\b|"
    r"\byou'll take\b|"
    r"\byou will take\b|"
    r"\byou'll get\b|"
    r"\byou will get\b|"
    r"\byou get\b|"
    r"\byou take\b|"
    r"\byou getting\b|"
    r"\bleaving you with\b|"
    r"\bi'll give you\b|"
    r"\bi will give you\b|"
    r"\bi give you\b"
    r")"
)

_ITEM_RE = re.compile(
    r"(?:(?P<item_eq>food|water|firewood)\s*=\s*(?P<count_eq>\d+))"
    r"|(?:(?P<count>\d+)\s*(?:pack(?:age)?s?)?(?:\s+of)?\s+(?P<item>food|water|firewood)s?\b)"
    r"|(?:(?P<all>all)\s+(?:the\s+)?(?P<item_all>food|water|firewood)\b)"
    r"|(?:(?:all\s+)?the\s+(?P<item_all_packages>food|water|firewood)\s+pack(?:age)?s?\b)"
    r"|(?:(?P<none>no)\s+(?:pack(?:age)?s?)?(?:\s+of)?\s+(?P<item_none>food|water|firewood)s?\b)"
)

_SUFFIX_BODY = (
    r"(?:[\s,]|and|packages?|package|packs?|pack|of|the|food|water|firewood|all|no|\d+)+"
)
_SELF_SUFFIX_RE = re.compile(r"(?P<body>" + _SUFFIX_BODY + r")\s+for me\b")

_SELF_REST_RE = re.compile(
    r"\bi(?:'ll| will)? take the rest\b|\bi get the rest\b|\brest for me\b|\bleaving the rest for me\b"
)
_OPP_REST_RE = re.compile(
    r"\byou(?:'ll| will)? take the rest\b|\byou get the rest\b|\brest for you\b|\bleaving the rest for you\b|\byou take the rest\b"
)


def _normalize_text(text: str) -> str:
    out = str(text or "").lower()
    out = out.replace("’", "'").replace("“", '"').replace("”", '"')
    out = out.replace("—", " ").replace("–", " ")
    out = _WORD_COUNT_RE.sub(lambda m: _WORD_COUNTS[m.group(1)], out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _extract_counts(clause: str) -> Optional[Dict[str, int]]:
    counts: Dict[str, int] = {}
    for match in _ITEM_RE.finditer(clause):
        if match.group("item_eq") is not None:
            item = match.group("item_eq").title()
            count = int(match.group("count_eq"))
        elif match.group("item") is not None:
            item = match.group("item").title()
            count = int(match.group("count"))
        elif match.group("item_all") is not None:
            item = match.group("item_all").title()
            count = 3
        elif match.group("item_all_packages") is not None:
            item = match.group("item_all_packages").title()
            count = 3
        else:
            item = match.group("item_none").title()
            count = 0
        if count < 0 or count > 3:
            return None
        if item in counts and counts[item] != count:
            return None
        counts[item] = count
    return counts

def _merge_side_counts(
    current: Dict[str, int],
    new_counts: Dict[str, int],
) -> bool:
    for item, count in new_counts.items():
        if item in current and current[item] != count:
            return False
        current[item] = count
    return True


def _resolve_split(
    self_counts: Dict[str, int],
    opp_counts: Dict[str, int],
    *,
    self_rest: bool,
    opp_rest: bool,
) -> Optional[Dict[str, object]]:
    if self_rest and opp_rest:
        return None

    self_full: Dict[str, int] = {}
    opp_full: Dict[str, int] = {}
    for item in ITEMS:
        self_has = item in self_counts
        opp_has = item in opp_counts
        if self_has and opp_has:
            if self_counts[item] + opp_counts[item] != 3:
                return None
            self_full[item] = self_counts[item]
            opp_full[item] = opp_counts[item]
            continue
        if self_has:
            self_full[item] = self_counts[item]
            opp_full[item] = 3 - self_counts[item]
            continue
        if opp_has:
            opp_full[item] = opp_counts[item]
            self_full[item] = 3 - opp_counts[item]
            continue
        if opp_rest:
            self_full[item] = 0
            opp_full[item] = 3
            continue
        if self_rest:
            self_full[item] = 3
            opp_full[item] = 0
            continue
        return None

    for item in ITEMS:
        if self_full[item] + opp_full[item] != 3:
            return None
        if self_full[item] < 0 or opp_full[item] < 0:
            return None

    return {
        "self_counts": self_full,
        "opp_counts": opp_full,
        "self_tuple": [self_full[item] for item in ITEMS],
        "opp_tuple": [opp_full[item] for item in ITEMS],
    }


def _merge_suffix_counts(
    text: str,
    regex: re.Pattern[str],
    target: Dict[str, int],
) -> bool:
    for match in regex.finditer(text):
        counts = _extract_counts(match.group("body"))
        if counts is None:
            return False
        if counts and not _merge_side_counts(target, counts):
            return False
    return True


def extract_bid_from_utterance(text: str) -> Optional[Dict[str, object]]:
    """Return a canonical CaSiNo split when one unique bid is stated."""
    norm = _normalize_text(text)
    if not norm:
        return None
    if (
        _DECIMAL_RE.search(norm)
        or _HALF_RE.search(norm)
        or _RANGE_RE.search(norm)
        or _ALTERNATIVE_RE.search(norm)
    ):
        return None

    cues = list(_SIDE_CUE_RE.finditer(norm))
    self_counts: Dict[str, int] = {}
    opp_counts: Dict[str, int] = {}
    for idx, cue in enumerate(cues):
        side = "self" if cue.group("self") is not None else "opp"
        start = cue.end()
        end = cues[idx + 1].start() if idx + 1 < len(cues) else len(norm)
        clause = norm[start:end]
        counts = _extract_counts(clause)
        if counts is None:
            return None
        if side == "self":
            if not _merge_side_counts(self_counts, counts):
                return None
        else:
            if not _merge_side_counts(opp_counts, counts):
                return None

    if not _merge_suffix_counts(norm, _SELF_SUFFIX_RE, self_counts):
        return None
    self_rest = bool(_SELF_REST_RE.search(norm))
    opp_rest = bool(_OPP_REST_RE.search(norm))
    return _resolve_split(
        self_counts,
        opp_counts,
        self_rest=self_rest,
        opp_rest=opp_rest,
    )


__all__ = ["extract_bid_from_utterance"]
