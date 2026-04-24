"""Parser for the Day 8 distilled student tagged-text format.

The student is trained to emit four sections:

    <posterior>...</posterior>
    <selected_intent>...</selected_intent>
    <selected_content>...</selected_content>
    <utterance>...</utterance>

This module is deliberately defensive. It never raises on malformed model
output; instead it returns a structured dict with parse-error details so
callers can log failures, abstain on broken fields, and keep long eval jobs
running.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from sft_8b.posterior import ORDERINGS
from sft_8b.prompts import ITEMS


VALID_INTENTS = frozenset({"submit", "accept", "reject", "walkaway", "utter"})
REQUIRED_TAGS = ("posterior", "selected_intent", "selected_content", "utterance")

_TAG_PATTERNS: Dict[str, re.Pattern] = {
    tag: re.compile(
        rf"<{tag}\s*>(.*?)</{tag}\s*>",
        re.DOTALL | re.IGNORECASE,
    )
    for tag in REQUIRED_TAGS
}

_POSTERIOR_LINE_RE = re.compile(
    r"^p\((?P<ordering>[^)]+)\)\s*=\s*(?P<prob>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)$"
)
_JSON_CODEFENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_tag(text: str, tag: str) -> Optional[str]:
    match = _TAG_PATTERNS[tag].search(text or "")
    if not match:
        open_match = re.search(rf"<{tag}\s*>", text or "", re.IGNORECASE)
        if not open_match:
            return None
        tail = (text or "")[open_match.end() :]
        next_tag_pos: Optional[int] = None
        for other in REQUIRED_TAGS:
            if other == tag:
                continue
            other_match = re.search(rf"<{other}\s*>", tail, re.IGNORECASE)
            if other_match is None:
                continue
            if next_tag_pos is None or other_match.start() < next_tag_pos:
                next_tag_pos = other_match.start()
        body = tail if next_tag_pos is None else tail[:next_tag_pos]
        body = body.strip()
        return body or None
    return match.group(1).strip()


def _extract_json_blob(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    fenced = _JSON_CODEFENCE.search(raw)
    if fenced:
        try:
            obj = json.loads(fenced.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(raw[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def _coerce_counts_map(
    raw: Mapping[str, Any],
    *,
    label: str,
) -> Tuple[Optional[Dict[str, int]], List[str]]:
    errs: List[str] = []
    counts: Dict[str, int] = {}
    for item in ITEMS:
        if item not in raw:
            errs.append(f"{label} missing key {item!r}")
            continue
        value = raw.get(item)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errs.append(f"{label}[{item!r}] must be integer, got {value!r}")
            continue
        if int(value) != float(value):
            errs.append(f"{label}[{item!r}] must be integer, got {value!r}")
            continue
        iv = int(value)
        if iv < 0 or iv > 3:
            errs.append(f"{label}[{item!r}]={iv} out of range [0, 3]")
            continue
        counts[item] = iv
    if errs:
        return None, errs
    return counts, []


def normalize_selected_content(
    raw: Mapping[str, Any],
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Return canonical submit content or errors.

    Accepted shapes:
        * {"self_counts": {...}, "opp_counts": {...}}
        * {"self": {...}, "opp": {...}}
        * {"issue2youget": {...}, "issue2theyget": {...}}
        * {"Food": 1, "Water": 2, "Firewood": 0}
        * {"self_tuple": [1, 2, 0], "opp_tuple": [2, 1, 3]}
    """
    errs: List[str] = []

    if "self_counts" in raw and "opp_counts" in raw:
        self_counts, e1 = _coerce_counts_map(raw["self_counts"], label="self_counts")
        opp_counts, e2 = _coerce_counts_map(raw["opp_counts"], label="opp_counts")
        errs.extend(e1)
        errs.extend(e2)
    elif "self" in raw and "opp" in raw:
        self_counts, e1 = _coerce_counts_map(raw["self"], label="self")
        opp_counts, e2 = _coerce_counts_map(raw["opp"], label="opp")
        errs.extend(e1)
        errs.extend(e2)
    elif "issue2youget" in raw and "issue2theyget" in raw:
        self_counts, e1 = _coerce_counts_map(raw["issue2youget"], label="issue2youget")
        opp_counts, e2 = _coerce_counts_map(raw["issue2theyget"], label="issue2theyget")
        errs.extend(e1)
        errs.extend(e2)
    elif all(item in raw for item in ITEMS):
        self_counts, e1 = _coerce_counts_map(raw, label="selected_content")
        errs.extend(e1)
        opp_counts = None
        if self_counts is not None:
            opp_counts = {item: 3 - self_counts[item] for item in ITEMS}
    elif "self_tuple" in raw:
        tup = raw.get("self_tuple")
        if not isinstance(tup, (list, tuple)) or len(tup) != 3:
            errs.append("self_tuple must be a length-3 sequence")
            self_counts = None
            opp_counts = None
        else:
            try:
                self_counts = {item: int(tup[i]) for i, item in enumerate(ITEMS)}
            except (TypeError, ValueError):
                errs.append("self_tuple must contain integers")
                self_counts = None
                opp_counts = None
            else:
                opp_counts = {item: 3 - self_counts[item] for item in ITEMS}
                if "opp_tuple" in raw:
                    opp_tup = raw.get("opp_tuple")
                    if not isinstance(opp_tup, (list, tuple)) or len(opp_tup) != 3:
                        errs.append("opp_tuple must be a length-3 sequence")
                    else:
                        try:
                            opp_counts = {
                                item: int(opp_tup[i]) for i, item in enumerate(ITEMS)
                            }
                        except (TypeError, ValueError):
                            errs.append("opp_tuple must contain integers")
    else:
        return None, ["selected_content JSON has no recognized bid shape"]

    if errs or self_counts is None or opp_counts is None:
        return None, errs

    for item in ITEMS:
        total = self_counts[item] + opp_counts[item]
        if total != 3:
            errs.append(
                f"selected_content violates CaSiNo invariant for {item}: "
                f"self={self_counts[item]}, opp={opp_counts[item]}, total={total}"
            )

    if errs:
        return None, errs

    return {
        "self_counts": self_counts,
        "opp_counts": opp_counts,
        "self_tuple": [self_counts[item] for item in ITEMS],
        "opp_tuple": [opp_counts[item] for item in ITEMS],
    }, []


def parse_posterior_block(
    text: str,
    *,
    orderings: Sequence[Sequence[str]] = ORDERINGS,
) -> Tuple[Optional[List[float]], List[str]]:
    errs: List[str] = []
    ordering_to_idx = {tuple(order): i for i, order in enumerate(orderings)}
    probs: List[Optional[float]] = [None] * len(orderings)

    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if len(lines) != len(orderings):
        errs.append(
            f"posterior has {len(lines)} non-empty lines; expected {len(orderings)}"
        )

    for line in lines:
        match = _POSTERIOR_LINE_RE.match(line)
        if not match:
            errs.append(f"malformed posterior line: {line!r}")
            continue
        ordering = tuple(part.strip() for part in match.group("ordering").split(">"))
        idx = ordering_to_idx.get(ordering)
        if idx is None:
            errs.append(f"unknown posterior ordering: {ordering!r}")
            continue
        if probs[idx] is not None:
            errs.append(f"duplicate posterior ordering: {ordering!r}")
            continue
        prob = float(match.group("prob"))
        if prob < 0:
            errs.append(f"negative posterior mass for {ordering!r}: {prob}")
            continue
        probs[idx] = prob

    missing = [
        " > ".join(orderings[i]) for i, prob in enumerate(probs) if prob is None
    ]
    if missing:
        errs.append(f"posterior missing orderings: {missing}")
        return None, errs

    total = float(sum(prob for prob in probs if prob is not None))
    if total <= 0:
        errs.append("posterior mass sums to zero")
        return None, errs

    norm = [float(prob) / total for prob in probs if prob is not None]
    return norm, errs


def parse_student_response(text: str) -> Dict[str, Any]:
    """Parse a student generation into structured fields plus error metadata."""
    result: Dict[str, Any] = {
        "posterior": None,
        "selected_intent": None,
        "selected_content": None,
        "utterance": None,
        "posterior_raw": None,
        "selected_content_raw": None,
        "parse_error": None,
        "missing_tags": [],
        "posterior_errors": [],
        "intent_errors": [],
        "selected_content_errors": [],
    }

    raw_blocks = {tag: _extract_tag(text or "", tag) for tag in REQUIRED_TAGS}
    result["posterior_raw"] = raw_blocks["posterior"]
    result["selected_content_raw"] = raw_blocks["selected_content"]
    result["utterance"] = raw_blocks["utterance"]

    missing = [tag for tag, body in raw_blocks.items() if body is None]
    result["missing_tags"] = missing

    if raw_blocks["posterior"] is not None:
        posterior, errs = parse_posterior_block(raw_blocks["posterior"])
        result["posterior_errors"] = errs
        result["posterior"] = posterior

    if raw_blocks["selected_intent"] is not None:
        intent = str(raw_blocks["selected_intent"]).strip().lower()
        if intent not in VALID_INTENTS:
            result["intent_errors"].append(
                f"selected_intent must be one of {sorted(VALID_INTENTS)}, got {intent!r}"
            )
        else:
            result["selected_intent"] = intent

    if raw_blocks["selected_content"] is not None:
        body = raw_blocks["selected_content"].strip()
        if body.lower() == "null":
            result["selected_content"] = None
        else:
            blob = _extract_json_blob(body)
            if blob is None:
                result["selected_content_errors"].append(
                    "could not extract JSON object from <selected_content>"
                )
            else:
                content, errs = normalize_selected_content(blob)
                result["selected_content"] = content
                result["selected_content_errors"].extend(errs)

    intent = result["selected_intent"]
    content = result["selected_content"]
    if intent == "submit" and content is None:
        result["selected_content_errors"].append(
            "submit intent requires non-null selected_content"
        )
    if (
        intent in {"accept", "walkaway", "utter"}
        and content is not None
    ):
        result["selected_content_errors"].append(
            f"{intent} intent should have selected_content=null"
        )

    parts: List[str] = []
    if missing:
        parts.append(f"missing tags: {missing}")
    if result["posterior_errors"]:
        parts.append(f"posterior errors: {result['posterior_errors']}")
    if result["intent_errors"]:
        parts.append(f"intent errors: {result['intent_errors']}")
    if result["selected_content_errors"]:
        parts.append(f"selected_content errors: {result['selected_content_errors']}")
    if parts:
        result["parse_error"] = "; ".join(parts)

    return result


__all__ = [
    "REQUIRED_TAGS",
    "VALID_INTENTS",
    "normalize_selected_content",
    "parse_posterior_block",
    "parse_student_response",
]
