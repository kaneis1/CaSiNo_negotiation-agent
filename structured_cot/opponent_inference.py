"""Structured opponent-inference prompts, parser, and scoring helpers.

This module is intentionally model-agnostic. The runner can use a HF
``transformers.pipeline`` or any other backend as long as it returns text.
"""

from __future__ import annotations

import json
import math
import re
from itertools import product
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from opponent_model.hypotheses import HYPOTHESES, ITEMS
from opponent_model.turn_level_metrics import normalized_brier


ITEM_SET = set(ITEMS)
LOWER_TO_ITEM = {item.lower(): item for item in ITEMS}
VALID_CONFIDENCE = frozenset({"high", "medium", "low"})
DEAL_ACTIONS = frozenset({"Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away"})
PRIORITY_POINTS = {"High": 5, "Medium": 4, "Low": 3}

_TAG_RE_CACHE: Dict[str, re.Pattern[str]] = {}
_JSON_CODEFENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


SYSTEM_MESSAGE = """\
You infer an opponent's hidden priorities in the CaSiNo camping negotiation.
There are exactly three issues: Food, Water, and Firewood.

Return exactly one XML block and nothing else:

<opponent_inference>
  <evidence>List each opponent utterance or offer so far that signals priority.
  Tag preference-related utterances with Self-Need, Other-Need, or No-Need
  when applicable, and include explicit offers.</evidence>
  <interpretation>For each evidence item, state the directional priority
  implication, e.g. "needs food for kids -> Food likely high".</interpretation>
  <ranking>{"food":1,"water":2,"firewood":3,"confidence":{"food":"high","water":"medium","firewood":"low"}}</ranking>
  <rationale>One sentence connecting the interpretation to the ranking.</rationale>
</opponent_inference>

Ranking rules:
- Rank 1 is the opponent's highest priority and rank 3 is lowest.
- Use exactly the JSON keys food, water, firewood, confidence.
- Confidence values must be high, medium, or low.
- Do not include markdown fences or prose outside the XML block.
"""

RANKING_ONLY_SYSTEM_MESSAGE = """\
You infer an opponent's hidden priorities in the CaSiNo camping negotiation.
There are exactly three issues: Food, Water, and Firewood.

Return exactly one XML block and nothing else:

<ranking>{"food":1,"water":2,"firewood":3,"confidence":{"food":"high","water":"medium","firewood":"low"}}</ranking>

Ranking rules:
- Rank 1 is the opponent's highest priority and rank 3 is lowest.
- Use exactly the JSON keys food, water, firewood, confidence.
- Confidence values must be high, medium, or low.
- Do not include markdown fences or prose outside the XML block.
"""


USER_TEMPLATE = """\
## Your side

Your priorities and reasons:
- High priority: {high_item} ({high_reason})
- Medium priority: {med_item} ({med_reason})
- Low priority: {low_item} ({low_reason})

## Dialogue so far

{history_block}

## Task

Infer the opponent's priority ranking from the dialogue so far. Emit only the
single <opponent_inference> block.
"""

RANKING_ONLY_USER_TEMPLATE = """\
## Your side

Your priorities and reasons:
- High priority: {high_item} ({high_reason})
- Medium priority: {med_item} ({med_reason})
- Low priority: {low_item} ({low_reason})

## Dialogue prefix

{history_block}

## Task

Infer the opponent's priority ranking from this dialogue prefix. Emit only the
single <ranking> block.
"""


def tag_pattern(name: str) -> re.Pattern[str]:
    pat = _TAG_RE_CACHE.get(name)
    if pat is None:
        pat = re.compile(rf"<{name}\s*>(.*?)</{name}\s*>", re.DOTALL | re.IGNORECASE)
        _TAG_RE_CACHE[name] = pat
    return pat


def extract_tag(text: str, name: str) -> Optional[str]:
    m = tag_pattern(name).search(text or "")
    if not m:
        return None
    return m.group(1).strip()


def _extract_json_blob(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    m = _JSON_CODEFENCE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
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
                    obj = json.loads(text[start : i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def _canonical_item(key: Any) -> Optional[str]:
    return LOWER_TO_ITEM.get(str(key).strip().lower())


def validate_ranking_json(obj: Mapping[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Validate and normalize the ``<ranking>`` JSON object."""
    errors: List[str] = []
    ranks: Dict[str, int] = {}
    for raw_key, raw_value in obj.items():
        if str(raw_key).strip().lower() == "confidence":
            continue
        item = _canonical_item(raw_key)
        if item is None:
            errors.append(f"unexpected ranking key {raw_key!r}")
            continue
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            errors.append(f"rank for {item} must be numeric, got {raw_value!r}")
            continue
        if int(raw_value) != float(raw_value):
            errors.append(f"rank for {item} must be an integer, got {raw_value!r}")
            continue
        ranks[item] = int(raw_value)

    missing = [item for item in ITEMS if item not in ranks]
    if missing:
        errors.append(f"ranking missing items {missing}")
    if sorted(ranks.values()) != [1, 2, 3]:
        errors.append(f"ranking values must be exactly [1, 2, 3], got {ranks}")

    conf_obj = obj.get("confidence")
    confidence: Dict[str, str] = {}
    if not isinstance(conf_obj, Mapping):
        errors.append("confidence must be an object with food/water/firewood keys")
    else:
        for raw_key, raw_value in conf_obj.items():
            item = _canonical_item(raw_key)
            if item is None:
                errors.append(f"unexpected confidence key {raw_key!r}")
                continue
            val = str(raw_value).strip().lower()
            if val not in VALID_CONFIDENCE:
                errors.append(
                    f"confidence for {item} must be high/medium/low, got {raw_value!r}"
                )
                continue
            confidence[item] = val
        missing_conf = [item for item in ITEMS if item not in confidence]
        if missing_conf:
            errors.append(f"confidence missing items {missing_conf}")

    if errors:
        return None, errors

    ordering = [item for item, _ in sorted(ranks.items(), key=lambda kv: kv[1])]
    return {
        "ranking": ranks,
        "confidence": confidence,
        "ordering": ordering,
        "hypothesis_index": hypothesis_index(ordering),
    }, []


def parse_opponent_inference_response(text: str) -> Dict[str, Any]:
    """Parse the nested opponent-inference response.

    The returned dict never raises. ``parse_error`` is ``None`` only when
    all required nested tags are present and ``<ranking>`` validates.
    """
    result: Dict[str, Any] = {
        "opponent_inference_raw": None,
        "evidence": None,
        "interpretation": None,
        "ranking_raw": None,
        "rationale": None,
        "ranking": None,
        "confidence": None,
        "ordering": None,
        "hypothesis_index": None,
        "parse_error": None,
        "missing_tags": [],
        "ranking_errors": [],
    }

    block = extract_tag(text or "", "opponent_inference")
    if block is None:
        result["parse_error"] = "missing <opponent_inference> block"
        result["missing_tags"] = ["opponent_inference"]
        return result
    result["opponent_inference_raw"] = block

    for tag in ("evidence", "interpretation", "ranking", "rationale"):
        result[tag if tag != "ranking" else "ranking_raw"] = extract_tag(block, tag)

    missing = [
        tag for tag in ("evidence", "interpretation", "ranking", "rationale")
        if not result[tag if tag != "ranking" else "ranking_raw"]
    ]
    result["missing_tags"] = missing

    if result["ranking_raw"]:
        obj = _extract_json_blob(result["ranking_raw"])
        if obj is None:
            result["ranking_errors"].append("could not parse ranking JSON object")
        else:
            normalized, errors = validate_ranking_json(obj)
            if errors:
                result["ranking_errors"].extend(errors)
            elif normalized is not None:
                result.update(normalized)

    if missing or result["ranking_errors"]:
        parts: List[str] = []
        if missing:
            parts.append(f"missing tags: {missing}")
        if result["ranking_errors"]:
            parts.append(f"ranking errors: {result['ranking_errors']}")
        result["parse_error"] = "; ".join(parts)

    return result


def parse_ranking_response(text: str) -> Dict[str, Any]:
    """Parse either standalone ``<ranking>`` or nested opponent inference."""
    ranking_raw = extract_tag(text or "", "ranking")
    if ranking_raw is None:
        nested = parse_opponent_inference_response(text)
        if nested.get("parse_error") is None or nested.get("ranking_raw") is not None:
            return nested
        return {
            "parse_error": "missing <ranking> block",
            "valid": False,
            "ranking_raw": None,
            "ranking": None,
            "confidence": None,
            "ordering": None,
            "hypothesis_index": None,
            "missing_tags": ["ranking"],
            "ranking_errors": [],
        }

    result: Dict[str, Any] = {
        "parse_error": None,
        "valid": False,
        "ranking_raw": ranking_raw,
        "ranking": None,
        "confidence": None,
        "ordering": None,
        "hypothesis_index": None,
        "missing_tags": [],
        "ranking_errors": [],
    }
    obj = _extract_json_blob(ranking_raw)
    if obj is None:
        result["ranking_errors"] = ["could not parse ranking JSON object"]
    else:
        normalized, errors = validate_ranking_json(obj)
        if errors:
            result["ranking_errors"] = errors
        elif normalized is not None:
            result.update(normalized)
    if result["ranking_errors"]:
        result["parse_error"] = f"ranking errors: {result['ranking_errors']}"
    result["valid"] = result["parse_error"] is None
    return result


def hypothesis_index(ordering: Sequence[str]) -> Optional[int]:
    tup = tuple(ordering)
    for idx, hyp in enumerate(HYPOTHESES):
        if tuple(hyp) == tup:
            return idx
    return None


def priorities_to_ordering(priorities: Mapping[str, str]) -> List[str]:
    return [str(priorities[level]) for level in ("High", "Medium", "Low")]


def priorities_to_points(priorities: Mapping[str, str]) -> Dict[str, int]:
    return {str(priorities[level]): PRIORITY_POINTS[level] for level in ("High", "Medium", "Low")}


def points(counts: Mapping[str, int], point_map: Mapping[str, int]) -> int:
    return sum(int(counts.get(item, 0)) * int(point_map[item]) for item in ITEMS)


def all_splits() -> List[Dict[str, Dict[str, int]]]:
    splits: List[Dict[str, Dict[str, int]]] = []
    for vals in product(range(4), repeat=3):
        self_counts = {item: int(val) for item, val in zip(ITEMS, vals)}
        opp_counts = {item: 3 - self_counts[item] for item in ITEMS}
        splits.append({"self": self_counts, "opp": opp_counts})
    return splits


def argmax_splits_for_ordering(
    *,
    my_priorities: Mapping[str, str],
    predicted_ordering: Sequence[str],
    lambda_: float = 1.0,
) -> Tuple[float, List[Dict[str, int]]]:
    """Return all self-count splits maximizing U_self + lambda * U_opp."""
    self_points = priorities_to_points(my_priorities)
    opp_points = priorities_to_points(
        {"High": predicted_ordering[0], "Medium": predicted_ordering[1], "Low": predicted_ordering[2]}
    )
    best_score = -math.inf
    best: List[Dict[str, int]] = []
    for split in all_splits():
        score = (
            points(split["self"], self_points)
            + float(lambda_) * points(split["opp"], opp_points)
        )
        if score > best_score:
            best_score = float(score)
            best = [dict(split["self"])]
        elif float(score) == best_score:
            best.append(dict(split["self"]))
    return best_score, best


def strict_action_consistent(
    *,
    offer_self_counts: Mapping[str, int],
    my_priorities: Mapping[str, str],
    predicted_ordering: Sequence[str],
    lambda_: float = 1.0,
) -> bool:
    _, best = argmax_splits_for_ordering(
        my_priorities=my_priorities,
        predicted_ordering=predicted_ordering,
        lambda_=lambda_,
    )
    offer = {item: int(offer_self_counts[item]) for item in ITEMS}
    return any(offer == split for split in best)


def loose_action_consistent(
    *,
    offer_self_counts: Mapping[str, int],
    predicted_ordering: Sequence[str],
) -> bool:
    """Direction-only consistency with the predicted opponent ranking."""
    opp_counts = {item: 3 - int(offer_self_counts[item]) for item in ITEMS}
    top, mid, low = predicted_ordering
    return (
        opp_counts[top] >= opp_counts[mid] >= opp_counts[low]
        and opp_counts[top] > opp_counts[low]
    )


def posterior_from_sample_indices(
    sample_indices: Sequence[Optional[int]],
    *,
    invalid_policy: str = "uniform",
) -> np.ndarray:
    """Convert sample hypothesis indices to a 6-way posterior."""
    n = len(HYPOTHESES)
    posterior = np.zeros(n, dtype=float)
    if not sample_indices:
        posterior[:] = 1.0 / n
        return posterior
    weight = 1.0 / len(sample_indices)
    for idx in sample_indices:
        if idx is None:
            if invalid_policy == "uniform":
                posterior += weight / n
            elif invalid_policy == "skip":
                pass
            else:
                raise ValueError(f"unknown invalid_policy {invalid_policy!r}")
        else:
            posterior[int(idx)] += weight
    s = float(posterior.sum())
    if s <= 0:
        posterior[:] = 1.0 / n
    elif abs(s - 1.0) > 1e-12:
        posterior /= s
    return posterior


def brier_from_sample_indices(
    sample_indices: Sequence[Optional[int]],
    true_index: int,
) -> float:
    posterior = posterior_from_sample_indices(sample_indices)
    return normalized_brier(posterior, int(true_index))


def format_history_for_prompt(
    history: Sequence[Mapping[str, Any]],
    *,
    my_role: str,
    opp_role: str,
    max_turns: int = 40,
) -> str:
    if not history:
        return "(conversation has not started yet)"
    lines: List[str] = []
    for turn in history[-max_turns:]:
        role = turn.get("id")
        label = "You" if role == my_role else "Opponent" if role == opp_role else str(role)
        lines.append(f"{label}: {render_turn_for_prompt(turn, my_role=my_role, opp_role=opp_role)}")
    if len(history) > max_turns:
        lines.insert(0, f"(... {len(history) - max_turns} earlier turns omitted ...)")
    return "\n".join(lines)


def render_turn_for_prompt(
    turn: Mapping[str, Any],
    *,
    my_role: str,
    opp_role: str,
) -> str:
    text = str(turn.get("text") or "").strip()
    if text != "Submit-Deal":
        return text

    td = turn.get("task_data") or {}
    speaker = turn.get("id")
    try:
        youget = {item: int((td.get("issue2youget") or {}).get(item, 0)) for item in ITEMS}
        theyget = {item: int((td.get("issue2theyget") or {}).get(item, 0)) for item in ITEMS}
    except (TypeError, ValueError):
        return "Submit-Deal"

    if speaker == my_role:
        self_counts, opp_counts = youget, theyget
        who = "You propose"
    else:
        self_counts, opp_counts = theyget, youget
        who = "Opponent proposes"
    return (
        f"Submit-Deal - {who}: you get Food={self_counts['Food']}, "
        f"Water={self_counts['Water']}, Firewood={self_counts['Firewood']}; "
        f"opponent gets Food={opp_counts['Food']}, Water={opp_counts['Water']}, "
        f"Firewood={opp_counts['Firewood']}."
    )


def build_messages(
    *,
    history: Sequence[Mapping[str, Any]],
    my_role: str,
    opp_role: str,
    my_priorities: Mapping[str, str],
    my_reasons: Mapping[str, str],
) -> List[Dict[str, str]]:
    user = USER_TEMPLATE.format(
        high_item=my_priorities.get("High", "unknown"),
        high_reason=my_reasons.get("High") or "no reason recorded",
        med_item=my_priorities.get("Medium", "unknown"),
        med_reason=my_reasons.get("Medium") or "no reason recorded",
        low_item=my_priorities.get("Low", "unknown"),
        low_reason=my_reasons.get("Low") or "no reason recorded",
        history_block=format_history_for_prompt(history, my_role=my_role, opp_role=opp_role),
    )
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user},
    ]


def build_ranking_messages(
    *,
    history: Sequence[Mapping[str, Any]],
    my_role: str,
    opp_role: str,
    my_priorities: Mapping[str, str],
    my_reasons: Mapping[str, str],
) -> List[Dict[str, str]]:
    user = RANKING_ONLY_USER_TEMPLATE.format(
        high_item=my_priorities.get("High", "unknown"),
        high_reason=my_reasons.get("High") or "no reason recorded",
        med_item=my_priorities.get("Medium", "unknown"),
        med_reason=my_reasons.get("Medium") or "no reason recorded",
        low_item=my_priorities.get("Low", "unknown"),
        low_reason=my_reasons.get("Low") or "no reason recorded",
        history_block=format_history_for_prompt(history, my_role=my_role, opp_role=opp_role),
    )
    return [
        {"role": "system", "content": RANKING_ONLY_SYSTEM_MESSAGE},
        {"role": "user", "content": user},
    ]


__all__ = [
    "DEAL_ACTIONS",
    "RANKING_ONLY_SYSTEM_MESSAGE",
    "SYSTEM_MESSAGE",
    "USER_TEMPLATE",
    "argmax_splits_for_ordering",
    "brier_from_sample_indices",
    "build_messages",
    "build_ranking_messages",
    "format_history_for_prompt",
    "hypothesis_index",
    "loose_action_consistent",
    "parse_opponent_inference_response",
    "parse_ranking_response",
    "posterior_from_sample_indices",
    "priorities_to_ordering",
    "render_turn_for_prompt",
    "strict_action_consistent",
    "validate_ranking_json",
]
