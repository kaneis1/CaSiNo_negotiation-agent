"""Parser + validator for the Structured-CoT response format.

The LLM returns five XML-tagged blocks plus a JSON decision. The parser
is deliberately defensive — it never raises; on any malformed input it
returns a dict with a populated ``parse_error`` field so the caller can
decide to retry or fall back.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("structured_cot.parser")

ITEMS = ("Food", "Water", "Firewood")
VALID_ACTIONS = frozenset({"accept", "reject", "walkaway"})

# One regex per tag. We use a non-greedy capture and DOTALL so newlines
# inside a block are matched. The tags are NOT nested, so a simple
# non-greedy match is sufficient.
_TAG_PATTERNS: Dict[str, re.Pattern] = {
    name: re.compile(rf"<{name}\s*>(.*?)</{name}\s*>", re.DOTALL | re.IGNORECASE)
    for name in ("observation", "opponent_inference", "plan", "utterance", "decision")
}

_JSON_CODEFENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

REQUIRED_TAGS = ("observation", "opponent_inference", "plan", "utterance", "decision")

REMINDER = """

## Reminder before you retry

Your previous response could not be parsed. Please:
- Emit exactly one of each tag, in this order: <observation>,
  <opponent_inference>, <plan>, <utterance>, <decision>.
- The <decision> block must contain valid JSON only, nothing else.
- counter_offer values must be integers in 0..3 (or counter_offer: null).
- Stop immediately after </decision>.
"""


def _extract_tag(text: str, name: str) -> Optional[str]:
    m = _TAG_PATTERNS[name].search(text or "")
    if not m:
        return None
    return m.group(1).strip()


def _extract_json_blob(decision_block: str) -> Optional[Dict[str, Any]]:
    """Pull the JSON object out of a <decision> body.

    Accepts:
      * raw JSON object
      * ```json ... ``` fenced JSON
      * a JSON object embedded in prose (picks the first balanced {...})
    """
    text = (decision_block or "").strip()
    if not text:
        return None

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    m = _JSON_CODEFENCE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
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
                blob = text[start : i + 1]
                try:
                    obj = json.loads(blob)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    return None
                break
    return None


def validate_decision(decision: Dict[str, Any]) -> List[str]:
    """Return a list of validation error strings. Empty list = valid.

    Rules mirror the system prompt:
      * ``action`` must be "accept" | "reject" | "walkaway".
      * ``counter_offer`` must be null *except* when ``action == "reject"``,
        in which case it can be null (conversational) or a three-item dict
        with integer counts in 0..3.
      * "accept" / "walkaway" MUST have counter_offer == null.
    """
    errs: List[str] = []
    if not isinstance(decision, dict):
        errs.append("decision is not a JSON object")
        return errs

    action = decision.get("action")
    if action not in VALID_ACTIONS:
        errs.append(
            f"action must be one of {sorted(VALID_ACTIONS)}, got {action!r}"
        )

    co = decision.get("counter_offer", None)
    if action in ("accept", "walkaway"):
        if co is not None:
            errs.append(
                f"counter_offer must be null when action is {action!r}, got {co!r}"
            )
    elif action == "reject":
        if co is None:
            pass  # conversational reject (no proposal yet)
        elif not isinstance(co, dict):
            errs.append(f"counter_offer must be a JSON object or null, got {type(co).__name__}")
        else:
            missing = [it for it in ITEMS if it not in co]
            if missing:
                errs.append(f"counter_offer missing keys {missing}")
            extras = [k for k in co if k not in ITEMS]
            if extras:
                errs.append(f"counter_offer has unexpected keys {extras}")
            for it in ITEMS:
                if it not in co:
                    continue
                v = co[it]
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    errs.append(f"counter_offer[{it!r}] must be integer, got {v!r}")
                    continue
                if float(v) != int(v):
                    errs.append(f"counter_offer[{it!r}] must be integer, got {v!r}")
                    continue
                iv = int(v)
                if iv < 0 or iv > 3:
                    errs.append(f"counter_offer[{it!r}] = {iv} out of range [0, 3]")
    return errs


def normalize_decision(decision: Dict[str, Any]) -> Dict[str, Any]:
    """Post-parse coercion: ints, lower-cased action, strip null counter_offer."""
    out: Dict[str, Any] = {}
    out["action"] = (
        str(decision.get("action", "")).strip().lower()
        if decision.get("action") is not None else None
    )
    co = decision.get("counter_offer", None)
    if isinstance(co, dict):
        normalized = {}
        for it in ITEMS:
            v = co.get(it)
            if v is None:
                normalized[it] = None
            else:
                try:
                    normalized[it] = int(v)
                except (TypeError, ValueError):
                    normalized[it] = v
        out["counter_offer"] = normalized
    else:
        out["counter_offer"] = None
    return out


def parse_response(llm_output: str) -> Dict[str, Any]:
    """Extract the five tagged blocks + validated decision.

    Returns a dict with these keys (always present):
        observation        (str or None)
        opponent_inference (str or None)
        plan               (str or None)
        utterance          (str or None)
        decision           (dict or None)        <- normalized, not raw
        decision_raw       (str or None)         <- the <decision> body as emitted
        parse_error        (str or None)         <- None on success
        missing_tags       (list[str])           <- any tags that weren't found
        decision_errors    (list[str])           <- validation errors on the JSON

    ``parse_error`` is set iff at least one of: a required tag is missing,
    the decision JSON is unparseable, or validation errors were found.
    """
    result: Dict[str, Any] = {
        "observation":        None,
        "opponent_inference": None,
        "plan":               None,
        "utterance":          None,
        "decision":           None,
        "decision_raw":       None,
        "parse_error":        None,
        "missing_tags":       [],
        "decision_errors":    [],
    }

    for tag in REQUIRED_TAGS:
        result[tag if tag != "decision" else "decision_raw"] = _extract_tag(
            llm_output or "", tag,
        )

    missing = [t for t in REQUIRED_TAGS if not result[t if t != "decision" else "decision_raw"]]
    result["missing_tags"] = missing

    decision_obj: Optional[Dict[str, Any]] = None
    if result["decision_raw"]:
        decision_obj = _extract_json_blob(result["decision_raw"])
        if decision_obj is None:
            result["decision_errors"].append(
                "could not extract JSON object from <decision> block"
            )
        else:
            errs = validate_decision(decision_obj)
            if errs:
                result["decision_errors"].extend(errs)
            result["decision"] = normalize_decision(decision_obj)

    if missing or result["decision_errors"]:
        parts: List[str] = []
        if missing:
            parts.append(f"missing tags: {missing}")
        if result["decision_errors"]:
            parts.append(f"decision errors: {result['decision_errors']}")
        result["parse_error"] = "; ".join(parts)

    return result


def safe_default(
    pending_offer: bool = False,
    reason: str = "parse failure after retry",
) -> Dict[str, Any]:
    """Fallback action when we can't get a parseable response.

    Per the spec: "reject current offer, propose equal split". The equal
    split is ``{Food:1, Water:1, Firewood:1}`` (agent gets 1 of each,
    opponent gets 2 of each — a conservative opening proposal that won't
    be mistaken for an acceptance).

    If there is no pending offer, we still emit an equal-split proposal
    so the simulation keeps moving rather than stalling.
    """
    return {
        "observation":        "(fallback) parser failure — safe default engaged.",
        "opponent_inference": "(fallback) reasoning unavailable; no belief update.",
        "plan":               "(fallback) reject any pending offer, propose equal split.",
        "utterance":          "Let me suggest an even split: 1 Food, 1 Water, 1 Firewood each. Does that work?",
        "decision": {
            "action": "reject",
            "counter_offer": {"Food": 1, "Water": 1, "Firewood": 1},
        },
        "decision_raw":      None,
        "parse_error":       None,
        "missing_tags":      [],
        "decision_errors":   [],
        "fallback":          True,
        "fallback_reason":   reason,
    }


__all__ = [
    "ITEMS",
    "VALID_ACTIONS",
    "REQUIRED_TAGS",
    "REMINDER",
    "parse_response",
    "validate_decision",
    "normalize_decision",
    "safe_default",
]
