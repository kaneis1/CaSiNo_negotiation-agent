"""Prompt helpers for the Day 8 distilled negotiator student.

The Day 7 corpus contains teacher posterior/menu state plus the human
target move. Day 8 trains a student that conditions on the speaker's
priorities, reasons, dialogue history, and an explicit style token, then
emits tagged intermediate quantities:

* posterior over opponent priority orderings
* next-move intent
* submitted content (or null)
* free-form utterance

The exact prompt/target strings live here so train-time data build and
later inference can share one source of truth.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional, Sequence


STUDENT_SYSTEM_PROMPT = """\
You are a CaSiNo negotiation policy model.

You will be given one speaker's perspective: a style token, that speaker's
own priorities and reasons, and the dialogue history so far.

Predict the speaker's current belief state and next move. Reply with the
four tagged fields below, in exactly this order and with no extra prose:

<posterior>
...
</posterior>
<selected_intent>
submit|accept|reject|walkaway|utter
</selected_intent>
<selected_content>
null or a JSON object
</selected_content>
<utterance>
...
</utterance>

The posterior must contain exactly six lines, one per ordering, formatted
as p(Food > Water > Firewood)=0.1234. Use JSON null for selected_content
unless the intent is submit.
"""


_TAG_RE = re.compile(
    r"<(?P<tag>[a-zA-Z0-9_]+)>\s*(?P<body>.*?)\s*</(?P=tag)>",
    re.DOTALL,
)


def extract_tagged_section(text: str, tag: str) -> str:
    pattern = re.compile(
        rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        raise ValueError(f"missing <{tag}> section")
    return match.group(1).strip()


def format_posterior(posterior: Sequence[float], orderings: Sequence[Sequence[str]]) -> str:
    return "\n".join(
        f"p({' > '.join(ordering)})={float(prob):.4f}"
        for ordering, prob in zip(orderings, posterior)
    )


def normalize_selected_content(raw: Optional[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    if raw is None:
        return None
    return {
        "self_counts": dict(raw.get("self_counts") or {}),
        "opp_counts": dict(raw.get("opp_counts") or {}),
        "self_tuple": list(raw.get("self_tuple") or []),
        "opp_tuple": list(raw.get("opp_tuple") or []),
    }


def build_student_user_prompt(
    *,
    self_priorities: str,
    self_reasons: str,
    history: str,
    style: str,
) -> str:
    return (
        "<style_token>\n"
        f"{style}\n"
        "</style_token>\n\n"
        "<self_priorities>\n"
        f"{self_priorities.strip()}\n"
        "</self_priorities>\n\n"
        "<self_reasons>\n"
        f"{self_reasons.strip()}\n"
        "</self_reasons>\n\n"
        "<history>\n"
        f"{history.strip()}\n"
        "</history>"
    )


def build_student_target(
    *,
    posterior: Sequence[float],
    orderings: Sequence[Sequence[str]],
    selected_intent: str,
    selected_content: Optional[Mapping[str, Any]],
    utterance: str,
) -> str:
    content = normalize_selected_content(selected_content)
    content_text = "null" if content is None else json.dumps(
        content,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "<posterior>\n"
        f"{format_posterior(posterior, orderings)}\n"
        "</posterior>\n"
        "<selected_intent>\n"
        f"{selected_intent}\n"
        "</selected_intent>\n"
        "<selected_content>\n"
        f"{content_text}\n"
        "</selected_content>\n"
        "<utterance>\n"
        f"{utterance}\n"
        "</utterance>"
    )


def validate_student_messages(messages: Sequence[Mapping[str, Any]]) -> None:
    if len(messages) != 3:
        raise ValueError("expected 3 chat messages")
    if messages[0].get("role") != "system":
        raise ValueError("message[0] must be system")
    if messages[1].get("role") != "user":
        raise ValueError("message[1] must be user")
    if messages[2].get("role") != "assistant":
        raise ValueError("message[2] must be assistant")
    user = str(messages[1].get("content", ""))
    assistant = str(messages[2].get("content", ""))
    for tag in ("style_token", "self_priorities", "self_reasons", "history"):
        extract_tagged_section(user, tag)
    for tag in ("posterior", "selected_intent", "selected_content", "utterance"):
        extract_tagged_section(assistant, tag)


__all__ = [
    "STUDENT_SYSTEM_PROMPT",
    "build_student_target",
    "build_student_user_prompt",
    "extract_tagged_section",
    "format_posterior",
    "normalize_selected_content",
    "validate_student_messages",
]
