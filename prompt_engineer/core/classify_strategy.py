#!/usr/bin/env python3
"""LLM-based negotiation strategy classifier for CaSiNo utterances.

Labels match the original CaSiNo-Ann annotation scheme (10 strategies).
See: https://aclanthology.org/2021.naacl-main.254.pdf

Each utterance is classified for:
  (A) strategy labels — which of the 10 negotiation strategies are present
  (B) opponent preference estimate — inferred High/Medium/Low ranking for
      Food, Water, Firewood based on everything seen so far in the dialogue
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# ── Strategy catalogue ─────────────────────────────────────────────────────

STRATEGY_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "small-talk": {
        "definition": (
            "Off-topic chat to build rapport — weather, trip plans, personal "
            "anecdotes. Typically at conversation open/close."
        ),
        "example": "Hello! Excited for the camping trip? 🏕️",
    },
    "self-need": {
        "definition": (
            "Speaker argues they PERSONALLY need an item — own health, "
            "preference, or circumstances. The subject is 'I' / 'we' "
            "(the speaker's own group)."
        ),
        "example": "I really need food — I'll be camping for a week with no shops.",
    },
    "other-need": {
        "definition": (
            "Speaker argues for an item on behalf of SOMEONE ELSE — "
            "children, elderly relatives, pets. Subject is a third party."
        ),
        "example": "My kids get cold at night, so we really need firewood.",
    },
    "elicit-pref": {
        "definition": (
            "Directly asks what the opponent wants, needs, or values most/least. "
            "Intent is to learn the opponent's preference order."
        ),
        "example": "What item is your top priority?",
    },
    "no-need": {
        "definition": (
            "Speaker says they do NOT need a specific item, signalling it is "
            "available for the opponent. Often implies a trade."
        ),
        "example": "We brought plenty of water already, so we can do without.",
    },
    "promote-coordination": {
        "definition": (
            "Explicit push toward a mutually beneficial deal — proposing a "
            "split, calling for collaboration, or offering a concrete trade."
        ),
        "example": "How about I take 2 food and you take 3 firewood and 2 water?",
    },
    "vouch-fair": {
        "definition": (
            "Invokes fairness norms — calling the current proposal unfair, "
            "requesting a more balanced split, or vouching for equity."
        ),
        "example": "That leaves me with nothing — we need a more balanced split.",
    },
    "uv-part": {
        "definition": (
            "Undervalue-Partner: questions or undermines the opponent's stated "
            "need for an item ('you don't really need that much')."
        ),
        "example": "Do you really need 3 packs of water? There's a stream nearby.",
    },
    "showing-empathy": {
        "definition": (
            "Positive acknowledgment of the opponent's situation — expressing "
            "understanding, compassion, or solidarity."
        ),
        "example": "I totally understand — keeping your family comfortable matters.",
    },
    "non-strategic": {
        "definition": (
            "No negotiation strategy is evident. Filler, acknowledgment, "
            "or clarification with no strategic function."
        ),
        "example": "Oh, I see. Sorry for the confusion!",
    },
}

VALID_LABELS = set(STRATEGY_DEFINITIONS.keys())

CASINO_ITEMS = ("Food", "Water", "Firewood")

# ── Prompt builders ────────────────────────────────────────────────────────


def _build_strategy_block() -> str:
    """Compact numbered list of strategy definitions + examples."""
    lines: List[str] = []
    for i, (name, info) in enumerate(STRATEGY_DEFINITIONS.items(), 1):
        lines.append(f"{i:>2}. {name}")
        lines.append(f"    Def : {info['definition']}")
        lines.append(f"    Ex  : \"{info['example']}\"")
    return "\n".join(lines)


def build_system_prompt() -> str:
    """System prompt: strategy taxonomy + preference inference role."""
    label_list = " | ".join(VALID_LABELS)
    return f"""\
You are an expert annotator for the CaSiNo negotiation dataset.
Two campers divide 9 packages (3 Food + 3 Water + 3 Firewood).

For each utterance you will output EXACTLY two lines — nothing else.

OUTPUT FORMAT:
STRATEGIES: <comma-separated labels>
OPPONENT_PREFERENCES: Food=<High|Medium|Low>, Water=<High|Medium|Low>, Firewood=<High|Medium|Low>

STRATEGY LABELS (valid values for STRATEGIES line):
{label_list}

STRATEGY DEFINITIONS:
{_build_strategy_block()}

RULES FOR STRATEGIES:
• List every applicable label; separate with ", ".
• Use "non-strategic" when no strategy applies (do not mix it with others).
• self-need  → speaker's OWN need ("I need food").
• other-need → need argued for a THIRD PARTY ("my kids need firewood").
• These two labels are mutually exclusive — never combine them.

RULES FOR OPPONENT_PREFERENCES:
• Infer what the opponent values most based on ALL context provided.
• Assign exactly one of High / Medium / Low to each item.
• Each level must be used exactly once (one High, one Medium, one Low).
• If truly uncertain, make your best guess — never leave a level blank.
"""


def build_classification_prompt(
    utterance: str,
    conversation_context: str,
) -> str:
    """User-turn prompt for a single utterance.

    Args:
        utterance: The text to classify.
        conversation_context: Prior turns as readable plain text.

    Returns:
        Prompt string ready for llm_client.generate().
    """
    return f"""\
CONVERSATION SO FAR:
{conversation_context}

UTTERANCE TO CLASSIFY:
\"{utterance}\"

Respond with exactly two lines:
STRATEGIES: <labels>
OPPONENT_PREFERENCES: Food=<High|Medium|Low>, Water=<High|Medium|Low>, Firewood=<High|Medium|Low>"""


# ── Output parsers ─────────────────────────────────────────────────────────


def _parse_labels(raw: str) -> List[str]:
    """Parse a comma-separated label string into a validated list."""
    raw = raw.strip().strip('"\'').strip()
    raw = re.sub(r"[\[\]]", "", raw)
    candidates = [c.strip().lower() for c in raw.split(",")]
    labels = [c for c in candidates if c in VALID_LABELS]
    return labels if labels else ["non-strategic"]


def _parse_preferences(raw: str) -> Dict[str, str]:
    """Parse 'Food=High, Water=Low, Firewood=Medium' into a dict."""
    result: Dict[str, str] = {item: "Medium" for item in CASINO_ITEMS}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if "=" in chunk:
            key, _, val = chunk.partition("=")
            key = key.strip().title()
            val = val.strip().title()
            if key in result and val in ("High", "Medium", "Low"):
                result[key] = val
    return result


def _parse_combined_output(raw: str) -> Tuple[List[str], Dict[str, str]]:
    """Parse the two-line LLM response into (strategies, preferences)."""
    strategies: List[str] = ["non-strategic"]
    preferences: Dict[str, str] = {item: "Medium" for item in CASINO_ITEMS}

    for line in raw.splitlines():
        line = line.strip()
        upper = line.upper()

        if upper.startswith("STRATEGIES:"):
            strategies = _parse_labels(line.split(":", 1)[1])

        elif upper.startswith("OPPONENT_PREFERENCES:"):
            preferences = _parse_preferences(line.split(":", 1)[1])

    return strategies, preferences


# ── Public API ─────────────────────────────────────────────────────────────


def classify_with_preferences(
    utterance: str,
    conversation_context: str,
    llm_client: Any,
) -> Dict[str, Any]:
    """Classify strategies AND infer opponent preferences in one LLM call.

    Returns:
        {
            "strategies":           List[str],
            "opponent_preferences": {"Food": "High", "Water": "Low", ...},
            "raw":                  str,   # raw model output for debugging
        }
    """
    prompt = build_classification_prompt(utterance, conversation_context)
    raw = llm_client.generate(prompt)
    strategies, preferences = _parse_combined_output(raw)
    return {"strategies": strategies, "opponent_preferences": preferences, "raw": raw}


def classify_strategies(
    utterance: str,
    conversation_context: str,
    llm_client: Any,
) -> List[str]:
    """Return strategy labels only — backward-compatible wrapper."""
    return classify_with_preferences(utterance, conversation_context, llm_client)[
        "strategies"
    ]


# ── Context formatting helper ──────────────────────────────────────────────


def format_context(
    chat_logs: List[Dict[str, Any]],
    current_index: int,
    window: int = 5,
) -> str:
    """Format preceding turns as readable context.

    Args:
        chat_logs: Full chat_logs list from a CaSiNo dialogue.
        current_index: Index of the utterance being classified.
        window: Number of preceding turns to include.
    """
    deal_actions = {"Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away"}
    start = max(0, current_index - window)
    lines: List[str] = []
    for turn in chat_logs[start:current_index]:
        if turn["text"] in deal_actions:
            continue
        speaker = turn["id"].replace("mturk_", "").replace("_", " ").title()
        lines.append(f"{speaker}: {turn['text']}")
    return "\n".join(lines) if lines else "(conversation start)"


# ── Batch classify a full dialogue ─────────────────────────────────────────


def classify_dialogue(
    dialogue: Dict[str, Any],
    llm_client: Any,
    context_window: int = 5,
) -> List[Dict[str, Any]]:
    """Classify every non-deal utterance in a dialogue.

    Returns:
        List of dicts, one per utterance:
            {"text": str, "strategies": List[str],
             "opponent_preferences": dict, "raw": str}
    """
    deal_actions = {"Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away"}
    chat_logs = dialogue["chat_logs"]
    results: List[Dict[str, Any]] = []

    for i, turn in enumerate(chat_logs):
        if turn["text"] in deal_actions:
            continue
        context = format_context(chat_logs, i, window=context_window)
        out = classify_with_preferences(turn["text"], context, llm_client)
        out["text"] = turn["text"]
        results.append(out)

    return results


# ── Benchmark against gold annotations ─────────────────────────────────────


def benchmark(
    dialogues: List[Dict[str, Any]],
    llm_client: Any,
    max_dialogues: Optional[int] = None,
    context_window: int = 5,
) -> Dict[str, Any]:
    """Evaluate classifier against CaSiNo-Ann gold labels.

    Evaluates both strategy detection (P/R/F1) and opponent preference
    inference accuracy (fraction of items ranked correctly per dialogue).

    Returns per-label metrics, macro/micro averages, and preference accuracy.
    """
    annotated = [d for d in dialogues if d.get("annotations")]
    if max_dialogues:
        annotated = annotated[:max_dialogues]

    tp: Dict[str, int] = {l: 0 for l in VALID_LABELS}
    fp: Dict[str, int] = {l: 0 for l in VALID_LABELS}
    fn: Dict[str, int] = {l: 0 for l in VALID_LABELS}
    total_correct = 0
    total_predicted = 0
    total_gold = 0
    n_utterances = 0

    pref_correct_items = 0   # correctly ranked items across all dialogues
    pref_total_items = 0     # total ranked items across all dialogues
    pref_exact_dialogues = 0 # dialogues where all 3 items ranked correctly
    pref_evaluated = 0       # dialogues where we have gold preferences

    for index, dialogue in enumerate(annotated):
        gold_annotations = dialogue["annotations"]
        chat_logs = dialogue["chat_logs"]
        deal_actions = {"Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away"}

        # Identify the two speakers in order of first appearance
        speakers_seen: List[str] = []
        for turn in chat_logs:
            if turn["id"] not in speakers_seen:
                speakers_seen.append(turn["id"])
            if len(speakers_seen) == 2:
                break
        opponent_id = speakers_seen[1] if len(speakers_seen) > 1 else None

        print(f"Dialogue {index + 1}/{len(annotated)}")

        utt_idx = 0
        last_pred_prefs: Optional[Dict[str, str]] = None

        for i, turn in enumerate(chat_logs):
            if turn["text"] in deal_actions:
                continue
            if utt_idx >= len(gold_annotations):
                break

            gold_text, gold_str = gold_annotations[utt_idx]
            utt_idx += 1

            gt_labels = {s.strip() for s in gold_str.split(",") if s.strip()}
            context = format_context(chat_logs, i, window=context_window)
            out = classify_with_preferences(turn["text"], context, llm_client)

            pred_set = set(out["strategies"])
            last_pred_prefs = out["opponent_preferences"]

            for label in VALID_LABELS:
                if label in gt_labels and label in pred_set:
                    tp[label] += 1
                elif label in pred_set and label not in gt_labels:
                    fp[label] += 1
                elif label in gt_labels and label not in pred_set:
                    fn[label] += 1

            total_correct += len(pred_set & gt_labels)
            total_predicted += len(pred_set)
            total_gold += len(gt_labels)
            n_utterances += 1

        # Evaluate preference inference using the last utterance's prediction
        if opponent_id and last_pred_prefs:
            gold_v2i = (
                dialogue.get("participant_info", {})
                .get(opponent_id, {})
                .get("value2issue", {})
            )
            if gold_v2i:
                # gold_v2i = {"High": "Food", ...} → {"Food": "High", ...}
                gold_prefs = {item: level for level, item in gold_v2i.items()}
                pref_evaluated += 1
                correct = sum(
                    last_pred_prefs.get(item) == gold_prefs.get(item)
                    for item in CASINO_ITEMS
                    if item in gold_prefs
                )
                total = sum(1 for item in CASINO_ITEMS if item in gold_prefs)
                pref_correct_items += correct
                pref_total_items += total
                if correct == total:
                    pref_exact_dialogues += 1

    # Per-label metrics
    per_label: Dict[str, Dict[str, float]] = {}
    for label in VALID_LABELS:
        p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
        r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_label[label] = {
            "precision": p, "recall": r, "f1": f1,
            "support": tp[label] + fn[label],
        }

    macro_p = sum(v["precision"] for v in per_label.values()) / len(per_label)
    macro_r = sum(v["recall"] for v in per_label.values()) / len(per_label)
    macro_f1 = sum(v["f1"] for v in per_label.values()) / len(per_label)

    micro_p = total_correct / total_predicted if total_predicted else 0.0
    micro_r = total_correct / total_gold if total_gold else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    return {
        "num_dialogues": len(annotated),
        "num_utterances": n_utterances,
        "per_label": per_label,
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "preference": {
            "item_accuracy": pref_correct_items / pref_total_items if pref_total_items else None,
            "exact_match_rate": pref_exact_dialogues / pref_evaluated if pref_evaluated else None,
            "dialogues_evaluated": pref_evaluated,
        },
    }
