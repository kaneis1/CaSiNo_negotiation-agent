#!/usr/bin/env python3
"""
SFT/prompts.py — Data loading and prompt formatting for CaSiNo SFT.

Converts casino_train.json dialogues into multi-turn chat conversations
formatted with the Llama-3.1 Instruct chat template.

Each dialogue produces two training examples (one per participant).
The system message encodes each participant's private priorities and reasons.
Loss is computed only on assistant turns via SFTConfig(assistant_only_loss=True).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from datasets import Dataset

# Action tokens that are not natural utterances — skip these turns
DEAL_ACTIONS = {"Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away"}


# ── System message builder ────────────────────────────────────────────────────

def build_system_message(
    priorities: Dict[str, str],
    reasons: Dict[str, str],
    personality: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build the system prompt for one participant.

    priorities  — {"High": "Food", "Medium": "Water", "Low": "Firewood"}
    reasons     — {"High": "...", "Medium": "...", "Low": "..."}
    personality — {"svo": "prosocial", "big-five": {...}}  (optional)
    """
    high = priorities["High"]
    med  = priorities["Medium"]
    low  = priorities["Low"]

    lines = [
        "You are a negotiation agent in the CaSiNo task.",
        "You and your conversation partner must divide 3 packages of Food, "
        "3 packages of Water, and 3 packages of Firewood between the two of you.",
        "",
        "YOUR PRIVATE PRIORITIES (keep secret — never state your point values):",
        f"  Most important  : {high} — {reasons['High']}",
        f"  Somewhat important: {med} — {reasons['Medium']}",
        f"  Least important : {low} — {reasons['Low']}",
        "",
        "GUIDELINES:",
        "- Write 1–3 natural, conversational sentences per turn.",
        "- Use casual language and contractions; occasional emojis are fine.",
        "- Never reveal your exact priority rankings or numeric point values.",
        "- Be friendly and collaborative, not aggressive.",
        "- When proposing an offer state it clearly "
        "(e.g. 'How about I get 2 food and 1 water, and you take the rest?').",
        "- Reference your personal reasons naturally rather than reciting them.",
    ]

    if personality:
        svo = personality.get("svo", "")
        bf  = personality.get("big-five", {})
        if svo:
            lines += ["", f"YOUR PERSONALITY: SVO = {svo}"]
        if bf:
            traits = ", ".join(f"{k}={v}" for k, v in bf.items())
            lines.append(f"Big-Five: {traits}")

    return "\n".join(lines)


# ── Dialogue → conversation examples ─────────────────────────────────────────

def dialogue_to_examples(dialogue: Dict[str, Any]) -> List[List[Dict[str, str]]]:
    """
    Convert one CaSiNo dialogue into up to 2 multi-turn message lists,
    one from each participant's perspective.

    Message list format:
        [{"role": "system",    "content": "..."},
         {"role": "user",      "content": "Opponent turn 1"},
         {"role": "assistant", "content": "My turn 1"},
         ...]
    """
    info      = dialogue.get("participant_info", {})
    chat_logs = dialogue.get("chat_logs", [])
    examples: List[List[Dict[str, str]]] = []

    for agent_id, participant in info.items():
        priorities  = participant["value2issue"]
        reasons     = participant["value2reason"]
        personality = participant.get("personality")

        system_msg = build_system_message(priorities, reasons, personality)
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_msg}]

        for turn in chat_logs:
            text = turn["text"].strip()
            if text in DEAL_ACTIONS:
                continue
            role = "assistant" if turn["id"] == agent_id else "user"
            messages.append({"role": role, "content": text})

        # Skip if no assistant turns to train on
        if not any(m["role"] == "assistant" for m in messages):
            continue

        # Ensure first non-system turn is from user (required by chat template)
        non_system = [m for m in messages if m["role"] != "system"]
        if non_system and non_system[0]["role"] == "assistant":
            messages.insert(1, {"role": "user", "content": "(start)"})

        examples.append(messages)

    return examples


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_casino_dataset(
    json_path: str,
    tokenizer,
    max_length: int = 2048,
) -> Dataset:
    """
    Load a CaSiNo JSON split, apply the chat template to every conversation,
    and return a HuggingFace Dataset with a single "text" column.

    Conversations that exceed max_length tokens are silently dropped.
    The ResponseOnlyCollator in collator.py handles loss masking at
    training time.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    texts: List[str] = []
    for dialogue in raw:
        for messages in dialogue_to_examples(dialogue):
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                continue
            token_len = len(
                tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
            )
            if token_len <= max_length:
                texts.append(text)

    print(f"Loaded {len(texts)} training examples from {json_path}")
    return Dataset.from_dict({"text": texts})
