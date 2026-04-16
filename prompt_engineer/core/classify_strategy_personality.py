#!/usr/bin/env python3
"""Personality-aware strategy classifier for CaSiNo utterances.

Identical interface to classify_strategy.py, but the system prompt adds
a Big-Five personality primer so the model can use trait cues — both the
speaker's known profile and inferred opponent tendencies — when assigning
strategy labels and inferring opponent item preferences.

Original (no personality): prompt_engineer/core/classify_strategy.py
This file (with personality): prompt_engineer/core/classify_strategy_personality.py
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from prompt_engineer.core.classify_strategy import (
    CASINO_ITEMS,
    VALID_LABELS,
    _build_strategy_block,
    _parse_combined_output,
    _parse_labels,
    _parse_preferences,
    classify_dialogue,       # re-exported unchanged — personality context
    format_context,          # comes from the system prompt, not the call
)

# ── Big-Five negotiation primer ────────────────────────────────────────────

BIG_FIVE_PRIMER = """\
PERSONALITY CONTEXT — Big-Five Traits in Negotiation  [scores: 1–7]
Use these as a prior when evidence in the text is ambiguous.
Explicit textual evidence always overrides personality inference.

Openness (O)
  High (≥5): creative trade-offs, hedging language ("maybe", "how about"),
             frames offers as collaborative puzzles.
  Low  (≤3): conventional splits, resists novel framings.

Conscientiousness (C)
  High (≥5): precise allocations ("that leaves you 1 food"), tracks running
             totals, logic-backed vouch-fair and promote-coordination.
  Low  (≤3): vague, may contradict earlier statements.

Extraversion (E)
  High (≥5): assertive, talkative, makes the first offer, uses emoji,
             opens with small-talk, pushes back quickly.
  Low  (≤3): brief, reserved, waits for opponent to move first.

Agreeableness (A)
  High (≥5): cooperative, conflict-averse, showing-empathy and
             promote-coordination favoured, rarely uses uv-part.
  Low  (≤3): competitive, uv-part and aggressive offers, little empathy.

Emotional Stability (ES)  [inverse of Neuroticism]
  Low  (≤3): reactive — escalates to vouch-fair or self-need under pressure,
             anxious/apologetic language, abrupt walk-away threats.
  High (≥5): calm, measured counter-offers.

SVO
  prosocial → weight on joint gains → promote-coordination, showing-empathy.
  proself   → weight on own gains  → self-need, uv-part.
"""

# ── System prompt ──────────────────────────────────────────────────────────


def build_system_prompt_with_personality() -> str:
    """System prompt: strategy taxonomy + Big-Five primer + preference role."""
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
• Use "non-strategic" when no strategy applies (never mix with other labels).
• self-need  → speaker's OWN need ("I need food").
• other-need → need argued for a THIRD PARTY ("my kids need firewood").
• self-need and other-need are mutually exclusive.
• Let the speaker's Big-Five scores below tilt ambiguous cases.

RULES FOR OPPONENT_PREFERENCES:
• Infer what the OPPONENT values most from context AND their personality.
• Assign exactly one of High / Medium / Low to each item.
• All three levels must appear exactly once (no two items share a level).
• Use Big-Five / SVO scores as a prior when the dialogue is ambiguous.

{BIG_FIVE_PRIMER}
"""


# ── Prompt builder ─────────────────────────────────────────────────────────


def _format_big_five(bf: Dict[str, float]) -> str:
    key_map = {
        "extraversion":           "E",
        "agreeableness":          "A",
        "conscientiousness":      "C",
        "emotional-stability":    "ES",
        "openness-to-experiences": "O",
    }
    parts = [
        f"{short}={bf[full]:.1f}"
        for full, short in key_map.items()
        if full in bf
    ]
    return ", ".join(parts) if parts else "N/A"


def build_classification_prompt_with_personality(
    utterance: str,
    conversation_context: str,
    speaker_big_five: Optional[Dict[str, float]] = None,
    speaker_svo: Optional[str] = None,
    opponent_big_five: Optional[Dict[str, float]] = None,
    opponent_svo: Optional[str] = None,
) -> str:
    """User-turn prompt with optional personality scores for both speakers.

    Personality scores are placed immediately before the utterance so the
    model conditions its predictions on them.

    Args:
        utterance: Text to classify.
        conversation_context: Prior turns as readable plain text.
        speaker_big_five: {"extraversion": 3.5, "agreeableness": 4.5, ...}
        speaker_svo: "prosocial" or "proself"
        opponent_big_five: Big-Five scores for the opponent (if known).
        opponent_svo: SVO for the opponent (if known).
    """
    speaker_block = ""
    if speaker_big_five or speaker_svo:
        bf_str = _format_big_five(speaker_big_five) if speaker_big_five else "N/A"
        svo_str = speaker_svo or "unknown"
        speaker_block = f"\nSPEAKER  Big-Five: {bf_str} | SVO: {svo_str}"

    opponent_block = ""
    if opponent_big_five or opponent_svo:
        bf_str = _format_big_five(opponent_big_five) if opponent_big_five else "N/A"
        svo_str = opponent_svo or "unknown"
        opponent_block = f"\nOPPONENT Big-Five: {bf_str} | SVO: {svo_str}"

    return f"""\
CONVERSATION SO FAR:
{conversation_context}
{speaker_block}{opponent_block}

UTTERANCE TO CLASSIFY:
\"{utterance}\"

Respond with exactly two lines:
STRATEGIES: <labels>
OPPONENT_PREFERENCES: Food=<High|Medium|Low>, Water=<High|Medium|Low>, Firewood=<High|Medium|Low>"""


# ── Public API ─────────────────────────────────────────────────────────────


def classify_with_preferences(
    utterance: str,
    conversation_context: str,
    llm_client: Any,
    speaker_big_five: Optional[Dict[str, float]] = None,
    speaker_svo: Optional[str] = None,
    opponent_big_five: Optional[Dict[str, float]] = None,
    opponent_svo: Optional[str] = None,
) -> Dict[str, Any]:
    """Classify strategies AND infer opponent preferences (personality-aware).

    Returns:
        {
            "strategies":           List[str],
            "opponent_preferences": {"Food": "High", "Water": "Low", ...},
            "raw":                  str,
        }
    """
    prompt = build_classification_prompt_with_personality(
        utterance, conversation_context,
        speaker_big_five, speaker_svo,
        opponent_big_five, opponent_svo,
    )
    raw = llm_client.generate(prompt)
    strategies, preferences = _parse_combined_output(raw)
    return {"strategies": strategies, "opponent_preferences": preferences, "raw": raw}


def classify_strategies(
    utterance: str,
    conversation_context: str,
    llm_client: Any,
    speaker_big_five: Optional[Dict[str, float]] = None,
    speaker_svo: Optional[str] = None,
) -> List[str]:
    """Return strategy labels only — drop-in replacement for baseline."""
    return classify_with_preferences(
        utterance, conversation_context, llm_client,
        speaker_big_five, speaker_svo,
    )["strategies"]


def classify_dialogue_with_personality(
    dialogue: Dict[str, Any],
    llm_client: Any,
    context_window: int = 5,
) -> List[Dict[str, Any]]:
    """Classify every non-deal utterance, injecting Big-Five scores from
    participant_info when available.

    Returns:
        List of dicts, one per utterance:
            {"text": str, "strategies": List[str],
             "opponent_preferences": dict, "raw": str}
    """
    deal_actions = {"Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away"}
    chat_logs = dialogue["chat_logs"]
    participant_info = dialogue.get("participant_info", {})

    # Identify player and opponent from first-appearance order
    speakers_seen: List[str] = []
    for turn in chat_logs:
        if turn["id"] not in speakers_seen:
            speakers_seen.append(turn["id"])
        if len(speakers_seen) == 2:
            break
    player_id   = speakers_seen[0] if speakers_seen else None
    opponent_id = speakers_seen[1] if len(speakers_seen) > 1 else None

    def _get_personality(agent_id: Optional[str]):
        if not agent_id:
            return None, None
        info = participant_info.get(agent_id, {})
        p = info.get("personality", {})
        return p.get("big-five"), p.get("svo")

    results: List[Dict[str, Any]] = []

    for i, turn in enumerate(chat_logs):
        if turn["text"] in deal_actions:
            continue

        speaker_id = turn["id"]
        if speaker_id == player_id:
            spk_bf, spk_svo = _get_personality(player_id)
            opp_bf, opp_svo = _get_personality(opponent_id)
        else:
            spk_bf, spk_svo = _get_personality(opponent_id)
            opp_bf, opp_svo = _get_personality(player_id)

        context = format_context(chat_logs, i, window=context_window)
        out = classify_with_preferences(
            turn["text"], context, llm_client,
            spk_bf, spk_svo, opp_bf, opp_svo,
        )
        out["text"] = turn["text"]
        results.append(out)

    return results


# ── Benchmark ──────────────────────────────────────────────────────────────


def benchmark(
    dialogues: List[Dict[str, Any]],
    llm_client: Any,
    max_dialogues: Optional[int] = None,
    context_window: int = 5,
) -> Dict[str, Any]:
    """Personality-aware benchmark — same return schema as baseline benchmark().

    Uses Big-Five and SVO from participant_info when available.
    Evaluates strategy detection (P/R/F1) and opponent preference accuracy.
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

    pref_correct_items = 0
    pref_total_items = 0
    pref_exact_dialogues = 0
    pref_evaluated = 0

    for index, dialogue in enumerate(annotated):
        gold_annotations = dialogue["annotations"]
        chat_logs = dialogue["chat_logs"]
        deal_actions = {"Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away"}
        participant_info = dialogue.get("participant_info", {})

        speakers_seen: List[str] = []
        for turn in chat_logs:
            if turn["id"] not in speakers_seen:
                speakers_seen.append(turn["id"])
            if len(speakers_seen) == 2:
                break
        player_id   = speakers_seen[0] if speakers_seen else None
        opponent_id = speakers_seen[1] if len(speakers_seen) > 1 else None

        def _get_personality(agent_id: Optional[str]):
            if not agent_id:
                return None, None
            p = participant_info.get(agent_id, {}).get("personality", {})
            return p.get("big-five"), p.get("svo")

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

            speaker_id = turn["id"]
            if speaker_id == player_id:
                spk_bf, spk_svo = _get_personality(player_id)
                opp_bf, opp_svo = _get_personality(opponent_id)
            else:
                spk_bf, spk_svo = _get_personality(opponent_id)
                opp_bf, opp_svo = _get_personality(player_id)

            out = classify_with_preferences(
                turn["text"], context, llm_client,
                spk_bf, spk_svo, opp_bf, opp_svo,
            )
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

        if opponent_id and last_pred_prefs:
            gold_v2i = (
                participant_info.get(opponent_id, {})
                .get("value2issue", {})
            )
            if gold_v2i:
                gold_prefs = {item: level for level, item in gold_v2i.items()}
                pref_evaluated += 1
                correct = sum(
                    last_pred_prefs.get(item) == gold_prefs.get(item)
                    for item in CASINO_ITEMS if item in gold_prefs
                )
                total = sum(1 for item in CASINO_ITEMS if item in gold_prefs)
                pref_correct_items += correct
                pref_total_items += total
                if correct == total:
                    pref_exact_dialogues += 1

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
