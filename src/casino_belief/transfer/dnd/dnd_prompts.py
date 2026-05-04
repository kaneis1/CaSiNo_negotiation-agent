"""Prefs-only prompts for DealOrNoDeal transfer experiments."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from casino_belief.transfer.dnd.dnd_data import (
    DNDRecord,
    DNDTurn,
    DND_ITEMS,
    item_label,
    labels_for_mode,
    map_text_names,
)


def prefs_system_prompt(*, name_mode: str) -> str:
    labels = ", ".join(labels_for_mode(name_mode))
    return (
        "You are an opponent-modeling assistant for the DealOrNoDeal "
        "negotiation game.\n"
        "Two players negotiate over three item types. You will see one "
        "player's item values and the dialogue so far, with Me for that "
        "player and Opponent for the other.\n\n"
        "Infer only the OPPONENT's hidden item-value ordering as [top, mid, low].\n"
        f"Valid items are exactly: {labels}.\n\n"
        "Reply with JSON only, no prose, in exactly this shape:\n"
        '{"prefs": ["<item>", "<item>", "<item>"]}\n\n'
        "Do not include a satisfaction field."
    )


def direct_posterior_system_prompt(*, name_mode: str) -> str:
    labels = ", ".join(labels_for_mode(name_mode))
    return (
        "You infer an opponent's hidden item-value ordering in DealOrNoDeal. "
        f"Return a calibrated probability distribution over the six orderings "
        f"of {{{labels}}}. Reply with one <posterior> block and no extra prose."
    )


def _values_block(record: DNDRecord, *, name_mode: str) -> str:
    lines = []
    for item, count, value in zip(DND_ITEMS, record.counts, record.self_values):
        label = item_label(item, name_mode=name_mode)
        lines.append(f"  {label}: count={int(count)}, value={int(value)} points each")
    return "\n".join(lines)


def _ranking_block(record: DNDRecord, *, name_mode: str) -> str:
    ordering = record.self_ordering_tiebreak
    return " > ".join(item_label(item, name_mode=name_mode) for item in ordering)


def format_history(turns: Sequence[DNDTurn], *, name_mode: str) -> str:
    lines = []
    for turn in turns:
        if turn.is_selection:
            continue
        speaker = "Me" if turn.speaker == "YOU" else "Opponent"
        text = map_text_names(turn.text, name_mode=name_mode)
        if text.strip():
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines) if lines else "(conversation not yet started)"


def build_prefs_user_prompt(
    *,
    record: DNDRecord,
    history: Sequence[DNDTurn],
    name_mode: str,
) -> str:
    return (
        "Your own item values:\n"
        f"{_values_block(record, name_mode=name_mode)}\n\n"
        "Your own value ordering:\n"
        f"  {_ranking_block(record, name_mode=name_mode)}\n\n"
        "Dialogue so far:\n"
        f"{format_history(history, name_mode=name_mode)}\n\n"
        "Predict the opponent's item-value ordering. Respond with JSON only."
    )


def build_direct_posterior_prompt(
    *,
    record: DNDRecord,
    history: Sequence[DNDTurn],
    name_mode: str,
    orderings: Sequence[Sequence[str]],
) -> str:
    candidates = "\n".join(
        f"{i + 1}. " + " > ".join(item_label(item, name_mode=name_mode) for item in ordering)
        for i, ordering in enumerate(orderings)
    )
    first = orderings[0]
    first_label = " > ".join(item_label(item, name_mode=name_mode) for item in first)
    return (
        "Your own item values:\n"
        f"{_values_block(record, name_mode=name_mode)}\n\n"
        "Your own value ordering:\n"
        f"  {_ranking_block(record, name_mode=name_mode)}\n\n"
        "Dialogue evidence so far:\n"
        f"{format_history(history, name_mode=name_mode)}\n\n"
        "Candidate opponent orderings:\n"
        f"{candidates}\n\n"
        "Output exactly:\n"
        "<posterior>\n"
        f"p({first_label})=...\n"
        "...\n"
        "</posterior>"
    )


def build_prefs_target_json(*, ordering: Sequence[str], name_mode: str) -> str:
    labels = [item_label(item, name_mode=name_mode) for item in ordering]
    return json.dumps({"prefs": labels}, ensure_ascii=False)
