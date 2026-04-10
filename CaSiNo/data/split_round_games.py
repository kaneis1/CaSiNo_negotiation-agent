#!/usr/bin/env python3
"""Split CaSiNo data into round-level game situation files.

This utility converts the large, minified `casino.json` into easier-to-read
artifacts so each round can be inspected independently.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _pick_value(obj: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in obj:
            return obj[key]
    return default


def _dialogue_id(dialogue: Dict[str, Any], index: int) -> str:
    raw_id = _pick_value(dialogue, ("dialogue_id", "id", "chat_id", "conversation_id"))
    if raw_id is None:
        return f"dialogue_{index:04d}"
    return str(raw_id)


def _extract_rounds(dialogue: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Best-effort extraction of turn/round entries across common key names."""
    candidate = _pick_value(
        dialogue,
        ("chat_logs", "dialogue", "messages", "utterances", "events", "history", "rounds"),
        default=[],
    )
    return candidate if isinstance(candidate, list) else []


def _round_record(dialogue_id: str, round_index: int, event: Any) -> Dict[str, Any]:
    if isinstance(event, dict):
        speaker = _pick_value(event, ("speaker", "agent", "participant", "name", "id"))
        text = _pick_value(event, ("text", "utterance", "message", "content", "msg"))
        round_id = _pick_value(event, ("round", "turn", "index", "step"), default=round_index)
        return {
            "dialogue_id": dialogue_id,
            "round_index": int(round_id) if isinstance(round_id, int) else round_index,
            "speaker": speaker,
            "text": text,
            "raw": event,
        }

    return {
        "dialogue_id": dialogue_id,
        "round_index": round_index,
        "speaker": None,
        "text": str(event),
        "raw": event,
    }


def split_round_games(input_path: Path, output_dir: Path) -> None:
    with input_path.open("r", encoding="utf-8") as infile:
        data = json.load(infile)

    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array in casino.json")

    output_dir.mkdir(parents=True, exist_ok=True)
    by_dialogue_dir = output_dir / "by_dialogue"
    by_dialogue_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    flat_path = output_dir / "casino_rounds_flat.jsonl"

    with flat_path.open("w", encoding="utf-8") as flat_out:
        for idx, dialogue in enumerate(data):
            if not isinstance(dialogue, dict):
                continue

            d_id = _dialogue_id(dialogue, idx)
            rounds = _extract_rounds(dialogue)
            normalized_rounds: List[Dict[str, Any]] = []

            for r_idx, event in enumerate(rounds):
                row = _round_record(d_id, r_idx, event)
                normalized_rounds.append(row)
                flat_out.write(json.dumps(row, ensure_ascii=False) + "\n")

            grouped[d_id] = normalized_rounds

            dialogue_out = {
                "dialogue_id": d_id,
                "num_rounds": len(normalized_rounds),
                "rounds": normalized_rounds,
            }
            with (by_dialogue_dir / f"{d_id}.json").open("w", encoding="utf-8") as df:
                json.dump(dialogue_out, df, ensure_ascii=False, indent=2)

    grouped_path = output_dir / "casino_rounds_grouped.json"
    with grouped_path.open("w", encoding="utf-8") as grouped_out:
        json.dump(grouped, grouped_out, ensure_ascii=False, indent=2)

    print(f"Wrote flat rounds: {flat_path}")
    print(f"Wrote grouped rounds: {grouped_path}")
    print(f"Wrote per-dialogue files: {by_dialogue_dir}")
    print(f"Dialogues processed: {len(grouped)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split CaSiNo casino.json into round-level readable files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("CaSiNo/data/casino.json"),
        help="Path to casino.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("CaSiNo/data/split/round_games"),
        help="Directory to write split outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_round_games(args.input, args.output_dir)


if __name__ == "__main__":
    main()
