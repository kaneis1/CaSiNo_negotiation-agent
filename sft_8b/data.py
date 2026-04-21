"""Build SFT training rows from CaSiNo dialogues.

For each (dialogue x perspective x k=1..max_k opponent utterance), emit
one chat-format row whose target is the JSON containing the opponent's
true preference ordering and the speaker's true satisfaction label.

This snapshot scheme MIRRORS exactly the eval loop in
``opponent_model/metrics.py`` (lines 134-189) so the SFT distribution
matches the eval distribution at every k.

Output schema (one JSON object per line):

    {
      "messages": [
        {"role": "system",    "content": <SYSTEM_PROMPT>},
        {"role": "user",      "content": <user prompt with prefix>},
        {"role": "assistant", "content": <ground-truth JSON target>}
      ],
      "dialogue_id": int,
      "perspective": "mturk_agent_1",
      "opp_role":    "mturk_agent_2",
      "k":           int,                # 1..max_k
      "true_prefs":  ["Firewood", "Food", "Water"],
      "true_satisfaction": "Extremely satisfied"
    }

Skip rules
----------
A row is dropped (logged once at the end with a count) if:
  * The opponent has fewer than 1 utterance in the dialogue (no signal).
  * ``participant_info[<role>]["value2issue"]`` is missing/malformed.
  * ``participant_info[<role>]["outcomes"]["satisfaction"]`` is missing
    or not in :data:`SATISFACTION_LABELS`.

Usage
-----
    # smoke test (5 dialogues from train split)
    python -m sft_8b.data --split train --max-dialogues 5 --output /tmp/x.jsonl

    # full build of all three splits (default)
    python -m sft_8b.data
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sft_8b.prompts import (
    DEAL_ACTIONS,
    SATISFACTION_LABELS,
    SYSTEM_PROMPT,
    build_target_json,
    build_user_prompt,
)

logger = logging.getLogger("sft_8b.data")


DEFAULT_SPLIT_DIR = Path("CaSiNo/data/split")
DEFAULT_OUTPUT_DIR = Path("sft_8b/results/sft_data")
DEFAULT_MAX_K = 5

ROLES: Tuple[str, str] = ("mturk_agent_1", "mturk_agent_2")


# ── Per-dialogue snapshot builder ──────────────────────────────────────────


def _safe_satisfaction(pinfo_role: Mapping[str, Any]) -> Optional[str]:
    sat = pinfo_role.get("outcomes", {}).get("satisfaction")
    if isinstance(sat, str) and sat in SATISFACTION_LABELS:
        return sat
    return None


def _safe_value2issue(pinfo_role: Mapping[str, Any]) -> Optional[Dict[str, str]]:
    v2i = pinfo_role.get("value2issue")
    if not isinstance(v2i, Mapping):
        return None
    if set(v2i.keys()) != {"High", "Medium", "Low"}:
        return None
    if set(v2i.values()) != {"Food", "Water", "Firewood"}:
        return None
    return dict(v2i)


def build_dialogue_rows(
    dialogue: Mapping[str, Any],
    *,
    max_k: int = DEFAULT_MAX_K,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Yield SFT rows for both perspectives of one dialogue.

    Returns ``(rows, skip_reasons)`` so callers can keep aggregate counts.
    """
    rows: List[Dict[str, Any]] = []
    skips: List[str] = []

    chat_logs = dialogue.get("chat_logs") or []
    pinfo = dialogue.get("participant_info") or {}
    dialogue_id = dialogue.get("dialogue_id")

    for me_role in ROLES:
        opp_role = ROLES[1] if me_role == ROLES[0] else ROLES[0]

        my_v2i = _safe_value2issue(pinfo.get(me_role, {}))
        opp_v2i = _safe_value2issue(pinfo.get(opp_role, {}))
        my_sat = _safe_satisfaction(pinfo.get(me_role, {}))

        if my_v2i is None or opp_v2i is None:
            skips.append("missing_or_bad_value2issue")
            continue
        if my_sat is None:
            skips.append("missing_or_bad_satisfaction")
            continue

        my_reasons = pinfo[me_role].get("value2reason", {}) or {}
        target_json = build_target_json(
            opp_value2issue=opp_v2i, my_satisfaction=my_sat,
        )
        true_prefs = [opp_v2i["High"], opp_v2i["Medium"], opp_v2i["Low"]]

        opp_count = 0
        partial: List[Dict[str, Any]] = []

        for turn in chat_logs:
            text = turn.get("text", "")
            if not text or text in DEAL_ACTIONS:
                continue
            partial.append(turn)
            if turn.get("id") != opp_role:
                continue

            opp_count += 1
            if opp_count > max_k:
                break

            user_prompt = build_user_prompt(
                partial=partial,
                my_priorities=my_v2i,
                my_reasons=my_reasons,
                me_role=me_role,
            )
            rows.append({
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": user_prompt},
                    {"role": "assistant", "content": target_json},
                ],
                "dialogue_id":      dialogue_id,
                "perspective":      me_role,
                "opp_role":         opp_role,
                "k":                opp_count,
                "true_prefs":       true_prefs,
                "true_satisfaction": my_sat,
            })

        if opp_count == 0:
            skips.append("no_opponent_utterances")

    return rows, skips


# ── Split-level driver ─────────────────────────────────────────────────────


def build_split(
    *,
    input_path: Path,
    output_path: Path,
    max_k: int = DEFAULT_MAX_K,
    max_dialogues: Optional[int] = None,
) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Building SFT rows: %s -> %s", input_path, output_path)

    with input_path.open() as f:
        dialogues = json.load(f)
    if max_dialogues is not None:
        dialogues = dialogues[:max_dialogues]

    n_rows = 0
    n_dialogues_with_rows = 0
    skip_counter: Counter[str] = Counter()

    with output_path.open("w") as out:
        for dialogue in dialogues:
            rows, skips = build_dialogue_rows(dialogue, max_k=max_k)
            for r in rows:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_rows += len(rows)
            if rows:
                n_dialogues_with_rows += 1
            skip_counter.update(skips)

    stats = {
        "input_path":            str(input_path),
        "output_path":           str(output_path),
        "n_dialogues_input":     len(dialogues),
        "n_dialogues_with_rows": n_dialogues_with_rows,
        "n_rows":                n_rows,
        "max_k":                 max_k,
        "skips":                 dict(skip_counter),
    }
    logger.info(
        "  done: %d dialogues -> %d SFT rows (skips=%s)",
        len(dialogues), n_rows, dict(skip_counter) or "{}",
    )
    return stats


# ── CLI ────────────────────────────────────────────────────────────────────


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--split-dir", type=Path, default=DEFAULT_SPLIT_DIR,
        help="Directory containing casino_{train,valid,test}.json.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory to write sft_{train,valid,test}_rows.jsonl into.",
    )
    p.add_argument(
        "--split", choices=("train", "valid", "test", "all"), default="all",
        help="Which split(s) to build. Default: all three.",
    )
    p.add_argument("--max-k", type=int, default=DEFAULT_MAX_K)
    p.add_argument(
        "--max-dialogues", type=int, default=None,
        help="Cap dialogues per split (smoke test).",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Override the per-split output file (only valid with one --split).",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_argparser().parse_args(argv)

    splits = ("train", "valid", "test") if args.split == "all" else (args.split,)
    if args.output is not None and len(splits) != 1:
        print("--output only valid when --split is a single split", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_stats: List[Dict[str, Any]] = []
    for split in splits:
        in_path = args.split_dir / f"casino_{split}.json"
        out_path = args.output if args.output is not None else (
            args.output_dir / f"sft_{split}_rows.jsonl"
        )
        stats = build_split(
            input_path=in_path,
            output_path=out_path,
            max_k=args.max_k,
            max_dialogues=args.max_dialogues,
        )
        all_stats.append(stats)

    summary_path = args.output_dir / "data_build_summary.json"
    with summary_path.open("w") as f:
        json.dump(all_stats, f, indent=2)
    logger.info("Wrote build summary -> %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
