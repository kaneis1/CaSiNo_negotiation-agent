"""Build Day 8 SFT rows for the distilled CaSiNo negotiator.

Inputs:
    * Day 7 distillation corpus (teacher posterior + human move)

Outputs:
    * train/eval chat rows for `sft_8b.train`
    * config + summary with deterministic split and oversampling metadata

Design choices:
    * style token is prompt conditioning
    * assistant target emits posterior + intent + content + utterance
    * submit-response side fields stay in metadata only (not prompt)
    * intent imbalance is handled by train-time row oversampling, with
      utter untouched and minority non-utter intents repeated up to the
      largest non-utter class
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sft_8b.build_distill_data import ORDERINGS, parse_styles
from sft_8b.student_prompts import (
    STUDENT_SYSTEM_PROMPT,
    build_student_target,
    build_student_user_prompt,
    extract_tagged_section,
    validate_student_messages,
)

logger = logging.getLogger("sft_8b.build_day8_sft_data")


DEFAULT_SOURCE_JSONL = Path("sft_8b/results/distill/day7/day7_distill.jsonl")
DEFAULT_OUTPUT_DIR = Path("sft_8b/results/day8_sft_data")
DEFAULT_EVAL_FRACTION = 0.10
DEFAULT_SEED = 42
DEFAULT_MAX_INTENT_REPEAT = 32


def stable_eval_split(dialogue_id: Any, *, seed: int, eval_fraction: float) -> str:
    key = f"{seed}:{dialogue_id}".encode("utf-8")
    bucket = int(hashlib.sha1(key).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if bucket < eval_fraction else "train"


def row_to_student_messages(row: Mapping[str, Any]) -> List[Dict[str, str]]:
    source_user = str(row["messages"][1]["content"])
    user_prompt = build_student_user_prompt(
        self_priorities=extract_tagged_section(source_user, "self_priorities"),
        self_reasons=extract_tagged_section(source_user, "self_reasons"),
        history=extract_tagged_section(source_user, "history"),
        style=str(row["style"]),
    )
    target = row["target"]
    assistant = build_student_target(
        posterior=row["posterior"],
        orderings=ORDERINGS,
        selected_intent=str(target["selected_intent"]),
        selected_content=target.get("selected_content"),
        utterance=str(target.get("utterance", "")),
    )
    messages = [
        {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant},
    ]
    validate_student_messages(messages)
    return messages


def load_source_rows(path: Path, *, styles: Sequence[str]) -> List[Dict[str, Any]]:
    styles_set = set(styles)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("style") not in styles_set:
                continue
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no Day 7 rows found in {path} for styles={styles}")
    return rows


def compute_repeat_map(
    counts: Mapping[str, int],
    *,
    mode: str,
    max_repeat: int,
) -> Dict[str, int]:
    intents = {str(k): int(v) for k, v in counts.items()}
    repeat = {intent: 1 for intent in intents}
    if mode == "none":
        return repeat

    non_utter = {
        intent: count for intent, count in intents.items()
        if intent != "utter" and count > 0
    }
    if not non_utter:
        return repeat

    anchor = max(non_utter.values())
    for intent, count in non_utter.items():
        repeat[intent] = min(max_repeat, max(1, int(math.ceil(anchor / count))))
    repeat["utter"] = 1
    return repeat


def expanded_rows(
    base_rows: Sequence[Dict[str, Any]],
    repeat_map: Mapping[str, int],
) -> Iterable[Dict[str, Any]]:
    for row in base_rows:
        intent = row["selected_intent"]
        repeat = int(repeat_map.get(intent, 1))
        for repeat_index in range(repeat):
            out = dict(row)
            out["repeat_factor"] = repeat
            out["repeat_index"] = repeat_index
            yield out


def make_student_row(row: Mapping[str, Any], *, split: str, source_index: int) -> Dict[str, Any]:
    messages = row_to_student_messages(row)
    return {
        "messages": messages,
        "split": split,
        "source_index": source_index,
        "dialogue_id": row["dialogue_id"],
        "perspective": row["perspective"],
        "style": row["style"],
        "selected_intent": row["target"]["selected_intent"],
        "selected_content": row["target"]["selected_content"],
        "n_opp_utterances_seen": row["n_opp_utterances_seen"],
        "submit_is_response_to_opp_offer": row.get("submit_is_response_to_opp_offer"),
        "max_prior_opp_submit_age_turns": row.get("max_prior_opp_submit_age_turns"),
        "posterior_entropy_bits": row.get("posterior_entropy_bits"),
        "source_model_commit": row.get("model_commit"),
    }


def build_config(args: argparse.Namespace, *, source_rows: int) -> Dict[str, Any]:
    styles = parse_styles(args.styles)
    return {
        "source_jsonl": str(args.input_jsonl),
        "source_rows_after_style_filter": source_rows,
        "styles": styles,
        "eval_fraction": args.eval_fraction,
        "split_seed": args.seed,
        "prompt_schema": "style_token + self_priorities + self_reasons + history",
        "target_schema": "posterior + selected_intent + selected_content + utterance",
        "submit_response_feature_usage": "analysis_only",
        "intent_balance_mode": args.intent_balance_mode,
        "max_intent_repeat": args.max_intent_repeat,
        "assistant_format": "tagged_text",
        "system_prompt_version": "day8_student_v1",
    }


def process(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    styles = parse_styles(args.styles)
    source_rows = load_source_rows(Path(args.input_jsonl), styles=styles)
    config = build_config(args, source_rows=len(source_rows))
    config_path = output_dir / "day8_data_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    train_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    pre_counts = {"train": Counter(), "eval": Counter()}
    pre_style_counts = {"train": Counter(), "eval": Counter()}
    split_dialogues = {"train": set(), "eval": set()}

    for idx, row in enumerate(source_rows):
        split = stable_eval_split(
            row["dialogue_id"],
            seed=args.seed,
            eval_fraction=args.eval_fraction,
        )
        student_row = make_student_row(row, split=split, source_index=idx)
        if split == "train":
            train_rows.append(student_row)
        else:
            eval_rows.append(student_row)
        pre_counts[split][student_row["selected_intent"]] += 1
        pre_style_counts[split][student_row["style"]] += 1
        split_dialogues[split].add(student_row["dialogue_id"])

    repeat_map = compute_repeat_map(
        pre_counts["train"],
        mode=args.intent_balance_mode,
        max_repeat=args.max_intent_repeat,
    )
    train_rows_expanded = list(expanded_rows(train_rows, repeat_map))
    post_train_counts = Counter(r["selected_intent"] for r in train_rows_expanded)

    train_path = output_dir / "day8_train_rows.jsonl"
    eval_path = output_dir / "day8_eval_rows.jsonl"
    with train_path.open("w", encoding="utf-8") as train_out:
        for row in train_rows_expanded:
            train_out.write(json.dumps(row, ensure_ascii=False) + "\n")
    with eval_path.open("w", encoding="utf-8") as eval_out:
        for row in eval_rows:
            eval_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "output_dir": str(output_dir),
        "train_jsonl": str(train_path),
        "eval_jsonl": str(eval_path),
        "config_json": str(config_path),
        "n_source_rows": len(source_rows),
        "n_train_rows_pre_oversample": len(train_rows),
        "n_train_rows_post_oversample": len(train_rows_expanded),
        "n_eval_rows": len(eval_rows),
        "n_dialogues_train": len(split_dialogues["train"]),
        "n_dialogues_eval": len(split_dialogues["eval"]),
        "rows_by_intent_pre_oversample": {
            split: dict(counter) for split, counter in pre_counts.items()
        },
        "rows_by_intent_post_oversample_train": dict(post_train_counts),
        "rows_by_style": {
            split: dict(counter) for split, counter in pre_style_counts.items()
        },
        "intent_repeat_factors": repeat_map,
        "config": config,
    }
    summary_path = output_dir / "day8_data_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-jsonl", type=Path, default=DEFAULT_SOURCE_JSONL)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--styles",
        default="0.2,0.5,0.8",
        help="Comma-separated styles or weights. Example: balanced or 0.5",
    )
    p.add_argument("--eval-fraction", type=float, default=DEFAULT_EVAL_FRACTION)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--intent-balance-mode",
        choices=("none", "oversample_to_anchor"),
        default="oversample_to_anchor",
    )
    p.add_argument("--max-intent-repeat", type=int, default=DEFAULT_MAX_INTENT_REPEAT)
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    summary = process(args)
    logger.info(
        "Wrote %d train rows (%d after oversample) and %d eval rows",
        summary["n_train_rows_pre_oversample"],
        summary["n_train_rows_post_oversample"],
        summary["n_eval_rows"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
