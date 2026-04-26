#!/usr/bin/env python3
"""Deterministic Day 9 spot check for student structured-bid coverage.

The script samples gold Submit-Deal turns where the student submitted a native
bid and where the student stayed in ``utter``. Stayed rows are classified with
a deliberately simple heuristic:

* conservative-correct if there is no formal pending offer, the student
  utterance does not itself contain an extractable complete bid, and recent
  natural-language context does not contain an extractable complete bid;
* coverage-failure otherwise.

The fixed threshold is the pre-registered decision rule from Day 9.2.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from opponent_model.bid_extractor import extract_bid_from_utterance
from opponent_model.cache import DiskCache
from opponent_model.scripts.analyze_extracted_bids import (
    _student_cache_namespace,
    _student_cache_prompt,
)
from sft_8b.student_parser import parse_student_response


DEAL_ACTIONS = {"Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away"}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _dialogue_lookup(data_path: Path) -> Dict[str, Mapping[str, Any]]:
    with data_path.open() as f:
        dialogues = json.load(f)
    return {str(d.get("dialogue_id", i)): d for i, d in enumerate(dialogues)}


def _unique_by_dialogue(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    n: int,
) -> list[Mapping[str, Any]]:
    pool = list(rows)
    random.Random(seed).shuffle(pool)
    selected: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for row in pool:
        did = str(row.get("dialogue_id"))
        if did in seen:
            continue
        selected.append(row)
        seen.add(did)
        if len(selected) >= n:
            break
    return selected


def _recover_student_utterance(
    record: Mapping[str, Any],
    *,
    summary: Mapping[str, Any],
    cache: DiskCache,
) -> tuple[Optional[str], Optional[str]]:
    logged = str(((record.get("pred") or {}).get("utterance") or "")).strip()
    if logged:
        return logged, "logged"
    cfg = summary.get("config") or {}
    style = str(cfg.get("style_token") or "balanced")
    raw = cache.get(
        _student_cache_prompt(record, style=style),
        namespace=_student_cache_namespace(summary),
    )
    if not raw:
        return None, None
    parsed = parse_student_response(raw)
    utterance = str(parsed.get("utterance") or "").strip()
    return (utterance or None), ("student_cache" if utterance else None)


def _recent_context(
    record: Mapping[str, Any],
    dialogue: Mapping[str, Any],
    *,
    window: int,
) -> list[Dict[str, Any]]:
    turn_index = int(record["turn_index"])
    logs = list(dialogue.get("chat_logs") or [])
    context: list[Dict[str, Any]] = []
    for idx in range(max(0, turn_index - window), turn_index):
        turn = logs[idx]
        text = str(turn.get("text") or "").strip()
        if not text or text in DEAL_ACTIONS:
            continue
        context.append({
            "turn_index": idx,
            "speaker": turn.get("id"),
            "text": text,
            "extracted_bid": extract_bid_from_utterance(text),
        })
    return context


def _classify_stayed_row(
    record: Mapping[str, Any],
    *,
    dialogue: Mapping[str, Any],
    student_utterance: Optional[str],
    context_window: int,
) -> tuple[bool, str, list[Dict[str, Any]]]:
    pending = record.get("pending_offer")
    student_extracted = extract_bid_from_utterance(student_utterance or "")
    context = _recent_context(record, dialogue, window=context_window)
    context_has_bid = any(row.get("extracted_bid") is not None for row in context)
    if pending is None and student_extracted is None and not context_has_bid:
        return True, "no formal pending offer and no complete extracted bid in student/context text", context
    reasons = []
    if pending is not None:
        reasons.append("formal pending offer exists")
    if student_extracted is not None:
        reasons.append("student utterance contains an extractable bid")
    if context_has_bid:
        reasons.append("recent context contains an extractable bid")
    return False, "; ".join(reasons) or "coverage expectation present", context


def _branch_from_count(conservative_count: int, total: int) -> str:
    if total == 5:
        if conservative_count >= 4:
            return "conservative_correct"
        if conservative_count <= 2:
            return "coverage_limitation"
        return "resample_required"
    if conservative_count >= 7:
        return "conservative_correct"
    return "coverage_limitation"


def _abstract_sentence(branch: str) -> str:
    if branch == "conservative_correct":
        return (
            "the student emits structured bids only when the dialogue commits "
            "to one, with high accuracy when it does."
        )
    return (
        "where the student commits to a structured bid, accuracy is high; "
        "coverage is addressed in Section 6."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=Path("opponent_model/results/turn_eval_student_balanced_full150"),
    )
    ap.add_argument("--data", type=Path, default=Path("data/casino_test.json"))
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--context-window", type=int, default=4)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("opponent_model/results/day9_student_coverage_spotcheck"),
    )
    args = ap.parse_args()

    summary = _load_json(args.run_dir / "turn_summary.json")
    records = _load_jsonl(args.run_dir / "turn_records.jsonl")
    dialogues = _dialogue_lookup(args.data)
    cache_path = Path(
        (summary.get("config") or {}).get(
            "student_cache_path",
            str(args.run_dir / "student_cache.sqlite"),
        )
    )
    cache = DiskCache(cache_path)

    gold_submit = [r for r in records if (r.get("true") or {}).get("bid") is not None]
    submit_rows = [
        r for r in gold_submit
        if (r.get("pred") or {}).get("action") == "submit"
    ]
    stayed_rows = [
        r for r in gold_submit
        if (r.get("pred") or {}).get("action") == "utter"
    ]

    selected_submit = _unique_by_dialogue(submit_rows, seed=args.seed, n=5)
    selected_stayed = _unique_by_dialogue(stayed_rows, seed=args.seed + 1, n=5)
    if len(selected_stayed) < 5:
        raise RuntimeError(f"needed 5 stayed rows, found {len(selected_stayed)}")
    if len(selected_submit) < 5:
        raise RuntimeError(f"needed 5 submit rows, found {len(selected_submit)}")

    debug_rows: list[Dict[str, Any]] = []
    stayed_results: list[bool] = []

    for label, selected in (("student_submit", selected_submit), ("student_utter", selected_stayed)):
        for rec in selected:
            utterance, source = _recover_student_utterance(rec, summary=summary, cache=cache)
            dialogue = dialogues.get(str(rec.get("dialogue_id")), {})
            row = {
                "sample_type": label,
                "dialogue_id": rec.get("dialogue_id"),
                "turn_index": rec.get("turn_index"),
                "perspective": rec.get("perspective"),
                "human_turn_text": rec.get("turn_text"),
                "student_action": (rec.get("pred") or {}).get("action"),
                "student_bid": (rec.get("pred") or {}).get("bid"),
                "gold_bid": (rec.get("true") or {}).get("bid"),
                "student_utterance": utterance,
                "utterance_source": source,
                "student_extracted_bid": extract_bid_from_utterance(utterance or ""),
            }
            if label == "student_utter":
                conservative, reason, context = _classify_stayed_row(
                    rec,
                    dialogue=dialogue,
                    student_utterance=utterance,
                    context_window=args.context_window,
                )
                stayed_results.append(conservative)
                row["conservative_correct"] = conservative
                row["classification_reason"] = reason
                row["recent_context"] = context
            debug_rows.append(row)

    first_branch = _branch_from_count(sum(stayed_results), len(stayed_results))
    if first_branch == "resample_required":
        already = {str(r.get("dialogue_id")) for r in selected_stayed}
        remaining = [r for r in stayed_rows if str(r.get("dialogue_id")) not in already]
        extra = _unique_by_dialogue(remaining, seed=args.seed + 2, n=5)
        if len(extra) < 5:
            raise RuntimeError(f"needed 5 extra stayed rows, found {len(extra)}")
        for rec in extra:
            utterance, source = _recover_student_utterance(rec, summary=summary, cache=cache)
            dialogue = dialogues.get(str(rec.get("dialogue_id")), {})
            conservative, reason, context = _classify_stayed_row(
                rec,
                dialogue=dialogue,
                student_utterance=utterance,
                context_window=args.context_window,
            )
            stayed_results.append(conservative)
            debug_rows.append({
                "sample_type": "student_utter_resample",
                "dialogue_id": rec.get("dialogue_id"),
                "turn_index": rec.get("turn_index"),
                "perspective": rec.get("perspective"),
                "human_turn_text": rec.get("turn_text"),
                "student_action": (rec.get("pred") or {}).get("action"),
                "student_bid": (rec.get("pred") or {}).get("bid"),
                "gold_bid": (rec.get("true") or {}).get("bid"),
                "student_utterance": utterance,
                "utterance_source": source,
                "student_extracted_bid": extract_bid_from_utterance(utterance or ""),
                "conservative_correct": conservative,
                "classification_reason": reason,
                "recent_context": context,
            })

    conservative_count = sum(stayed_results)
    total_stayed = len(stayed_results)
    branch = _branch_from_count(conservative_count, total_stayed)
    sentence = _abstract_sentence(branch)
    report = {
        "metadata": {
            "run_dir": str(args.run_dir),
            "data": str(args.data),
            "seed": args.seed,
            "context_window": args.context_window,
            "threshold_rule": (
                "5-row sample: >=4/5 conservative-correct -> Section 3.3; "
                "<=2/5 -> Section 6; 3/5 -> sample 5 more and decide "
                ">=7/10 vs <=6/10"
            ),
        },
        "support": {
            "gold_submit_turns": len(gold_submit),
            "student_submit_gold_submit_turns": len(submit_rows),
            "student_utter_gold_submit_turns": len(stayed_rows),
        },
        "spotcheck": {
            "stayed_conservative_correct": conservative_count,
            "stayed_total": total_stayed,
            "branch": branch,
            "active_abstract_sentence": sentence,
        },
        "note": (
            "This is a deterministic heuristic triage, not a replacement for "
            "manual annotation; the debug JSONL contains the sampled text."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    debug_path = args.output_dir / "debug_rows.jsonl"
    note_path = args.output_dir / "methods_note.md"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with debug_path.open("w", encoding="utf-8") as f:
        for row in debug_rows:
            f.write(json.dumps(row) + "\n")
    note_path.write_text(
        "\n".join(
            [
                "# Day 9 Student Coverage Spot Check",
                "",
                (
                    f"On the fixed spot check, {conservative_count}/{total_stayed} "
                    "sampled gold Submit-Deal turns where the student stayed in "
                    "`utter` were classified as conservative-correct by the "
                    "pre-registered heuristic."
                ),
                "",
                f"Active branch: `{branch}`.",
                "",
                f"Abstract wording: \"{sentence}\"",
                "",
                (
                    "Native bid cosine for the student must not be reported "
                    "without its support (`n=14` in the current balanced run)."
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote {summary_path}")
    print(f"Wrote {debug_path}")
    print(f"Wrote {note_path}")
    print(f"branch: {branch} ({conservative_count}/{total_stayed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
