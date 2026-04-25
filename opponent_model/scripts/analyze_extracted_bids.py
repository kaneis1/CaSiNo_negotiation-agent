#!/usr/bin/env python3
"""Analyze native and extracted bid support for student / teacher / baseline.

The extractor is intentionally high precision and is applied uniformly to all
three agents. Student utterances can be backfilled from the saved SQLite cache;
teacher utterances can be reconstructed from the deterministic template; the
baseline must be rerun with utterance logging so the logged text is available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from opponent_model.bid_extractor import extract_bid_from_utterance
from opponent_model.cache import DiskCache
from opponent_model.hypotheses import ITEMS
from opponent_model.turn_level_metrics import cosine_similarity
from sft_8b.bayesian_agent import template_utterance
from sft_8b.student_parser import parse_student_response


NOTE = (
    "student content_ok reflects the current distillation target design: "
    "the balanced Day 9 student produced parseable intents on all turns, but "
    "only emitted non-null selected_content on a small subset of turns. "
    "This is not a student parse-failure story."
)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _record_key(rec: Mapping[str, Any]) -> Tuple[Any, str, int]:
    return (
        rec["dialogue_id"],
        str(rec.get("perspective", "")),
        int(rec["turn_index"]),
    )


def _self_counts_from_bid(bid: Any) -> Optional[Dict[str, int]]:
    if bid is None or not isinstance(bid, (list, tuple)):
        return None
    if len(bid) == 6:
        arr = bid[:3]
    elif len(bid) == 3:
        arr = bid
    else:
        return None
    try:
        return {item: int(arr[i]) for i, item in enumerate(ITEMS)}
    except (TypeError, ValueError):
        return None


def _bid_vector_from_split(split: Mapping[str, Any]) -> List[float]:
    self_tuple = list(split["self_tuple"])
    opp_tuple = list(split["opp_tuple"])
    return [float(x) for x in (self_tuple + opp_tuple)]


def _student_cache_namespace(summary: Mapping[str, Any]) -> str:
    cfg = summary.get("config") or {}
    return "|".join([
        "distilled_student_turn_agent_v1",
        f"base_model={cfg.get('base_model')}",
        f"adapter={cfg.get('adapter')}",
        f"max_new_tokens={cfg.get('student_max_new_tokens')}",
        f"temperature={cfg.get('temperature')}",
    ])


def _student_cache_prompt(
    record: Mapping[str, Any],
    *,
    style: str,
) -> str:
    return json.dumps(
        {
            "dialogue_id": record["dialogue_id"],
            "turn_index": record["turn_index"],
            "perspective": record["perspective"],
            "style": style,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _recover_student_utterance(
    record: Mapping[str, Any],
    *,
    summary: Mapping[str, Any],
    cache: DiskCache,
) -> Tuple[Optional[str], Optional[str]]:
    utterance = str(((record.get("pred") or {}).get("utterance") or "")).strip()
    if utterance:
        return utterance, "logged"

    cfg = summary.get("config") or {}
    style = str(cfg.get("style_token") or "balanced")
    prompt = _student_cache_prompt(record, style=style)
    raw = cache.get(prompt, namespace=_student_cache_namespace(summary))
    if not raw:
        return None, None
    parsed = parse_student_response(raw)
    utterance = str(parsed.get("utterance") or "").strip()
    return (utterance or None), ("student_cache" if utterance else None)


def _recover_teacher_utterance(
    record: Mapping[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    utterance = str(((record.get("pred") or {}).get("utterance") or "")).strip()
    if utterance:
        return utterance, "logged"

    counts = _self_counts_from_bid((record.get("pred") or {}).get("bid"))
    if counts is None:
        return None, None
    return template_utterance(counts), "teacher_template"


def _recover_logged_utterance(
    record: Mapping[str, Any],
    *,
    agent_name: str,
) -> Tuple[Optional[str], Optional[str]]:
    utterance = str(((record.get("pred") or {}).get("utterance") or "")).strip()
    if utterance:
        return utterance, "logged"
    return None, None


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _analyze_records(
    *,
    name: str,
    records: List[Dict[str, Any]],
    utterance_fn,
) -> Tuple[Dict[str, Any], Dict[Tuple[Any, str, int], Dict[str, Any]], List[Dict[str, Any]]]:
    gold_submit_turns = 0
    native_predicted = 0
    native_overlap = 0
    native_cosines: List[float] = []
    extracted_predicted = 0
    extracted_overlap = 0
    extracted_cosines: List[float] = []
    action_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}

    rows_by_key: Dict[Tuple[Any, str, int], Dict[str, Any]] = {}
    debug_rows: List[Dict[str, Any]] = []

    for rec in records:
        key = _record_key(rec)
        pred = rec.get("pred") or {}
        truth = rec.get("true") or {}
        gold_bid = truth.get("bid")
        native_bid = pred.get("bid")

        action = pred.get("action")
        action_key = "None" if action is None else str(action)
        action_counts[action_key] = action_counts.get(action_key, 0) + 1

        if gold_bid is not None:
            gold_submit_turns += 1
        if native_bid is not None:
            native_predicted += 1
        if gold_bid is not None and native_bid is not None:
            native_overlap += 1
            native_cosines.append(cosine_similarity(native_bid, gold_bid))

        utterance, source = utterance_fn(rec)
        if source is not None:
            source_counts[source] = source_counts.get(source, 0) + 1
        extracted = extract_bid_from_utterance(utterance or "")
        extracted_bid = _bid_vector_from_split(extracted) if extracted is not None else None

        if extracted_bid is not None:
            extracted_predicted += 1
        if gold_bid is not None and extracted_bid is not None:
            extracted_overlap += 1
            extracted_cosines.append(cosine_similarity(extracted_bid, gold_bid))
            debug_rows.append(
                {
                    "agent": name,
                    "dialogue_id": rec["dialogue_id"],
                    "perspective": rec["perspective"],
                    "turn_index": rec["turn_index"],
                    "action": pred.get("action"),
                    "utterance": utterance,
                    "source": source,
                    "native_bid": native_bid,
                    "extracted_bid": extracted_bid,
                    "gold_bid": gold_bid,
                    "native_cosine": (
                        cosine_similarity(native_bid, gold_bid)
                        if native_bid is not None else None
                    ),
                    "extracted_cosine": cosine_similarity(extracted_bid, gold_bid),
                }
            )

        rows_by_key[key] = {
            "gold_bid": gold_bid,
            "native_bid": native_bid,
            "utterance": utterance,
            "source": source,
            "extracted_bid": extracted_bid,
        }

    summary = {
        "native": {
            "gold_submit_turns": gold_submit_turns,
            "predicted_bid_turns": native_predicted,
            "scored_overlap": native_overlap,
            "coverage_vs_gold": (
                float(native_overlap / gold_submit_turns) if gold_submit_turns else None
            ),
            "bid_cosine_mean": _mean(native_cosines),
        },
        "extracted": {
            "gold_submit_turns": gold_submit_turns,
            "predicted_bid_turns": extracted_predicted,
            "scored_overlap": extracted_overlap,
            "coverage_vs_gold": (
                float(extracted_overlap / gold_submit_turns) if gold_submit_turns else None
            ),
            "bid_cosine_mean": _mean(extracted_cosines),
        },
        "action_counts": action_counts,
        "utterance_sources": source_counts,
    }
    return summary, rows_by_key, debug_rows


def _intersection_summary(
    rows: Mapping[str, Mapping[Tuple[Any, str, int], Mapping[str, Any]]],
    names: Tuple[str, ...],
) -> Dict[str, Any]:
    key_sets = []
    for name in names:
        key_sets.append({
            key for key, row in rows[name].items()
            if row.get("gold_bid") is not None and row.get("extracted_bid") is not None
        })
    keys = set.intersection(*key_sets) if key_sets else set()
    per_agent: Dict[str, float] = {}
    for name in names:
        vals = [
            cosine_similarity(rows[name][key]["extracted_bid"], rows[name][key]["gold_bid"])
            for key in sorted(keys)
        ]
        per_agent[name] = _mean(vals)
    return {
        "count": len(keys),
        "bid_cosine_mean": per_agent,
        "keys": [
            {"dialogue_id": did, "perspective": role, "turn_index": turn}
            for did, role, turn in sorted(keys)
        ],
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--student-summary",
        type=Path,
        default=Path("opponent_model/results/turn_eval_student_balanced_full150/turn_summary.json"),
    )
    ap.add_argument(
        "--student-records",
        type=Path,
        default=Path("opponent_model/results/turn_eval_student_balanced_full150/turn_records.jsonl"),
    )
    ap.add_argument(
        "--teacher-records",
        type=Path,
        default=Path("opponent_model/results/turn_eval_bayesian_lambda1.0_m5_f0.50_full150/turn_records.jsonl"),
    )
    ap.add_argument(
        "--baseline-records",
        type=Path,
        default=Path("opponent_model/results/turn_eval_structured_cot_p3_live_70b_m1_150_with_utterance/turn_records.jsonl"),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("opponent_model/results/day9_extracted_bid_analysis"),
    )
    ap.add_argument(
        "--student-cache-path",
        type=Path,
        default=None,
        help="Override the cache path from student_summary.json if needed.",
    )
    args = ap.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    student_summary = _load_json(args.student_summary)
    student_records = _load_jsonl(args.student_records)
    teacher_records = _load_jsonl(args.teacher_records)
    baseline_records = _load_jsonl(args.baseline_records)

    cache_path = args.student_cache_path
    if cache_path is None:
        cfg = student_summary.get("config") or {}
        raw = cfg.get("student_cache_path")
        cache_path = Path(raw) if raw else None
    if cache_path is None:
        raise RuntimeError("student cache path missing; cannot backfill student utterances")

    student_cache = DiskCache(cache_path)
    try:
        student_summary_block, student_rows, student_debug = _analyze_records(
            name="student",
            records=student_records,
            utterance_fn=lambda rec: _recover_student_utterance(
                rec,
                summary=student_summary,
                cache=student_cache,
            ),
        )
    finally:
        student_cache.close()

    teacher_summary_block, teacher_rows, teacher_debug = _analyze_records(
        name="teacher",
        records=teacher_records,
        utterance_fn=_recover_teacher_utterance,
    )
    baseline_summary_block, baseline_rows, baseline_debug = _analyze_records(
        name="baseline_live",
        records=baseline_records,
        utterance_fn=lambda rec: _recover_logged_utterance(rec, agent_name="baseline_live"),
    )
    if baseline_summary_block["utterance_sources"].get("logged", 0) == 0:
        raise RuntimeError(
            "baseline_live records contain no logged pred.utterance values. "
            "Rerun the live baseline after the utterance-logging patch."
        )

    rows = {
        "student": student_rows,
        "teacher": teacher_rows,
        "baseline_live": baseline_rows,
    }
    pairwise = {
        "student__teacher": _intersection_summary(rows, ("student", "teacher")),
        "student__baseline_live": _intersection_summary(rows, ("student", "baseline_live")),
        "teacher__baseline_live": _intersection_summary(rows, ("teacher", "baseline_live")),
    }
    three_way = _intersection_summary(rows, ("student", "teacher", "baseline_live"))

    summary = {
        "metadata": {
            "student_summary": str(args.student_summary),
            "student_records": str(args.student_records),
            "teacher_records": str(args.teacher_records),
            "baseline_records": str(args.baseline_records),
            "student_cache_path": str(cache_path),
            "note": NOTE,
            "student_agent_summary": student_summary.get("agent_summary"),
        },
        "agents": {
            "student": student_summary_block,
            "teacher": teacher_summary_block,
            "baseline_live": baseline_summary_block,
        },
        "extracted_intersections": {
            "pairwise": pairwise,
            "three_way": three_way,
        },
    }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    debug_path = args.output_dir / "debug_scored_turns.jsonl"
    with debug_path.open("w", encoding="utf-8") as f:
        for row in student_debug + teacher_debug + baseline_debug:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {summary_path}")
    print(f"Wrote {debug_path}")
    print(
        "Three-way extracted shared support:",
        summary["extracted_intersections"]["three_way"]["count"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
