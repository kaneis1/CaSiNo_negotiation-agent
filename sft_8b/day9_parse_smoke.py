"""Three-dialogue parse smoke for baseline replay, teacher, and student.

Purpose:
    Before launching the full Day 9 held-out eval, confirm that the new
    distilled-student parser works end-to-end on real held-out prefixes and
    that the existing baseline/teacher paths still align with the same turn
    evaluator.

Default dialogue IDs were chosen from the held-out split because, under the
existing saved replay/teacher runs, they jointly cover natural utterances,
submit turns, and accept/reject decisions for ``mturk_agent_1``.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from opponent_model.turn_agents import (
    DistilledStudentTurnAgent,
    KeywordStrategyClassifier,
)
from opponent_model.turn_level_metrics import TurnRecord, turn_level_eval
from sft_8b.bayesian_agent import BayesianTurnAgent
from sft_8b.predict import SftModelFn
from sft_8b.student_model import StudentModelFn
from structured_cot.replay_turn_agent import StructuredCoTReplayAgent

logger = logging.getLogger("sft_8b.day9_parse_smoke")

DEFAULT_BASE_MODEL = (
    "/sc/arion/scratch/cuiz02/hf_cache/transformers/Meta-Llama-3.1-8B-Instruct"
)
DEFAULT_TEACHER_ADAPTER = "sft_8b/results/lora_run/lora_best"
DEFAULT_STUDENT_ADAPTER = "sft_8b/results/day8_lora_run/lora_best"
DEFAULT_DATA = "data/casino_test.json"
DEFAULT_ANNOTATIONS = "CaSiNo/data/casino_ann.json"
DEFAULT_BASELINE_TURNS = "structured_cot/results/protocol1_70b_full/turns.jsonl"
DEFAULT_DIALOGUE_IDS = (33, 467, 548)


def _parse_dialogue_ids(raw: str) -> List[int]:
    out: List[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("no dialogue ids supplied")
    return out


def _load_dialogues(path: Path, *, dialogue_ids: Sequence[int]) -> List[Dict[str, Any]]:
    with path.open() as f:
        dialogues = json.load(f)
    wanted = set(int(x) for x in dialogue_ids)
    out = [d for d in dialogues if int(d.get("dialogue_id")) in wanted]
    missing = sorted(wanted - {int(d.get("dialogue_id")) for d in out})
    if missing:
        raise ValueError(f"dialogue ids not found in {path}: {missing}")
    out.sort(key=lambda d: dialogue_ids.index(int(d.get("dialogue_id"))))
    return out


def _load_annotations(path: Path) -> Dict[Any, Any]:
    with path.open() as f:
        rows = json.load(f)
    return {row["dialogue_id"]: row.get("annotations", []) for row in rows}


def _records_to_jsonl(path: Path, records: Iterable[TurnRecord]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict()) + "\n")


def _summary_without_records(result: Mapping[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in dict(result).items() if k != "records"}
    return out


def _cleanup_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _run_one(
    *,
    name: str,
    agent: Any,
    dialogues: Sequence[Mapping[str, Any]],
    annotations: Mapping[Any, Sequence[Any]],
    perspectives: Sequence[str],
    output_dir: Path,
) -> Dict[str, Any]:
    logger.info("Running smoke for %s", name)
    result = turn_level_eval(
        dialogues,
        agent,
        perspectives=perspectives,
        annotations_by_dialogue=annotations,
    )
    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    _records_to_jsonl(run_dir / "turn_records.jsonl", result["records"])
    summary = _summary_without_records(result)
    (run_dir / "turn_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return result


def _compact_supports(result: Mapping[str, Any]) -> Dict[str, int]:
    return {
        "accept": int((result.get("accept") or {}).get("support") or 0),
        "bid": int((result.get("bid_cosine") or {}).get("support") or 0),
        "strategy": int((result.get("strategy_macro_f1") or {}).get("support") or 0),
        "brier": int((result.get("brier") or {}).get("support") or 0),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--teacher-adapter", default=DEFAULT_TEACHER_ADAPTER)
    p.add_argument("--student-adapter", default=DEFAULT_STUDENT_ADAPTER)
    p.add_argument("--baseline-turns-path", default=DEFAULT_BASELINE_TURNS)
    p.add_argument("--dialogue-ids", default=",".join(str(x) for x in DEFAULT_DIALOGUE_IDS))
    p.add_argument("--perspectives", default="mturk_agent_1")
    p.add_argument("--style-token", default="balanced")
    p.add_argument("--teacher-posterior-k", type=int, default=16)
    p.add_argument("--teacher-posterior-temperature", type=float, default=0.7)
    p.add_argument("--teacher-accept-margin", type=int, default=5)
    p.add_argument("--teacher-accept-floor", type=float, default=0.50)
    p.add_argument("--teacher-max-new-tokens", type=int, default=128)
    p.add_argument("--student-max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--output-dir", default="sft_8b/results/day9_parse_smoke")
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero if parse/support checks fail.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    dialogue_ids = _parse_dialogue_ids(args.dialogue_ids)
    perspectives = tuple(
        tok.strip() for tok in str(args.perspectives).split(",") if tok.strip()
    )
    if not perspectives:
        raise ValueError("at least one perspective is required")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dialogues = _load_dialogues(Path(args.data), dialogue_ids=dialogue_ids)
    annotations = _load_annotations(Path(args.annotations))

    overall: Dict[str, Any] = {
        "config": {
            **vars(args),
            "dialogue_ids": dialogue_ids,
            "perspectives": list(perspectives),
        }
    }

    baseline = StructuredCoTReplayAgent(
        Path(args.baseline_turns_path),
        strategy_classifier=KeywordStrategyClassifier(),
    )
    baseline_result = _run_one(
        name="baseline_replay",
        agent=baseline,
        dialogues=dialogues,
        annotations=annotations,
        perspectives=perspectives,
        output_dir=output_dir,
    )
    overall["baseline_replay"] = {
        "supports": _compact_supports(baseline_result),
        "replay_summary": baseline.summary,
    }

    teacher_model = SftModelFn(
        base_model=args.base_model,
        adapter_path=args.teacher_adapter,
        max_new_tokens=args.teacher_max_new_tokens,
        temperature=max(args.temperature, 0.7),
    )
    teacher = BayesianTurnAgent(
        teacher_model,
        K=args.teacher_posterior_k,
        temperature=args.teacher_posterior_temperature,
        accept_margin=args.teacher_accept_margin,
        accept_floor=args.teacher_accept_floor,
        strategy_classifier=KeywordStrategyClassifier(),
    )
    teacher_result = _run_one(
        name="bayesian_teacher",
        agent=teacher,
        dialogues=dialogues,
        annotations=annotations,
        perspectives=perspectives,
        output_dir=output_dir,
    )
    overall["bayesian_teacher"] = {
        "supports": _compact_supports(teacher_result),
    }
    del teacher
    del teacher_model
    _cleanup_gpu()

    student_model = StudentModelFn(
        base_model=args.base_model,
        adapter_path=args.student_adapter,
        max_new_tokens=args.student_max_new_tokens,
        temperature=args.temperature,
    )
    student = DistilledStudentTurnAgent(
        student_model,
        style=args.style_token,
        strategy_classifier=KeywordStrategyClassifier(),
    )
    student_result = _run_one(
        name="distilled_student",
        agent=student,
        dialogues=dialogues,
        annotations=annotations,
        perspectives=perspectives,
        output_dir=output_dir,
    )
    overall["distilled_student"] = {
        "supports": _compact_supports(student_result),
        "parse_summary": student.summary,
        "last_parse": student.last_parse,
    }

    overall["checks"] = {
        "baseline_replay_hits_positive": baseline.summary.get("hits", 0) > 0,
        "teacher_has_accept_support": overall["bayesian_teacher"]["supports"]["accept"] > 0,
        "teacher_has_bid_support": overall["bayesian_teacher"]["supports"]["bid"] > 0,
        "teacher_has_brier_support": overall["bayesian_teacher"]["supports"]["brier"] > 0,
        "student_has_accept_support": overall["distilled_student"]["supports"]["accept"] > 0,
        "student_has_bid_support": overall["distilled_student"]["supports"]["bid"] > 0,
        "student_has_strategy_support": overall["distilled_student"]["supports"]["strategy"] > 0,
        "student_has_brier_support": overall["distilled_student"]["supports"]["brier"] > 0,
        "student_parse_errors_zero": overall["distilled_student"]["parse_summary"]["parse_errors"] == 0,
        "student_all_calls_have_posterior": (
            overall["distilled_student"]["parse_summary"]["posterior_ok"]
            == overall["distilled_student"]["parse_summary"]["calls"]
        ),
        "student_all_calls_have_intent": (
            overall["distilled_student"]["parse_summary"]["intent_ok"]
            == overall["distilled_student"]["parse_summary"]["calls"]
        ),
    }

    (output_dir / "smoke_summary.json").write_text(
        json.dumps(overall, indent=2),
        encoding="utf-8",
    )

    print("Day 9 parse smoke")
    print(f"  dialogues: {dialogue_ids}")
    print(
        "  baseline replay supports:",
        json.dumps(overall["baseline_replay"]["supports"], sort_keys=True),
        "| replay",
        json.dumps(overall["baseline_replay"]["replay_summary"], sort_keys=True),
    )
    print(
        "  bayesian teacher supports:",
        json.dumps(overall["bayesian_teacher"]["supports"], sort_keys=True),
    )
    print(
        "  distilled student supports:",
        json.dumps(overall["distilled_student"]["supports"], sort_keys=True),
        "| parse",
        json.dumps(overall["distilled_student"]["parse_summary"], sort_keys=True),
    )

    failed = [name for name, ok in overall["checks"].items() if not ok]
    if failed:
        print("  checks failed:", json.dumps(failed))
        return 1 if args.strict else 0
    print("  checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
