"""Run :func:`opponent_model.turn_level_eval` on held-out CaSiNo dialogues.

Produces a summary.json + per-turn records.jsonl, mirroring the layout
of ``opponent_model/eval_run.py`` so downstream analysis scripts can
share helpers.

Usage:
    # smoke-test: 5 dialogues, dummy LLM, uniform-baseline agent
    python -m opponent_model.turn_eval_run \\
        --data data/casino_test.json \\
        --output-dir opponent_model/results/turn_eval_smoke \\
        --max-dialogues 5 --agent uniform --annotations CaSiNo/data/casino_ann.json

    # full run on the held-out test split, hybrid agent, dummy LLM
    python -m opponent_model.turn_eval_run \\
        --data data/casino_test.json \\
        --output-dir opponent_model/results/turn_eval_hybrid \\
        --agent hybrid --dummy-llm \\
        --annotations CaSiNo/data/casino_ann.json

    # SFT 8B agent (LoRA adapter), strategy via keyword baseline
    python -m opponent_model.turn_eval_run \\
        --data data/casino_test.json \\
        --output-dir opponent_model/results/turn_eval_sft \\
        --agent sft --base-model <path> --adapter <lora_dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from opponent_model.turn_agents import (
    DistilledStudentTurnAgent,
    HybridTurnAgent,
    KeywordStrategyClassifier,
    SftTurnAgent,
    UniformTurnAgent,
)
from opponent_model.turn_level_metrics import (
    TurnRecord,
    format_turn_level_summary,
    turn_level_eval,
)


def _setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("opponent_model.turn_eval_run")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def _load_annotations_lookup(path: Optional[Path]) -> Dict[Any, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"annotations file not found: {path}")
    with path.open() as f:
        ann_data = json.load(f)
    return {d["dialogue_id"]: d.get("annotations", []) for d in ann_data}


def _build_dummy_llm() -> Any:
    """Reuse the same dummy LLM as opponent_model.eval_run for parity."""
    from opponent_model.eval_run import _build_dummy_llm
    return _build_dummy_llm()


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True,
                   help="held-out dialogues JSON (e.g. data/casino_test.json)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--annotations", default=None,
                   help="casino_ann.json with per-utterance strategy tags. "
                        "If omitted, falls back to dialogue.annotations if present.")
    p.add_argument("--max-dialogues", type=int, default=None)
    p.add_argument("--agent",
                   choices=("uniform", "hybrid", "sft",
                            "bayesian", "distilled_student",
                            "structured_cot_replay",
                            "structured_cot_live"),
                   default="uniform",
                   help="which TurnLevelAgent to evaluate.")

    # perspective restriction (handy for apples-to-apples comparisons
    # against a Protocol-1 replay that only covers mturk_agent_1).
    p.add_argument("--perspectives", default="mturk_agent_1,mturk_agent_2",
                   help="comma-separated list of perspective role ids to score on.")

    # hybrid / sft shared LLM args
    p.add_argument("--dummy-llm", action="store_true",
                   help="use the deterministic dummy LLM (no GPU).")
    p.add_argument("--model-id", default=None,
                   help="HF model id / local snapshot path (hybrid / structured_cot_live). "
                        "structured_cot_live defaults to Llama-3.3-70B-Instruct snapshot.")

    # sft / bayesian args
    p.add_argument("--base-model", default=None,
                   help="base model path for the SFT adapter (sft / bayesian agents).")
    p.add_argument("--adapter", default=None,
                   help="LoRA adapter directory (sft / bayesian agents).")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--style-token", default="balanced",
                   help="style token for the distilled_student agent.")
    p.add_argument("--student-max-new-tokens", type=int, default=256,
                   help="generation budget for distilled_student tagged outputs.")
    p.add_argument("--student-cache-path", default=None,
                   help="SQLite cache for distilled student raw generations. "
                        "Defaults to <output_dir>/student_cache.sqlite.")
    p.add_argument("--student-parse-log", default=None,
                   help="JSONL for distilled student parse failures with raw generations. "
                        "Defaults to <output_dir>/student_parse_failures.jsonl.")

    # bayesian agent knobs
    p.add_argument("--lambda", dest="lambda_", type=float, default=1.0,
                   help="λ style knob for the menu builder (bayesian agent).")
    p.add_argument("--lambda-from-svo", action="store_true",
                   help="condition Bayesian λ on participant_info[role].personality.svo.")
    p.add_argument("--posterior-k", type=int, default=16,
                   help="# MC samples for the SFT posterior (bayesian agent).")
    p.add_argument("--posterior-temperature", type=float, default=0.7,
                   help="sampling temperature for the SFT posterior (bayesian agent).")
    p.add_argument("--accept-margin", type=int, default=5,
                   help="continuation-cost margin in the accept rule "
                        "(bayesian agent); 0 = strict argmax.")
    p.add_argument("--accept-floor", type=float, default=0.50,
                   help="Pareto floor (fraction of MAX_SELF_POINTS=36) in "
                        "the accept rule; an offer at or above this fraction "
                        "is accepted unconditionally. 1.0 disables the floor.")

    # structured-cot replay knobs
    p.add_argument("--replay-turns-path", default=None,
                   help="path to a finished Protocol-1 turns.jsonl "
                        "(structured_cot_replay agent).")

    # structured-cot live (Protocol 3 — gold history, full support)
    p.add_argument("--structured-cot-seed", type=int, default=2024,
                   help="RNG seed forwarded to StructuredLLMClient (structured_cot_live).")
    p.add_argument("--structured-cot-parse-log", default=None,
                   help="optional JSONL path for parse failures (structured_cot_live).")

    # hybrid knobs (matched to opponent_model/eval_run.py defaults)
    p.add_argument("--likelihood-temperature", type=float, default=25.0)
    p.add_argument("--likelihood-clip", default="-3,3")
    return p


def _parse_clip(s: str) -> tuple:
    if s.lower() in ("none", "off", ""):
        return (None, None)
    parts = s.split(",")
    if len(parts) != 2:
        raise ValueError(f"--likelihood-clip must be 'lo,hi' or 'none' (got {s!r})")
    return (float(parts[0]), float(parts[1]))


def _build_agent(args: argparse.Namespace, *, output_dir: Path) -> Any:
    if args.agent == "uniform":
        return UniformTurnAgent()

    if args.agent == "hybrid":
        if args.dummy_llm:
            client = _build_dummy_llm()
        else:
            from prompt_engineer.llm.client import LlamaClient
            if args.model_id is None:
                raise ValueError("--model-id is required for hybrid agent (or pass --dummy-llm).")
            client = LlamaClient(
                model_id=args.model_id,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
        return HybridTurnAgent(
            client,
            strategy_classifier=KeywordStrategyClassifier(),
            propose_bid=True,
            likelihood_temperature=args.likelihood_temperature,
            likelihood_clip=_parse_clip(args.likelihood_clip),
            strict_likelihood=False,
        )

    if args.agent == "sft":
        from sft_8b.predict import SftModelFn
        if args.base_model is None:
            raise ValueError("--base-model is required for sft agent.")
        sft = SftModelFn(
            base_model=args.base_model,
            adapter_path=args.adapter,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        return SftTurnAgent(sft, strategy_classifier=KeywordStrategyClassifier())

    if args.agent == "bayesian":
        from sft_8b.bayesian_agent import BayesianTurnAgent
        if args.dummy_llm:
            class _DummyPosteriorModel:
                base_model = "dummy"
                adapter_path = None
                max_new_tokens = 0
                temperature = 0.0

                def generate_raw(self, prompt: str, *, K: int, temperature: float):
                    return [
                        '{"prefs":["Food","Water","Firewood"],'
                        '"satisfaction":"Slightly satisfied"}'
                    ] * int(K)

            sft = _DummyPosteriorModel()
        else:
            from sft_8b.predict import SftModelFn
            if args.base_model is None:
                raise ValueError("--base-model is required for bayesian agent.")
            # K samples need do_sample=True; temperature at load-time is ignored
            # by generate_raw() which passes `posterior_temperature` directly,
            # but we still need max_new_tokens tuned for ~96 tokens of JSON.
            sft = SftModelFn(
                base_model=args.base_model,
                adapter_path=args.adapter,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        lambda_fn = None
        if args.lambda_from_svo:
            from sft_8b.svo_to_lambda import svo_to_lambda
            lambda_fn = lambda personality: svo_to_lambda(
                (personality or {}).get("svo")
            )
        return BayesianTurnAgent(
            sft,
            lambda_=args.lambda_,
            lambda_fn=lambda_fn,
            K=args.posterior_k,
            temperature=args.posterior_temperature,
            accept_margin=args.accept_margin,
            accept_floor=args.accept_floor,
            strategy_classifier=KeywordStrategyClassifier(),
        )

    if args.agent == "distilled_student":
        from sft_8b.student_model import StudentModelFn
        if args.base_model is None:
            raise ValueError("--base-model is required for distilled_student agent.")
        student = StudentModelFn(
            base_model=args.base_model,
            adapter_path=args.adapter,
            max_new_tokens=args.student_max_new_tokens,
            temperature=args.temperature,
        )
        return DistilledStudentTurnAgent(
            student,
            style=args.style_token,
            strategy_classifier=KeywordStrategyClassifier(),
            cache_path=(
                Path(args.student_cache_path)
                if args.student_cache_path else output_dir / "student_cache.sqlite"
            ),
            parse_log_path=(
                Path(args.student_parse_log)
                if args.student_parse_log else output_dir / "student_parse_failures.jsonl"
            ),
        )

    if args.agent == "structured_cot_replay":
        from structured_cot.replay_turn_agent import StructuredCoTReplayAgent
        if args.replay_turns_path is None:
            raise ValueError(
                "--replay-turns-path is required for structured_cot_replay agent."
            )
        from pathlib import Path as _Path
        return StructuredCoTReplayAgent(
            _Path(args.replay_turns_path),
            strategy_classifier=KeywordStrategyClassifier(),
        )

    if args.agent == "structured_cot_live":
        from pathlib import Path as _Path

        from structured_cot.live_turn_agent import StructuredCoTLiveTurnAgent
        if args.dummy_llm:
            from structured_cot.llm_client import DummyStructuredLLM
            llm = DummyStructuredLLM()
        else:
            from structured_cot.llm_client import LLAMA_33_70B_DEFAULT, StructuredLLMClient
            mid = args.model_id or LLAMA_33_70B_DEFAULT
            llm = StructuredLLMClient(
                model_id=mid,
                default_max_tokens=args.max_new_tokens,
                default_temperature=args.temperature,
                seed=args.structured_cot_seed,
            )
        plog = (
            _Path(args.structured_cot_parse_log)
            if args.structured_cot_parse_log else None
        )
        return StructuredCoTLiveTurnAgent(
            llm,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            parse_log_path=plog,
            strategy_classifier=KeywordStrategyClassifier(),
        )

    raise ValueError(f"unknown agent type {args.agent!r}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logging(output_dir / "turn_eval.log")
    log.info("Args: %s", vars(args))

    data_path = Path(args.data)
    if not data_path.exists():
        log.error("data file not found: %s", data_path)
        return 2
    with data_path.open() as f:
        dialogues = json.load(f)
    if args.max_dialogues:
        dialogues = dialogues[: args.max_dialogues]
    log.info("Loaded %d dialogues from %s", len(dialogues), data_path)

    ann_lookup = _load_annotations_lookup(
        Path(args.annotations) if args.annotations else None
    )
    log.info(
        "Annotations: %d dialogues with strategy tags (from %s).",
        len(ann_lookup), args.annotations or "(dialogue-embedded)",
    )

    agent = _build_agent(args, output_dir=output_dir)
    log.info("Built agent: %s", type(agent).__name__)

    # Stamp dialogue_id onto every chat_logs turn so the replay adapter
    # can match turns without ambiguity. ``turn_level_eval`` also stamps
    # ``dialogue_id`` on history entries; this keeps JSON dialogues self-contained.
    if args.agent in ("structured_cot_replay", "structured_cot_live"):
        from structured_cot.replay_turn_agent import attach_dialogue_ids
        dialogues = attach_dialogue_ids(dialogues)

    records_path = output_dir / "turn_records.jsonl"
    if records_path.exists():
        records_path.unlink()
    rec_writer = records_path.open("a")

    def _on_record(r: TurnRecord) -> None:
        rec_writer.write(json.dumps(asdict(r), default=str) + "\n")

    perspectives = tuple(
        p.strip() for p in args.perspectives.split(",") if p.strip()
    )
    log.info("Perspectives scored: %s", perspectives)

    t0 = time.time()
    result = turn_level_eval(
        dialogues=dialogues,
        agent=agent,
        perspectives=perspectives,
        annotations_by_dialogue=ann_lookup or None,
        on_record=_on_record,
    )
    rec_writer.close()
    elapsed = time.time() - t0

    log.info("Eval finished in %.1fs", elapsed)
    log.info("\n%s", format_turn_level_summary(result))
    if hasattr(agent, "summary"):
        log.info("Agent summary: %s", agent.summary)

    summary_path = output_dir / "turn_summary.json"
    with summary_path.open("w") as f:
        json.dump(
            {
                "config":         vars(args),
                "n_dialogues":    len(dialogues),
                "n_records":      result["n_records"],
                "elapsed_seconds": elapsed,
                "accept":            result["accept"],
                "bid_cosine":        result["bid_cosine"],
                "strategy_macro_f1": result["strategy_macro_f1"],
                "brier":             result["brier"],
                "brier_by_turn_index": result["brier_by_turn_index"],
                "agent_summary": (
                    agent.summary if hasattr(agent, "summary") else None
                ),
            },
            f,
            indent=2,
        )
    log.info("Wrote %s", summary_path)
    log.info("Wrote %s (%d records)", records_path, result["n_records"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
