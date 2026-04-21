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
    p.add_argument("--agent", choices=("uniform", "hybrid", "sft"), default="uniform",
                   help="which TurnLevelAgent to evaluate.")

    # hybrid / sft shared LLM args
    p.add_argument("--dummy-llm", action="store_true",
                   help="use the deterministic dummy LLM (no GPU).")
    p.add_argument("--model-id", default=None,
                   help="HF model id / local snapshot path (hybrid agent).")

    # sft args
    p.add_argument("--base-model", default=None,
                   help="base model path for the SFT adapter (sft agent).")
    p.add_argument("--adapter", default=None,
                   help="LoRA adapter directory (sft agent).")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)

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


def _build_agent(args: argparse.Namespace) -> Any:
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

    agent = _build_agent(args)
    log.info("Built agent: %s", type(agent).__name__)

    records_path = output_dir / "turn_records.jsonl"
    if records_path.exists():
        records_path.unlink()
    rec_writer = records_path.open("a")

    def _on_record(r: TurnRecord) -> None:
        rec_writer.write(json.dumps(asdict(r), default=str) + "\n")

    t0 = time.time()
    result = turn_level_eval(
        dialogues=dialogues,
        agent=agent,
        annotations_by_dialogue=ann_lookup or None,
        on_record=_on_record,
    )
    rec_writer.close()
    elapsed = time.time() - t0

    log.info("Eval finished in %.1fs", elapsed)
    log.info("\n%s", format_turn_level_summary(result))

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
            },
            f,
            indent=2,
        )
    log.info("Wrote %s", summary_path)
    log.info("Wrote %s (%d records)", records_path, result["n_records"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
