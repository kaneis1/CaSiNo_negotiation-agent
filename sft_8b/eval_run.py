"""Evaluate the SFT'd 8B opponent model on a CaSiNo split.

Same `predictions.jsonl` + `summary.json` schema as the 70B hybrid run
(see ``opponent_model/eval_run.py``), with two extra columns per
prediction (``pred_satisfaction``, ``true_satisfaction``) and an extra
satisfaction-metrics block at the bottom of ``summary.json``.

Usage
-----
    # smoke run on 3 dialogues from valid
    python -m sft_8b.eval_run --max-dialogues 3

    # full run on the 30-dialogue valid split (default)
    python -m sft_8b.eval_run

    # zero-shot baseline (no LoRA adapter; same prompt/parser)
    python -m sft_8b.eval_run --no-adapter
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from opponent_model.metrics import (
    OpponentModelFn,
    evaluate_opponent_model,
    format_summary,
    summarize,
)
from sft_8b.metrics_satisfaction import (
    format_satisfaction_summary,
    summarize_satisfaction,
)


# ── Defaults ───────────────────────────────────────────────────────────────

DEFAULT_BASE_MODEL = (
    "/sc/arion/scratch/cuiz02/hf_cache/transformers/Meta-Llama-3.1-8B-Instruct"
)
DEFAULT_ADAPTER    = "sft_8b/results/lora_run/lora_best"
DEFAULT_DATA_PATH  = "CaSiNo/data/split/casino_valid.json"
DEFAULT_OUTPUT_DIR = "sft_8b/results/sft_eval"


# ── Logging ────────────────────────────────────────────────────────────────


def _setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sft_8b.eval_run")
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


# ── Checkpoint state ───────────────────────────────────────────────────────


@dataclass
class CheckpointState:
    completed_dialogue_ids: List[Any] = field(default_factory=list)
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(asdict(self), f)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "CheckpointState":
        with path.open() as f:
            data = json.load(f)
        return cls(**data)


def _bucket_from_predictions(
    predictions: List[Dict[str, Any]], max_k: int,
) -> Dict[int, Dict[str, List[float]]]:
    bucket = {k: {"ema": [], "top1": [], "ndcg": []} for k in range(1, max_k + 1)}
    for r in predictions:
        k = r.get("k")
        if k not in bucket:
            continue
        bucket[k]["ema"].append(r["ema"])
        bucket[k]["top1"].append(r["top1"])
        bucket[k]["ndcg"].append(r["ndcg"])
    return bucket


# ── Eval loop with checkpoint + satisfaction enrichment ───────────────────


def _build_sat_lookup(
    dialogues: Sequence[Dict[str, Any]],
) -> Dict[Any, Dict[str, Optional[str]]]:
    """{dialogue_id -> {perspective_role -> true satisfaction or None}}."""
    out: Dict[Any, Dict[str, Optional[str]]] = {}
    for i, d in enumerate(dialogues):
        did = d.get("dialogue_id", i)
        pinfo = d.get("participant_info", {})
        out[did] = {
            role: (pinfo.get(role, {}).get("outcomes", {}).get("satisfaction"))
            for role in ("mturk_agent_1", "mturk_agent_2")
        }
    return out


def run_eval(
    *,
    dialogues: List[Dict[str, Any]],
    sft_model: Any,                # SftModelFn instance (for last_satisfaction)
    output_dir: Path,
    checkpoint_every: int,
    summary_every: int,
    max_k: int,
    resume: bool,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    log_path = output_dir / "eval.log"
    ckpt_path = output_dir / "checkpoint.json"
    summary_path = output_dir / "summary.json"
    pred_path = output_dir / "predictions.jsonl"

    logger = _setup_logging(log_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    sat_lookup = _build_sat_lookup(dialogues)

    if resume and ckpt_path.exists():
        state = CheckpointState.load(ckpt_path)
        completed = set(state.completed_dialogue_ids)
        logger.info(
            "Resuming from checkpoint: %d dialogues completed, %d predictions.",
            len(completed), len(state.predictions),
        )
    else:
        if ckpt_path.exists():
            logger.info(
                "Existing checkpoint found but --resume not set; starting fresh."
            )
        state = CheckpointState(config=config)
        completed = set()
        if pred_path.exists():
            pred_path.unlink()

    pred_writer = pred_path.open("a")
    t_start = time.time()
    n_started = len(completed)

    try:
        for i, dialogue in enumerate(dialogues):
            did = dialogue.get("dialogue_id", i)
            if did in completed:
                continue

            t0 = time.time()
            new_predictions: List[Dict[str, Any]] = []

            def _enrich(rec: Dict[str, Any]) -> None:
                # Right after evaluate_opponent_model called sft_model(...), so
                # sft_model.last_satisfaction reflects THIS prediction.
                rec["pred_satisfaction"] = sft_model.last_satisfaction
                rec["true_satisfaction"] = (
                    sat_lookup.get(rec.get("dialogue_id"), {}).get(rec.get("perspective"))
                )
                rec["parse_flags"] = dict(getattr(sft_model, "last_flags", {}))
                new_predictions.append(rec)

            try:
                evaluate_opponent_model(
                    [dialogue],
                    sft_model,
                    max_k=max_k,
                    on_prediction=_enrich,
                )
            except Exception as e:
                logger.exception(
                    "Dialogue %s failed (%s); skipping.", did, type(e).__name__,
                )
                continue

            for rec in new_predictions:
                pred_writer.write(json.dumps(rec) + "\n")
            pred_writer.flush()

            state.predictions.extend(new_predictions)
            state.completed_dialogue_ids.append(did)
            completed.add(did)

            n_done_total = len(completed)
            n_done_session = n_done_total - n_started
            elapsed = time.time() - t_start
            rate = n_done_session / max(elapsed, 1e-6)
            remaining = len(dialogues) - n_done_total
            eta_min = remaining / max(rate, 1e-6) / 60 if rate > 0 else float("inf")

            logger.info(
                "[%d/%d] dialogue %s done in %.1fs  (%.2f dial/s, ETA %.1f min)",
                n_done_total, len(dialogues), did, time.time() - t0,
                rate, eta_min,
            )

            if n_done_total % summary_every == 0:
                bucket = _bucket_from_predictions(state.predictions, max_k)
                pref_summ = summarize(bucket, max_k=max_k)
                sat_summ = summarize_satisfaction(state.predictions, max_k=max_k)
                logger.info(
                    "  running summary @ %d dialogues:\n%s\n%s",
                    n_done_total,
                    format_summary(pref_summ),
                    format_satisfaction_summary(sat_summ),
                )

            if n_done_total % checkpoint_every == 0:
                state.save(ckpt_path)
                logger.info("  checkpoint saved -> %s", ckpt_path)

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt — saving checkpoint before exit.")
    finally:
        pred_writer.close()
        state.save(ckpt_path)

    bucket = _bucket_from_predictions(state.predictions, max_k)
    pref_final = summarize(bucket, predictions=state.predictions, max_k=max_k)
    sat_final = summarize_satisfaction(state.predictions, max_k=max_k)

    with summary_path.open("w") as f:
        json.dump(
            {
                "config":         config,
                "n_dialogues":    len(state.completed_dialogue_ids),
                "n_predictions":  len(state.predictions),
                "elapsed_seconds": time.time() - state.started_at,
                "per_k_means":    pref_final["per_k_means"],
                "per_k_counts":   pref_final["per_k_counts"],
                "kpenalty":       pref_final["kpenalty"],
                "summary":        pref_final["summary"],
                "satisfaction": {
                    "per_k_means":  sat_final["per_k_means"],
                    "per_k_counts": sat_final["per_k_counts"],
                    "kpenalty":     sat_final["kpenalty"],
                    "summary":      sat_final["summary"],
                },
            },
            f, indent=2,
        )

    logger.info("Final preferences summary:\n%s", format_summary(pref_final))
    logger.info("Final satisfaction summary:\n%s", format_satisfaction_summary(sat_final))
    logger.info("Wrote %s", summary_path)
    return {"prefs": pref_final, "satisfaction": sat_final}


# ── CLI ────────────────────────────────────────────────────────────────────


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",        default=DEFAULT_DATA_PATH,
                   help="CaSiNo split JSON to evaluate on. "
                        "Per the user's split: valid for final reporting.")
    p.add_argument("--output-dir",  default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--base-model",  default=DEFAULT_BASE_MODEL)
    p.add_argument("--adapter",     default=DEFAULT_ADAPTER,
                   help="Path to LoRA adapter dir (lora_best/).")
    p.add_argument("--no-adapter", action="store_true",
                   help="Run zero-shot against the base 8B (skip LoRA).")
    p.add_argument("--max-dialogues", type=int, default=None)
    p.add_argument("--max-k", type=int, default=5)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--summary-every", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.0)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "adapter": None if args.no_adapter else args.adapter,
    }

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}", file=sys.stderr)
        return 2
    with data_path.open() as f:
        dialogues = json.load(f)
    if args.max_dialogues:
        dialogues = dialogues[: args.max_dialogues]

    from sft_8b.predict import SftModelFn

    sft_model = SftModelFn(
        base_model=args.base_model,
        adapter_path=None if args.no_adapter else args.adapter,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    run_eval(
        dialogues=dialogues,
        sft_model=sft_model,
        output_dir=output_dir,
        checkpoint_every=args.checkpoint_every,
        summary_every=args.summary_every,
        max_k=args.max_k,
        resume=args.resume,
        config=config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
