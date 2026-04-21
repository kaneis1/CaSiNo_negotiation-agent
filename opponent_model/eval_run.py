"""Run the Hybrid Bayesian opponent model on CaSiNo and report
EMA / top1 / NDCG@3 at k = 1..5 + k-penalty.

Designed for long Llama-3.3-70B runs:
    * Disk-cached LLM calls (re-runs of the same (prompt, model_id) are free).
    * Checkpoint every N dialogues; --resume picks up where the last run left off.
    * Periodic running summary prints — easy to ``tail -f`` the log file.

Usage (single GPU node, after activating the casino env):
    python -m opponent_model.eval_run \\
        --data       CaSiNo/data/casino.json \\
        --output-dir results/hybrid_eval \\
        --model-id   /sc/arion/scratch/cuiz02/hf_cache/transformers/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b \\
        --max-dialogues 50            # smoke run
    python -m opponent_model.eval_run --resume     # continue an interrupted run
    python -m opponent_model.eval_run --dummy-llm  # CPU-only sanity check
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

import numpy as np

from opponent_model import CachedLLM, HybridAgent
from opponent_model.metrics import (
    OpponentModelFn,
    evaluate_opponent_model,
    format_summary,
    get_ordering,
    summarize,
)

# Default Llama-3.3-70B local snapshot on the cluster.
DEFAULT_MODEL_ID = (
    "/sc/arion/scratch/cuiz02/hf_cache/transformers/"
    "models--meta-llama--Llama-3.3-70B-Instruct/snapshots/"
    "6f6073b423013f6a7d4d9f39144961bfbfbc386b"
)
DEFAULT_DATA_PATH = "CaSiNo/data/casino.json"
DEFAULT_OUTPUT_DIR = "opponent_model/results/hybrid_eval"


# ── Logging setup ──────────────────────────────────────────────────────────


def _setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("opponent_model.eval_run")
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


# ── Adapter: HybridAgent → OpponentModelFn ─────────────────────────────────


def make_hybrid_model_fn(
    llm_client: Any,
    *,
    likelihood_temperature: float = 25.0,
    likelihood_clip: tuple[Optional[float], Optional[float]] = (-3.0, 3.0),
    strict_likelihood: bool = False,
) -> OpponentModelFn:
    """Build a stateless ``opponent_model_fn`` backed by the HybridAgent.

    The metric harness calls this once per (dialogue, perspective, k). We
    spin up a fresh agent per call and feed every opponent utterance in
    the partial. With caching, repeated work across k=1..5 collapses
    onto the same prompts, so cost is ~1 LLM call per unique opponent
    utterance per perspective.
    """

    def fn(
        partial: List[Dict[str, Any]],
        perspective_priorities: Dict[str, str],
        opp_role: str,
        my_role: str,
        my_reasons: Dict[str, str],
    ) -> Sequence[str]:
        agent = HybridAgent(
            my_priorities=perspective_priorities,
            llm_client=llm_client,
            likelihood_temperature=likelihood_temperature,
            likelihood_clip=likelihood_clip,
            strict_likelihood=strict_likelihood,
        )
        for turn in partial:
            text = turn.get("text", "")
            if not text:
                continue
            if turn["id"] == opp_role:
                agent.observe(text)
            else:
                agent.history.append({"role": "me", "text": text})
        idx = int(np.argmax(agent.log_posterior))
        return list(agent.hypotheses[idx])

    return fn


# ── Checkpointing ──────────────────────────────────────────────────────────


@dataclass
class CheckpointState:
    """All the per-prediction state needed to resume an interrupted run."""

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
        k = r["k"]
        if k not in bucket:
            continue
        bucket[k]["ema"].append(r["ema"])
        bucket[k]["top1"].append(r["top1"])
        bucket[k]["ndcg"].append(r["ndcg"])
    return bucket


# ── Main eval loop with checkpointing ──────────────────────────────────────


def run_eval(
    *,
    dialogues: List[Dict[str, Any]],
    model_fn: OpponentModelFn,
    output_dir: Path,
    checkpoint_every: int,
    summary_every: int,
    max_k: int,
    cached_client: Optional[CachedLLM],
    resume: bool,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    log_path = output_dir / "eval.log"
    ckpt_path = output_dir / "checkpoint.json"
    summary_path = output_dir / "summary.json"
    pred_path = output_dir / "predictions.jsonl"

    logger = _setup_logging(log_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if resume and ckpt_path.exists():
        state = CheckpointState.load(ckpt_path)
        completed = set(state.completed_dialogue_ids)
        logger.info(
            "Resuming from checkpoint: %d dialogues already completed, %d predictions.",
            len(completed), len(state.predictions),
        )
    else:
        if ckpt_path.exists():
            logger.info("Existing checkpoint found but --resume not set; starting fresh.")
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
            try:
                evaluate_opponent_model(
                    [dialogue],
                    model_fn,
                    max_k=max_k,
                    on_prediction=lambda r: new_predictions.append(r),
                )
            except Exception as e:
                logger.exception(
                    "Dialogue %s failed (%s); skipping. (Set strict_likelihood=False "
                    "to silence prompt issues.)", did, type(e).__name__,
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

            if cached_client is not None and n_done_session % 10 == 0:
                stats = cached_client.stats()
                logger.info(
                    "  llm cache: %d hits / %d misses (rate %.2f%%, size %d)",
                    stats["hits"], stats["misses"],
                    100 * stats["hit_rate"], stats["size"],
                )

            if n_done_total % summary_every == 0:
                bucket = _bucket_from_predictions(state.predictions, max_k)
                summ = summarize(bucket, max_k=max_k)
                logger.info("  running summary @ %d dialogues:\n%s",
                            n_done_total, format_summary(summ))

            if n_done_total % checkpoint_every == 0:
                state.save(ckpt_path)
                logger.info("  checkpoint saved → %s", ckpt_path)

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt — saving checkpoint before exit.")
    finally:
        pred_writer.close()
        state.save(ckpt_path)

    bucket = _bucket_from_predictions(state.predictions, max_k)
    final = summarize(bucket, predictions=state.predictions, max_k=max_k)

    with summary_path.open("w") as f:
        json.dump(
            {
                "config": config,
                "n_dialogues": len(state.completed_dialogue_ids),
                "n_predictions": len(state.predictions),
                "elapsed_seconds": time.time() - state.started_at,
                "per_k_means": final["per_k_means"],
                "per_k_counts": final["per_k_counts"],
                "kpenalty": final["kpenalty"],
                "summary": final["summary"],
            },
            f,
            indent=2,
        )

    logger.info("Final summary:\n%s", format_summary(final))
    logger.info("Wrote %s", summary_path)
    return final


# ── CLI ────────────────────────────────────────────────────────────────────


def _parse_clip(s: str) -> tuple[Optional[float], Optional[float]]:
    if s.lower() in ("none", "off", ""):
        return (None, None)
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "--likelihood-clip must be 'lo,hi' or 'none' (got %r)" % s
        )
    return (float(parts[0]), float(parts[1]))


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=DEFAULT_DATA_PATH)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                   help="HF model id or local snapshot path.")
    p.add_argument("--max-dialogues", type=int, default=None,
                   help="Cap dialogues processed (debug / smoke).")
    p.add_argument("--max-k", type=int, default=5)
    p.add_argument("--checkpoint-every", type=int, default=10,
                   help="Save the checkpoint every N completed dialogues.")
    p.add_argument("--summary-every", type=int, default=25,
                   help="Print a running metric summary every N dialogues.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from output_dir/checkpoint.json if present.")
    p.add_argument("--dummy-llm", action="store_true",
                   help="Use a deterministic dummy LLM (no GPU). For pipeline tests only.")
    p.add_argument("--no-cache", action="store_true",
                   help="Disable the disk LLM cache.")
    p.add_argument("--cache-path", default=None,
                   help="Override cache file (default: <output_dir>/llm_cache.sqlite).")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="LLM sampling temperature (0 = greedy).")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--likelihood-temperature", type=float, default=25.0)
    p.add_argument("--likelihood-clip", type=_parse_clip, default=(-3.0, 3.0),
                   help="Pass 'lo,hi' (e.g. '-3,3') or 'none' to disable.")
    return p


def _build_dummy_llm() -> Any:
    """Tiny canned LLM for pipeline smoke tests (no GPU required)."""
    import json as _json
    from opponent_model.hypotheses import HYPOTHESES, hypothesis_label

    class DummyLLM:
        model_id = "dummy"
        temperature = 0.0
        top_p = 1.0
        max_new_tokens = 64

        def generate(self, prompt: str) -> str:
            low = prompt.lower()
            if "evidence model" in low:
                scores: Dict[str, float] = {hypothesis_label(i): 50.0 for i in range(6)}
                if "water" in low and ("need water" in low or "thirst" in low):
                    for i, h in enumerate(HYPOTHESES):
                        scores[hypothesis_label(i)] = (
                            70.0 if h[0] == "Water" else 40.0 if h[1] == "Water" else 25.0
                        )
                if "firewood" in low and ("need firewood" in low or "cold" in low or "warm" in low):
                    for i, h in enumerate(HYPOTHESES):
                        scores[hypothesis_label(i)] = (
                            70.0 if h[0] == "Firewood" else 40.0 if h[1] == "Firewood" else 25.0
                        )
                if "food" in low and ("need food" in low or "hungry" in low or "feed" in low):
                    for i, h in enumerate(HYPOTHESES):
                        scores[hypothesis_label(i)] = (
                            70.0 if h[0] == "Food" else 40.0 if h[1] == "Food" else 25.0
                        )
                return _json.dumps({"evidence_scores": scores, "short_rationale": "dummy"})
            return _json.dumps({"utterance": "(dummy)", "offer": None})

    return DummyLLM()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "likelihood_clip": list(args.likelihood_clip)}

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}", file=sys.stderr)
        return 2
    with data_path.open() as f:
        dialogues = json.load(f)
    if args.max_dialogues:
        dialogues = dialogues[: args.max_dialogues]

    if args.dummy_llm:
        raw_client = _build_dummy_llm()
    else:
        from prompt_engineer.llm.client import LlamaClient

        raw_client = LlamaClient(
            model_id=args.model_id,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )

    cached_client: Optional[CachedLLM] = None
    if args.no_cache:
        client = raw_client
    else:
        cache_path = Path(args.cache_path) if args.cache_path else output_dir / "llm_cache.sqlite"
        cached_client = CachedLLM(raw_client, cache_path=cache_path)
        client = cached_client

    model_fn = make_hybrid_model_fn(
        client,
        likelihood_temperature=args.likelihood_temperature,
        likelihood_clip=args.likelihood_clip,
        strict_likelihood=False,
    )

    run_eval(
        dialogues=dialogues,
        model_fn=model_fn,
        output_dir=output_dir,
        checkpoint_every=args.checkpoint_every,
        summary_every=args.summary_every,
        max_k=args.max_k,
        cached_client=cached_client,
        resume=args.resume,
        config=config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
