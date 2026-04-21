"""LoRA SFT for Llama-3.1-8B-Instruct on CaSiNo opponent + satisfaction.

Single-GPU bf16 LoRA fine-tune via TRL's ``SFTTrainer`` + PEFT's
``LoraConfig``. Designed for 1x H100/A100-80GB. With max_seq_length=2048
and effective batch size 16 (per_device=4 x grad_accum=4) the full 9k
training rows take about 90 minutes for 3 epochs on an H100-NVL.

Inputs (built by ``sft_8b.data``):
    sft_8b/results/sft_data/sft_train_rows.jsonl  (~9k rows)
    sft_8b/results/sft_data/sft_test_rows.jsonl   (~1k rows, in-training eval)

Outputs:
    sft_8b/results/lora_run/                      (trainer working dir)
    sft_8b/results/lora_run/lora_best/            (best adapter, by eval_loss)
    sft_8b/results/lora_run/training_args.json
    sft_8b/results/lora_run/trainer_state.json    (TRL-emitted history)

Inference picks up the adapter at ``lora_best/`` (see
``sft_8b/predict.py``); base weights are NOT merged or saved here.

Usage
-----
    # smoke test on a handful of steps with the unsplit data
    python -m sft_8b.train --max-steps 10 --num-train-epochs 1

    # full run
    python -m sft_8b.train
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger("sft_8b.train")


# ── Defaults (mirrored as CLI flags below) ─────────────────────────────────

DEFAULT_BASE_MODEL = (
    "/sc/arion/scratch/cuiz02/hf_cache/transformers/Meta-Llama-3.1-8B-Instruct"
)
DEFAULT_DATA_DIR   = Path("sft_8b/results/sft_data")
DEFAULT_OUTPUT_DIR = Path("sft_8b/results/lora_run")
DEFAULT_BEST_DIR   = "lora_best"

DEFAULT_LORA_R           = 16
DEFAULT_LORA_ALPHA       = 32
DEFAULT_LORA_DROPOUT     = 0.05
DEFAULT_TARGET_MODULES   = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)
DEFAULT_LR               = 1e-4
DEFAULT_WARMUP_RATIO     = 0.03
DEFAULT_NUM_EPOCHS       = 3
DEFAULT_PER_DEVICE_BS    = 4
DEFAULT_GRAD_ACCUM       = 4
DEFAULT_MAX_SEQ_LEN      = 2048
DEFAULT_SEED             = 42


# ── Dataset loading ────────────────────────────────────────────────────────


def _load_chat_jsonl(path):
    from datasets import Dataset

    path = Path(path)
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({"messages": obj["messages"]})
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return Dataset.from_list(rows)


# ── Main ───────────────────────────────────────────────────────────────────


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = Path(args.train_file)
    eval_path  = Path(args.eval_file)
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training file not found: {train_path}. "
            f"Run `python -m sft_8b.data` first."
        )

    # Heavy imports deferred so --help / arg validation are instant.
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    logger.info("Loading tokenizer/model: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        # Llama-3 ships without a pad token; reuse eos so collator doesn't crash.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=list(args.target_modules),
        task_type="CAUSAL_LM",
        bias="none",
    )

    # Wrap with LoRA ourselves so we can pass `autocast_adapter_dtype=False`.
    # peft 0.19 + torch < 2.7 crashes inside `cast_adapter_dtype` because it
    # blindly does `getattr(torch, "float8_e8m0fnu")` (introduced in torch 2.7).
    # Disabling autocast short-circuits that buggy code path; adapter weights
    # stay in their natural dtype, which is fine for bf16 LoRA training.
    # If we instead let TRL call `get_peft_model(...)` internally it would use
    # the default `autocast_adapter_dtype=True` and crash.
    model = get_peft_model(model, lora_cfg, autocast_adapter_dtype=False)
    model.print_trainable_parameters()

    logger.info("Loading SFT data:")
    logger.info("  train: %s", train_path)
    logger.info("  eval : %s", eval_path)
    train_ds = _load_chat_jsonl(train_path)
    eval_ds = _load_chat_jsonl(eval_path) if eval_path.exists() else None
    logger.info("  train rows: %d", len(train_ds))
    if eval_ds is not None:
        logger.info("  eval  rows: %d", len(eval_ds))
    else:
        logger.warning(
            "Eval file not found at %s; training without in-loop eval.", eval_path,
        )

    sft_cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        # TRL 1.x renamed `max_seq_length` -> `max_length`. Same semantics:
        # sequences longer than this are truncated; shorter ones are padded
        # to this length within the batch. Our data tops out at 1242 tokens
        # (verified) so anything >= 1280 is safe.
        max_length=args.max_seq_length,
        packing=False,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        # NOTE: `load_best_model_at_end=False` because peft 0.19 + torch < 2.7
        # crashes inside `model.load_adapter(...)` (same `float8_e8m0fnu` bug
        # we worked around above). The best checkpoint is still tracked in
        # `trainer_state.json` (`best_model_checkpoint`), and we copy it to
        # `lora_best/` ourselves below.
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss" if eval_ds is not None else None,
        greater_is_better=False if eval_ds is not None else None,
        save_total_limit=4,  # keep all 3 epoch checkpoints + a buffer
        report_to="none",
        seed=args.seed,
        max_steps=args.max_steps if args.max_steps is not None else -1,
    )

    with (output_dir / "training_args.json").open("w") as f:
        json.dump(
            {
                **{k: v for k, v in vars(args).items()},
                "trl_sft_config": {
                    k: v for k, v in asdict(sft_cfg).items()
                    if isinstance(v, (int, float, str, bool, list, dict, type(None)))
                },
            },
            f, indent=2, default=str,
        )

    # NOTE: do NOT pass `peft_config=` here. We already wrapped the model with
    # LoRA above (with autocast disabled). Passing peft_config would make TRL
    # call `get_peft_model(model, peft_config)` again with default kwargs and
    # re-trigger the torch.float8_e8m0fnu AttributeError on torch < 2.7.
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    logger.info("Starting training. Effective batch size = %d", (
        args.per_device_train_batch_size * args.gradient_accumulation_steps
    ))
    trainer.train()

    # We disabled `load_best_model_at_end` (see SFTConfig comment), so the
    # in-memory model is whatever the last epoch produced — NOT necessarily the
    # best one. Find the best checkpoint that the trainer recorded and copy
    # its files into `lora_best/` so downstream code (predict.py / eval_run.py)
    # can find a stable path.
    import shutil
    best_dir = output_dir / args.best_subdir
    best_src = trainer.state.best_model_checkpoint
    if best_src is None:
        # No eval ran (eval_ds was None); fall back to the latest checkpoint.
        logger.warning(
            "No best_model_checkpoint recorded; using current model state."
        )
        trainer.save_model(str(best_dir))
    else:
        best_src_path = Path(best_src)
        logger.info(
            "Best checkpoint = %s (%s = %.4f). Copying -> %s",
            best_src_path.name,
            sft_cfg.metric_for_best_model,
            trainer.state.best_metric,
            best_dir,
        )
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(best_src_path, best_dir)
    tokenizer.save_pretrained(str(best_dir))

    # Persist the run config alongside the adapter for reproducibility.
    with (best_dir / "sft_run_meta.json").open("w") as f:
        json.dump({
            "base_model":   args.base_model,
            "train_file":   str(train_path),
            "eval_file":    str(eval_path) if eval_ds is not None else None,
            "n_train_rows": len(train_ds),
            "n_eval_rows":  len(eval_ds) if eval_ds is not None else 0,
            "best_checkpoint":     trainer.state.best_model_checkpoint,
            "best_metric_name":    sft_cfg.metric_for_best_model,
            "best_metric_value":   trainer.state.best_metric,
            "lora": {
                "r":              args.lora_r,
                "alpha":          args.lora_alpha,
                "dropout":        args.lora_dropout,
                "target_modules": list(args.target_modules),
            },
        }, f, indent=2)

    logger.info("Training complete.")
    return 0


# ── CLI ────────────────────────────────────────────────────────────────────


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model",   default=DEFAULT_BASE_MODEL)
    p.add_argument("--train-file",   default=str(DEFAULT_DATA_DIR / "sft_train_rows.jsonl"))
    p.add_argument("--eval-file",    default=str(DEFAULT_DATA_DIR / "sft_test_rows.jsonl"),
                   help="Used as in-training eval set for early stopping. "
                        "Per the user's split: casino_test (100 dialogues) -> ~1k rows.")
    p.add_argument("--output-dir",   type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--best-subdir",  default=DEFAULT_BEST_DIR)

    p.add_argument("--lora-r",       type=int,   default=DEFAULT_LORA_R)
    p.add_argument("--lora-alpha",   type=int,   default=DEFAULT_LORA_ALPHA)
    p.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    p.add_argument("--target-modules", nargs="+", default=list(DEFAULT_TARGET_MODULES))

    p.add_argument("--learning-rate",                type=float, default=DEFAULT_LR)
    p.add_argument("--warmup-ratio",                 type=float, default=DEFAULT_WARMUP_RATIO)
    p.add_argument("--num-train-epochs",             type=float, default=DEFAULT_NUM_EPOCHS)
    p.add_argument("--per-device-train-batch-size",  type=int,   default=DEFAULT_PER_DEVICE_BS)
    p.add_argument("--per-device-eval-batch-size",   type=int,   default=DEFAULT_PER_DEVICE_BS)
    p.add_argument("--gradient-accumulation-steps",  type=int,   default=DEFAULT_GRAD_ACCUM)
    p.add_argument("--max-seq-length",               type=int,   default=DEFAULT_MAX_SEQ_LEN)
    p.add_argument("--seed",                         type=int,   default=DEFAULT_SEED)
    p.add_argument("--max-steps", type=int, default=None,
                   help="Cap optimizer steps (smoke testing).")
    return p


if __name__ == "__main__":
    raise SystemExit(main())
