#!/usr/bin/env python3
"""
SFT/train.py — Training entry point for CaSiNo Llama-3.1-8B SFT.

Wires together:
  - SFT/client.py  : model + tokenizer loading (QLoRA + LoRA)
  - SFT/prompts.py : data loading + chat-template formatting

Usage (single GPU):
    python SFT/train.py

Usage (multi-GPU via torchrun):
    torchrun --nproc_per_node=4 SFT/train.py

LSF (Arion) single A100 80 GB:
    bsub -q gpu -n 8 -gpu "num=1:mode=exclusive_process" \\
         -R "rusage[mem=80000]" -R "select[gpu_model0==A100_SXM4_80GB]" \\
         python SFT/train.py

Extra packages required:
    pip install "trl>=0.9" "peft>=0.11" "bitsandbytes>=0.43" datasets
"""

from __future__ import annotations

import argparse
import os

from client import load_model_and_tokenizer, DEFAULT_MODEL_PATH
from collator import ResponseOnlyCollator
from prompts import load_casino_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SFT fine-tune Llama-3.1-8B-Instruct on CaSiNo dialogues."
    )
    # ── paths ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model_path",
        default=DEFAULT_MODEL_PATH,
        help="Local path or HuggingFace model ID.",
    )
    parser.add_argument(
        "--train_file",
        default="CaSiNo/data/split/casino_train.json",
    )
    parser.add_argument(
        "--output_dir",
        default="SFT/checkpoints",
        help="Directory to save LoRA adapter checkpoints.",
    )
    # ── model ──────────────────────────────────────────────────────────────
    parser.add_argument("--no_4bit", action="store_true",
                        help="Disable 4-bit QLoRA (use bf16 full LoRA instead).")
    parser.add_argument("--lora_r",       type=int,   default=16)
    parser.add_argument("--lora_alpha",   type=int,   default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    # ── training ───────────────────────────────────────────────────────────
    parser.add_argument("--max_length",                    type=int,   default=2048)
    parser.add_argument("--num_train_epochs",              type=int,   default=3)
    parser.add_argument("--per_device_train_batch_size",   type=int,   default=2)
    parser.add_argument("--gradient_accumulation_steps",   type=int,   default=4)
    parser.add_argument("--learning_rate",                 type=float, default=2e-4)
    parser.add_argument("--warmup_ratio",                  type=float, default=0.05)
    parser.add_argument("--lr_scheduler_type",             default="cosine")
    parser.add_argument("--logging_steps",                 type=int,   default=10)
    parser.add_argument("--save_steps",                    type=int,   default=200)
    parser.add_argument("--save_total_limit",              type=int,   default=3)
    parser.add_argument("--seed",                          type=int,   default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        raise ImportError(
            "trl is required: pip install 'trl>=0.9' peft bitsandbytes datasets"
        ) from e

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load model + tokenizer ──────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model_path,
        use_4bit=not args.no_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # ── 2. Load + format training data ────────────────────────────────────
    train_dataset = load_casino_dataset(
        args.train_file, tokenizer, max_length=args.max_length
    )

    # ── 3. Response-only collator ─────────────────────────────────────────
    # Masks system + user tokens with -100; loss only on assistant turns.
    collator = ResponseOnlyCollator(tokenizer, max_length=args.max_length)

    # ── 4. Training config ────────────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        dataset_text_field="text",
        max_length=args.max_length,
        packing=False,
        report_to="none",
    )

    # ── 5. Train ──────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    print(f"\nStarting SFT on {len(train_dataset)} examples …")
    trainer.train()

    # ── 5. Save final LoRA adapter ────────────────────────────────────────
    final_dir = os.path.join(args.output_dir, "final_adapter")
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nLoRA adapter saved to: {final_dir}")


if __name__ == "__main__":
    main()
