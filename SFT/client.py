#!/usr/bin/env python3
"""
SFT/client.py — Llama model + tokenizer loader for SFT training.

Loads Meta-Llama-3.1-8B-Instruct with optional 4-bit QLoRA quantization
and wraps it with a LoRA adapter ready for fine-tuning.
"""

from __future__ import annotations

import torch

DEFAULT_MODEL_PATH = (
    "/sc/arion/scratch/cuiz02/hf_cache/transformers/Meta-Llama-3.1-8B-Instruct"
)

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def load_tokenizer(model_path: str = DEFAULT_MODEL_PATH):
    """Load and configure the Llama tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Right-padding required for DataCollatorForCompletionOnlyLM
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(
    model_path: str = DEFAULT_MODEL_PATH,
    use_4bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    """
    Load Llama with LoRA (and optional 4-bit NF4 quantization).

    Args:
        model_path:    Local path or HuggingFace model ID.
        use_4bit:      Enable QLoRA (bitsandbytes NF4). Recommended for A100 40 GB.
        lora_r:        LoRA rank.
        lora_alpha:    LoRA scaling factor.
        lora_dropout:  Dropout on LoRA layers.

    Returns:
        PEFT-wrapped model with trainable LoRA weights.
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # autocast_adapter_dtype=False: prevents peft from trying to cast to
    # float8_e8m0fnu, which requires torch >= 2.6 (installed: 2.5.x)
    model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
    model.print_trainable_parameters()
    return model


def load_model_and_tokenizer(
    model_path: str = DEFAULT_MODEL_PATH,
    use_4bit: bool = True,
    **lora_kwargs,
):
    """Convenience wrapper: returns (model, tokenizer)."""
    tokenizer = load_tokenizer(model_path)
    model = load_model(model_path, use_4bit=use_4bit, **lora_kwargs)
    return model, tokenizer
