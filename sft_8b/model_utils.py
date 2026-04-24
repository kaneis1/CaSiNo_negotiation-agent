"""Shared utilities for SFT / student model loading."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger("sft_8b.model_utils")


def choose_inference_dtype() -> torch.dtype:
    """Prefer bf16 when available, otherwise fall back to a safe dtype."""
    if torch.cuda.is_available():
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            logger.warning("Could not query bf16 support; falling back to fp16.")
        return torch.float16
    return torch.float32


__all__ = ["choose_inference_dtype"]
