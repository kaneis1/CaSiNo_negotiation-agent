"""SFT pipeline for fine-tuning Llama-3.1-8B-Instruct on CaSiNo opponent
modeling + satisfaction prediction.

Public surface kept tiny on purpose; everything lives in module-level
files that can be invoked via ``python -m sft_8b.<module>``.
"""

from sft_8b.prompts import (
    SYSTEM_PROMPT,
    SATISFACTION_LABELS,
    build_user_prompt,
    build_target_json,
)

__all__ = [
    "SYSTEM_PROMPT",
    "SATISFACTION_LABELS",
    "build_user_prompt",
    "build_target_json",
]
