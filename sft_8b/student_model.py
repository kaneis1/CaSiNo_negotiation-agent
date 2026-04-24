"""Inference wrapper for the Day 8 distilled negotiation student."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

import torch

from sft_8b.build_distill_data import (
    format_history,
    format_priorities,
    format_reasons,
)
from sft_8b.model_utils import choose_inference_dtype
from sft_8b.student_parser import parse_student_response
from sft_8b.student_prompts import (
    STUDENT_SYSTEM_PROMPT,
    build_student_user_prompt,
)

logger = logging.getLogger("sft_8b.student_model")


class StudentModelFn:
    """Load the Day 8 LoRA student and emit parsed tagged generations."""

    def __init__(
        self,
        *,
        base_model: str,
        adapter_path: Optional[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        device_map: str = "auto",
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.base_model = base_model
        self.adapter_path = adapter_path
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.last_raw_response: str = ""
        self.last_parse: Dict[str, Any] = {}

        tok_src = adapter_path if adapter_path else base_model
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = choose_inference_dtype()
        logger.info("Loading student base model: %s (dtype=%s)", base_model, dtype)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation="sdpa",
        )

        if adapter_path:
            from peft import PeftModel

            logger.info("Attaching student LoRA adapter: %s", adapter_path)
            model = PeftModel.from_pretrained(
                model,
                adapter_path,
                autocast_adapter_dtype=False,
            )
        else:
            logger.warning(
                "No student adapter_path supplied; running zero-shot against the base model."
            )

        model = model.eval()
        model.config.use_cache = True
        self.model = model

    def predict(
        self,
        *,
        history: List[Mapping[str, Any]],
        my_role: str,
        my_priorities: Mapping[str, str],
        my_reasons: Mapping[str, str],
        style: str,
    ) -> Dict[str, Any]:
        user_prompt = build_student_user_prompt(
            self_priorities=format_priorities(my_priorities),
            self_reasons=format_reasons(my_reasons),
            history=format_history(history, perspective=my_role),
            style=style,
        )
        raw = self._generate_chat(user_prompt)
        self.last_raw_response = raw
        parsed = parse_student_response(raw)
        self.last_parse = parsed
        return parsed

    def _generate_chat(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        attention_mask = torch.ones_like(input_ids)

        do_sample = self.temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = self.temperature

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        new_tokens = output_ids[0, input_ids.shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


__all__ = ["StudentModelFn"]
