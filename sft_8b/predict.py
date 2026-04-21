"""Load the SFT'd 8B + LoRA adapter and expose it as an ``OpponentModelFn``.

Same prompt builder is used as at training time (see ``sft_8b/prompts.py``).
The model emits a JSON object; we parse it with safe fallbacks so a single
malformed generation never derails a long eval run.

Two outputs per prediction:
    * ``ordering`` (returned to satisfy ``OpponentModelFn``): the
      predicted [top, mid, low] over Food/Water/Firewood.
    * ``last_satisfaction`` (stashed on the callable instance): the
      predicted satisfaction label, picked up by ``sft_8b/eval_run.py``
      via the prediction callback.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from sft_8b.prompts import (
    ITEMS,
    SATISFACTION_LABELS,
    SYSTEM_PROMPT,
    build_user_prompt,
)

logger = logging.getLogger("sft_8b.predict")


# Mode for satisfaction in the train split — used as a soft fallback when the
# model output is unparseable. (5/4/3/2/1 -> middle index = "Undecided".)
DEFAULT_SATISFACTION_FALLBACK = "Slightly satisfied"
DEFAULT_PREFS_FALLBACK = ["Food", "Water", "Firewood"]


# ── JSON parsing ────────────────────────────────────────────────────────────


_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_response(
    text: str,
    *,
    on_warning: Optional[callable] = None,
) -> Tuple[List[str], Optional[str], Dict[str, bool]]:
    """Best-effort parse of the model's JSON response.

    Returns ``(prefs, satisfaction, flags)`` where ``flags`` records which
    fallbacks (if any) we had to apply. ``satisfaction`` may be ``None``
    if even the fallback couldn't be inferred.
    """
    flags = {"json_malformed": False, "prefs_malformed": False, "sat_malformed": False}

    def _warn(msg: str) -> None:
        if on_warning is not None:
            on_warning(msg)
        else:
            logger.warning(msg)

    obj: Optional[Dict[str, Any]] = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if m is not None:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

    if not isinstance(obj, dict):
        flags["json_malformed"] = True
        flags["prefs_malformed"] = True
        flags["sat_malformed"] = True
        _warn(f"unparseable LLM JSON: {text[:120]!r}")
        return list(DEFAULT_PREFS_FALLBACK), DEFAULT_SATISFACTION_FALLBACK, flags

    raw_prefs = obj.get("prefs")
    prefs = _coerce_prefs(raw_prefs)
    if prefs is None:
        flags["prefs_malformed"] = True
        _warn(f"prefs not a permutation of {ITEMS}: {raw_prefs!r}")
        prefs = list(DEFAULT_PREFS_FALLBACK)

    raw_sat = obj.get("satisfaction")
    sat = raw_sat if raw_sat in SATISFACTION_LABELS else None
    if sat is None:
        flags["sat_malformed"] = True
        _warn(f"satisfaction not in label set: {raw_sat!r}")
        sat = DEFAULT_SATISFACTION_FALLBACK

    return prefs, sat, flags


def _coerce_prefs(raw: Any) -> Optional[List[str]]:
    if not isinstance(raw, list) or len(raw) != 3:
        return None
    cleaned = [str(x).strip().capitalize() for x in raw]
    if set(cleaned) != set(ITEMS):
        return None
    return cleaned


# ── Model loading + inference ──────────────────────────────────────────────


class SftModelFn:
    """Callable adapter: ``OpponentModelFn`` interface backed by an SFT'd 8B.

    The instance is reused across all (dialogue, perspective, k) calls so
    the model is loaded once. ``last_satisfaction`` and ``last_flags``
    expose the most recent predicted satisfaction + parse flags so the
    eval harness can persist them alongside each prediction record.
    """

    def __init__(
        self,
        *,
        base_model: str,
        adapter_path: Optional[str],
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        device_map: str = "auto",
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.base_model = base_model
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.last_satisfaction: Optional[str] = None
        self.last_flags: Dict[str, bool] = {}
        self.last_raw_response: str = ""

        logger.info("Loading base model: %s", base_model)
        tok_src = adapter_path if adapter_path else base_model
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation="sdpa",
        )

        if adapter_path:
            from peft import PeftModel

            logger.info("Attaching LoRA adapter: %s", adapter_path)
            # `autocast_adapter_dtype=False` short-circuits peft's
            # `cast_adapter_dtype`, which on torch < 2.7 crashes with
            # `AttributeError: module 'torch' has no attribute 'float8_e8m0fnu'`.
            # The adapter weights stay in their saved dtype (bf16), which is
            # what we want for inference anyway.
            model = PeftModel.from_pretrained(
                model, adapter_path, autocast_adapter_dtype=False,
            )
            model = model.eval()
        else:
            logger.warning(
                "No adapter_path supplied — running ZERO-SHOT against the base "
                "model. Useful as a baseline; not the SFT'd numbers."
            )

        model.config.use_cache = True
        self.model = model

    # ── OpponentModelFn signature ───────────────────────────────────────

    def __call__(
        self,
        partial: List[Mapping[str, Any]],
        my_priorities: Mapping[str, str],
        opp_role: str,
        my_role: str,
        my_reasons: Mapping[str, str],
    ) -> Sequence[str]:
        prompt = build_user_prompt(
            partial=partial,
            my_priorities=my_priorities,
            my_reasons=my_reasons,
            me_role=my_role,
        )
        response = self._generate_chat(prompt)
        self.last_raw_response = response

        prefs, sat, flags = parse_response(response)
        self.last_satisfaction = sat
        self.last_flags = flags
        return prefs

    # ── Internals ───────────────────────────────────────────────────────

    def _generate_chat(self, user_prompt: str) -> str:
        messages = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": user_prompt},
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

        new_tokens = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


__all__ = ["SftModelFn", "parse_response", "DEFAULT_SATISFACTION_FALLBACK"]
