"""Thin LLM wrappers used by CaSiNo belief experiments.

The blueprint expects ``llm.generate(prompt, max_tokens=800, temperature=0.3)``
to work. ``StructuredLLMClient`` wraps ``LlamaClient`` and forwards per-call
overrides through a small re-configuration.

For CPU-only smoke tests we also provide ``DummyStructuredLLM`` which
emits a syntactically valid five-block response without any model.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger("casino_belief.llm.llm_client")


LLAMA_33_70B_DEFAULT = (
    ""
    "models--meta-llama--Llama-3.3-70B-Instruct/snapshots/"
    "6f6073b423013f6a7d4d9f39144961bfbfbc386b"
)

LLAMA_31_8B_DEFAULT = (
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
)


class LlamaClient:
    """Thin wrapper around a chat-style Llama Instruct pipeline."""

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.3-70B-Instruct",
        system_prompt: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        device_map: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device_map = device_map

        import torch
        import transformers

        logger.info("Loading text-generation pipeline: %s", model_id)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map=device_map,
        )
        logger.info("Pipeline ready.")

    def generate(self, prompt: str) -> str:
        """Run inference on a fully formatted prompt string."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        do_sample = self.temperature > 0.0
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p

        outputs = self.pipeline(messages, **generation_kwargs)
        generated = outputs[0]["generated_text"]

        if isinstance(generated, list):
            last_message = generated[-1]
            if isinstance(last_message, dict):
                return str(last_message.get("content", "")).strip()
            return str(last_message).strip()

        return str(generated).strip()


class StructuredLLMClient:
    """Common ``.generate(prompt, max_tokens=..., temperature=...)`` interface.

    Wraps ``LlamaClient`` lazily (the heavy pipeline only loads on first
    call). Swap the backing model by passing a different ``model_id`` to
    the constructor.

    Determinism note: Llama inference on GPU is only approximately
    deterministic even with a fixed seed due to non-associative floating
    point operations. We set the torch seed if provided, but the paper
    should still report means over multiple runs.
    """

    def __init__(
        self,
        model_id: str = LLAMA_33_70B_DEFAULT,
        *,
        default_max_tokens: int = 800,
        default_temperature: float = 0.3,
        top_p: float = 0.9,
        device_map: str = "auto",
        seed: Optional[int] = 2024,
    ) -> None:
        self.model_id = model_id
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self.top_p = top_p
        self.device_map = device_map
        self.seed = seed
        self._backend: Any = None
        self._current_max_tokens: int = default_max_tokens
        self._current_temperature: float = default_temperature

    def _load_backend(self) -> Any:
        if self._backend is not None:
            return self._backend
        if self.seed is not None:
            try:
                import torch

                torch.manual_seed(self.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.seed)
            except Exception:
                logger.warning("Could not set torch seed; continuing.", exc_info=True)

        self._backend = LlamaClient(
            model_id=self.model_id,
            max_new_tokens=self.default_max_tokens,
            temperature=self.default_temperature,
            top_p=self.top_p,
            device_map=self.device_map,
        )
        return self._backend

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        backend = self._load_backend()
        mt = int(max_tokens if max_tokens is not None else self.default_max_tokens)
        tp = float(temperature if temperature is not None else self.default_temperature)

        if mt != self._current_max_tokens:
            backend.max_new_tokens = mt
            self._current_max_tokens = mt
        if tp != self._current_temperature:
            backend.temperature = tp
            self._current_temperature = tp

        return backend.generate(prompt)


# ── Dummy LLM for CPU smoke tests ──────────────────────────────────────────


class DummyStructuredLLM:
    """Emits a well-formed five-block response every call (CPU-only).

    Useful to verify the parser / agent loop / logger before paying for
    GPU time. The decision alternates between a counter-offer and an
    accept after a few exchanges so the simulation terminates cleanly.
    """

    def __init__(self, accept_after_turns: int = 4) -> None:
        self.accept_after_turns = accept_after_turns
        self._call_count = 0

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        self._call_count += 1
        pending = "submit-deal" in (prompt or "").lower() or "proposes" in (prompt or "").lower()

        if self._call_count >= self.accept_after_turns and pending:
            decision = {"action": "accept", "counter_offer": None}
            utterance = "That works for me — let's lock it in."
            plan = (
                "The opponent's latest offer gives me enough of my top priority "
                "that accepting is better than continuing."
            )
        else:
            decision = {
                "action": "reject",
                "counter_offer": {"Food": 2, "Water": 1, "Firewood": 1},
            }
            utterance = (
                "I could use 2 food, 1 water, and 1 firewood — food is what "
                "I really need for the trip."
            )
            plan = (
                "Counter-propose a split weighted toward my High-priority item; "
                "this is consistent with my <opponent_inference> that the "
                "opponent does not have Food as their top item."
            )

        return (
            "<observation>\n"
            "The opponent spoke or made an offer. Details unknown to dummy model.\n"
            "</observation>\n\n"
            "<opponent_inference>\n"
            "Insufficient signal for a confident ranking; most likely the opponent "
            "values Water above Firewood above Food, but Firewood > Water is also "
            "plausible.\n"
            "</opponent_inference>\n\n"
            f"<plan>\n{plan}\n</plan>\n\n"
            f"<utterance>\n{utterance}\n</utterance>\n\n"
            f"<decision>\n{json.dumps(decision)}\n</decision>\n"
        )


__all__ = [
    "LLAMA_33_70B_DEFAULT",
    "LLAMA_31_8B_DEFAULT",
    "LlamaClient",
    "StructuredLLMClient",
    "DummyStructuredLLM",
]
