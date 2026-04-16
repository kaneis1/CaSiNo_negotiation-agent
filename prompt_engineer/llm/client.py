#!/usr/bin/env python3
"""Transformers pipeline wrapper for Llama Instruct models.

Usage:
    from prompt_engineer.llm.client import LlamaClient
    client = LlamaClient()
    response = client.generate("your prompt here")
"""

from __future__ import annotations

import torch
import transformers


class LlamaClient:
    """Thin wrapper around a chat-style Llama Instruct pipeline.

    Implements the .generate(prompt: str) -> str interface expected by
    all CaSiNo prompt_engineer modules (agent.py, classify_strategy.py,
    judge.py, etc.).

    Args:
        model_id: HuggingFace model ID or local path.
        system_prompt: Optional system message prepended to every call.
        max_new_tokens: Max tokens to generate per call.
        temperature: Sampling temperature. Set to 0.0 for greedy decoding.
        top_p: Nucleus sampling threshold (ignored when temperature=0).
        device_map: Passed through to transformers.pipeline().
    """

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

        print(f"Loading text-generation pipeline: {model_id}")
        self.pipeline = transformers.pipeline(
            "text-generation",
            model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map=device_map,
        )
        print("Pipeline ready.")

    def generate(self, prompt: str) -> str:
        """Run inference on a fully-formatted prompt string.

        The repo already builds a full prompt string, so we wrap it as a
        single user message and let the instruct chat template handle the
        formatting for the model.
        """
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

        # Chat pipelines return the full conversation as a list of messages.
        if isinstance(generated, list):
            last_message = generated[-1]
            if isinstance(last_message, dict):
                return str(last_message.get("content", "")).strip()
            return str(last_message).strip()

        return str(generated).strip()


# ── Quick smoke test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test the LlamaClient.")
    parser.add_argument("--model-id", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--prompt", default="Say hello in one sentence.")
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt to prepend to the chat.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    client = LlamaClient(
        model_id=args.model_id,
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
    )

    print("\nPrompt:", args.prompt)
    print("Response:", client.generate(args.prompt))
