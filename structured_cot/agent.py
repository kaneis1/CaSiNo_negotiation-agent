"""StructuredCoTAgent — the prompt + LLM + parser loop.

One class; the blueprint from the planning phase dropped almost
unchanged. Retry policy: one retry with a reminder appended to the
prompt; if the retry also fails, emit ``safe_default()`` and log the
failure so later sweeps can compute a parse-error rate.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from structured_cot.parser import REMINDER, parse_response, safe_default
from structured_cot.prompts import build_prompt

logger = logging.getLogger("structured_cot.agent")


@dataclass
class ActResult:
    """One turn's output: parsed blocks, raw LLM text, retry metadata."""

    parsed: Dict[str, Any]
    raw_first: str
    raw_retry: Optional[str] = None
    retried: bool = False
    fell_back: bool = False
    elapsed_seconds: float = 0.0
    parse_errors: List[str] = field(default_factory=list)


class StructuredCoTAgent:
    """Blueprint-shaped agent: priorities + arguments + LLM -> action.

    Args:
        priorities: ``{"High": "Food", "Medium": "Water", "Low": "Firewood"}``.
        arguments:  ``{"High": "...", "Medium": "...", "Low": "..."}`` (free-text
            justifications from CaSiNo's ``value2reason``).
        llm_client: any object exposing
            ``generate(prompt: str, *, max_tokens: int, temperature: float) -> str``.
            See ``structured_cot.llm_client.StructuredLLMClient`` /
            ``DummyStructuredLLM``.
        max_tokens: forwarded to ``llm.generate`` (blueprint default 800).
        temperature: forwarded to ``llm.generate`` (blueprint default 0.3).
        parse_log_path: if set, every parse-failure event is appended to
            this file as a single JSON line. Leave ``None`` to disable.
    """

    def __init__(
        self,
        priorities: Mapping[str, str],
        arguments: Mapping[str, str],
        llm_client: Any,
        *,
        max_tokens: int = 800,
        temperature: float = 0.3,
        parse_log_path: Optional[Path] = None,
    ) -> None:
        self.priorities: Dict[str, str] = dict(priorities)
        self.arguments: Dict[str, str] = dict(arguments)
        self.llm = llm_client
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.parse_log_path: Optional[Path] = (
            Path(parse_log_path) if parse_log_path is not None else None
        )
        self.turn_count: int = 0
        self.parse_failures: int = 0

    # ── Convenience ────────────────────────────────────────────────────
    def state(self) -> Dict[str, Any]:
        return {
            "priorities": dict(self.priorities),
            "arguments": dict(self.arguments),
            "turn_index": self.turn_count,
        }

    def reset(self) -> None:
        self.turn_count = 0

    # ── Core loop ──────────────────────────────────────────────────────
    def act(
        self,
        dialogue_history: Sequence[Tuple[str, str]],
        *,
        pending_offer: Optional[Mapping[str, Any]] = None,
    ) -> ActResult:
        """Produce one turn's decision.

        ``pending_offer`` is accepted for future use (e.g. a prompt hint)
        but currently the agent re-derives offer state from
        ``dialogue_history``; the field is plumbed through so callers can
        record it in the eval logs without changing the interface later.
        """
        prompt = build_prompt(self.state(), dialogue_history)

        t0 = time.time()
        raw_first = self.llm.generate(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        parsed = parse_response(raw_first)

        raw_retry: Optional[str] = None
        retried = False
        if parsed.get("parse_error"):
            retried = True
            logger.warning(
                "Parse failure (turn=%d): %s", self.turn_count, parsed["parse_error"],
            )
            self._log_parse_failure(
                prompt=prompt, raw=raw_first, parse_error=parsed["parse_error"],
                phase="first",
            )
            raw_retry = self.llm.generate(
                prompt + REMINDER,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            parsed = parse_response(raw_retry)

        fell_back = False
        parse_errors: List[str] = []
        if parsed.get("parse_error"):
            self._log_parse_failure(
                prompt=prompt + REMINDER,
                raw=raw_retry or "",
                parse_error=parsed["parse_error"],
                phase="retry",
            )
            parse_errors.append(parsed["parse_error"])
            parsed = safe_default(
                pending_offer=bool(pending_offer),
                reason=f"parse failure after retry: {parsed['parse_error']}",
            )
            fell_back = True
            self.parse_failures += 1

        self.turn_count += 1
        return ActResult(
            parsed=parsed,
            raw_first=raw_first,
            raw_retry=raw_retry,
            retried=retried,
            fell_back=fell_back,
            elapsed_seconds=time.time() - t0,
            parse_errors=parse_errors,
        )

    # ── Failure logging ────────────────────────────────────────────────
    def _log_parse_failure(
        self,
        *,
        prompt: str,
        raw: str,
        parse_error: str,
        phase: str,
    ) -> None:
        if self.parse_log_path is None:
            return
        import json as _json

        self.parse_log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "time": time.time(),
            "turn": self.turn_count,
            "phase": phase,
            "parse_error": parse_error,
            "raw_output": raw,
            "prompt_tail": prompt[-600:],
        }
        try:
            with self.parse_log_path.open("a") as f:
                f.write(_json.dumps(record) + "\n")
        except Exception:
            logger.exception("Failed to append parse-failure log entry.")


__all__ = ["StructuredCoTAgent", "ActResult"]
