"""Posterior providers and parsers for DND transfer runs."""

from __future__ import annotations

import json
import logging
import re
from itertools import permutations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from sft_8b.dnd_data import DNDRecord, DNDTurn, DND_ITEMS, canonical_item, item_label
from sft_8b.dnd_prompts import (
    build_direct_posterior_prompt,
    build_prefs_user_prompt,
    direct_posterior_system_prompt,
    prefs_system_prompt,
)
from sft_8b.dnd_rules import combine_evidence, posterior_from_evidence, score_utterance

logger = logging.getLogger("sft_8b.dnd_posterior")

ORDERINGS: List[Tuple[str, str, str]] = list(permutations(DND_ITEMS))
N_ORDERINGS = len(ORDERINGS)
ORDERING_INDEX = {tuple(o): i for i, o in enumerate(ORDERINGS)}

_JSON_RE = re.compile(r"\{.*?\}", re.S)
_POST_LINE_RE = re.compile(
    r"p\((?P<ordering>[^)]+)\)\s*=\s*(?P<prob>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)",
    re.I,
)


def normalize_posterior(raw: Sequence[float]) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64).flatten()
    if arr.shape != (N_ORDERINGS,):
        raise ValueError(f"posterior must have length {N_ORDERINGS}, got {arr.shape}")
    arr = np.maximum(np.where(np.isfinite(arr), arr, 0.0), 0.0)
    total = float(arr.sum())
    if total <= 0:
        return np.full(N_ORDERINGS, 1.0 / N_ORDERINGS, dtype=np.float64)
    return arr / total


def ordering_label(ordering: Sequence[str], *, name_mode: str) -> str:
    return " > ".join(item_label(item, name_mode=name_mode) for item in ordering)


def _extract_json(text: str) -> Optional[Mapping[str, Any]]:
    raw = (text or "").strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, Mapping):
            return obj
    except json.JSONDecodeError:
        pass
    m = _JSON_RE.search(raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, Mapping) else None


def coerce_ordering(raw: Any) -> Optional[Tuple[str, str, str]]:
    if isinstance(raw, str):
        parts = re.split(r"\s*>\s*|,\s*|\s*/\s*", raw.strip())
    elif isinstance(raw, Sequence):
        parts = [str(x) for x in raw]
    else:
        return None
    items: List[str] = []
    for part in parts:
        if not str(part).strip():
            continue
        item = canonical_item(part)
        if item is None:
            return None
        items.append(item)
    if len(items) != 3 or set(items) != set(DND_ITEMS):
        return None
    return tuple(items)  # type: ignore[return-value]


def parse_prefs_response(text: str) -> Tuple[Optional[Tuple[str, str, str]], List[str]]:
    errors: List[str] = []
    obj = _extract_json(text)
    if obj is None:
        errors.append("no JSON object")
        return None, errors
    raw = obj.get("prefs") or obj.get("preference") or obj.get("ordering")
    ordering = coerce_ordering(raw)
    if ordering is None:
        errors.append(f"prefs not a valid DND ordering: {raw!r}")
    return ordering, errors


def _one_hot(ordering: Tuple[str, str, str]) -> np.ndarray:
    out = np.zeros(N_ORDERINGS, dtype=np.float64)
    out[ORDERING_INDEX[tuple(ordering)]] = 1.0
    return out


def parse_posterior_response(text: str) -> Tuple[np.ndarray, List[str]]:
    errors: List[str] = []
    raw = text or ""
    block = re.search(r"<posterior\s*>(.*?)</posterior\s*>", raw, re.I | re.S)
    if block:
        out = np.zeros(N_ORDERINGS, dtype=np.float64)
        found = 0
        for m in _POST_LINE_RE.finditer(block.group(1)):
            ordering = coerce_ordering(m.group("ordering"))
            if ordering is None:
                errors.append(f"unknown ordering {m.group('ordering')!r}")
                continue
            out[ORDERING_INDEX[ordering]] = float(m.group("prob"))
            found += 1
        if found:
            return normalize_posterior(out), errors
        errors.append("posterior block contained no probabilities")

    obj = _extract_json(raw)
    if obj is not None:
        vals = obj.get("posterior")
        if isinstance(vals, list):
            try:
                return normalize_posterior(vals), errors
            except Exception as exc:
                errors.append(f"posterior list invalid: {exc}")
        probs = obj.get("probabilities") or obj.get("probs")
        if isinstance(probs, Mapping):
            out = np.zeros(N_ORDERINGS, dtype=np.float64)
            found = 0
            for key, val in probs.items():
                ordering = coerce_ordering(key)
                if ordering is None:
                    errors.append(f"unknown ordering key {key!r}")
                    continue
                out[ORDERING_INDEX[ordering]] = float(val)
                found += 1
            if found:
                return normalize_posterior(out), errors
        ordering, pref_errors = parse_prefs_response(raw)
        if ordering is not None:
            errors.append("parsed prefs JSON as one-hot posterior")
            return _one_hot(ordering), errors
        errors.extend(pref_errors)

    errors.append("no parseable posterior")
    return np.full(N_ORDERINGS, 1.0 / N_ORDERINGS, dtype=np.float64), errors


class DNDChatModel:
    def __init__(
        self,
        *,
        base_model: str,
        adapter_path: Optional[str],
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        device_map: str = "auto",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from sft_8b.model_utils import choose_inference_dtype

        self.base_model = base_model
        self.adapter_path = adapter_path
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        tok_src = adapter_path if adapter_path else base_model
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=choose_inference_dtype(),
            device_map=device_map,
            attn_implementation="sdpa",
        )
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path, autocast_adapter_dtype=False)
        model = model.eval()
        model.config.use_cache = True
        self.model = model

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        K: int = 1,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        attention_mask = torch.ones_like(input_ids)
        temp = self.temperature if temperature is None else float(temperature)
        do_sample = temp > 0.0 or int(K) > 1
        kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens or self.max_new_tokens),
            "do_sample": do_sample,
            "num_return_sequences": int(K),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs["temperature"] = max(temp, 1e-6)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
        new_tokens = output_ids[:, input_ids.shape[1] :]
        return [
            self.tokenizer.decode(row, skip_special_tokens=True).strip()
            for row in new_tokens
        ]


class DNDPosteriorProvider:
    parse_errors = 0

    def posterior(self, *, record: DNDRecord, history: Sequence[DNDTurn], name_mode: str) -> np.ndarray:
        raise NotImplementedError

    def summary(self) -> Dict[str, Any]:
        return {"parse_errors": int(getattr(self, "parse_errors", 0))}


class UniformDNDProvider(DNDPosteriorProvider):
    def posterior(self, *, record: DNDRecord, history: Sequence[DNDTurn], name_mode: str) -> np.ndarray:
        return np.full(N_ORDERINGS, 1.0 / N_ORDERINGS, dtype=np.float64)


class RuleDNDProvider(DNDPosteriorProvider):
    def posterior(self, *, record: DNDRecord, history: Sequence[DNDTurn], name_mode: str) -> np.ndarray:
        counts = {item: int(record.counts[i]) for i, item in enumerate(DND_ITEMS)}
        rows = [
            score_utterance(turn.text, counts=counts)
            for turn in history
            if turn.speaker == "THEM" and not turn.is_selection
        ]
        evidence = combine_evidence(rows)
        return posterior_from_evidence(evidence, ORDERINGS)


class MCPrefsDNDProvider(DNDPosteriorProvider):
    def __init__(self, model: DNDChatModel, *, K: int = 16, temperature: float = 0.7) -> None:
        self.model = model
        self.K = int(K)
        self.temperature = float(temperature)
        self.parse_errors = 0

    def posterior(self, *, record: DNDRecord, history: Sequence[DNDTurn], name_mode: str) -> np.ndarray:
        system = prefs_system_prompt(name_mode=name_mode)
        user = build_prefs_user_prompt(record=record, history=history, name_mode=name_mode)
        samples = self.model.generate(
            system_prompt=system,
            user_prompt=user,
            K=self.K,
            temperature=self.temperature,
            max_new_tokens=self.model.max_new_tokens,
        )
        counts = np.zeros(N_ORDERINGS, dtype=np.float64)
        parsed = 0
        for raw in samples:
            ordering, errors = parse_prefs_response(raw)
            if ordering is None:
                self.parse_errors += 1
                continue
            counts[ORDERING_INDEX[ordering]] += 1.0
            parsed += 1
        if parsed == 0:
            return np.full(N_ORDERINGS, 1.0 / N_ORDERINGS, dtype=np.float64)
        return counts / float(parsed)


class DirectPosteriorDNDProvider(DNDPosteriorProvider):
    def __init__(self, model: DNDChatModel) -> None:
        self.model = model
        self.parse_errors = 0

    def posterior(self, *, record: DNDRecord, history: Sequence[DNDTurn], name_mode: str) -> np.ndarray:
        system = direct_posterior_system_prompt(name_mode=name_mode)
        user = build_direct_posterior_prompt(
            record=record,
            history=history,
            name_mode=name_mode,
            orderings=ORDERINGS,
        )
        raw = self.model.generate(
            system_prompt=system,
            user_prompt=user,
            K=1,
            temperature=0.0,
            max_new_tokens=256,
        )[0]
        posterior, errors = parse_posterior_response(raw)
        if errors:
            self.parse_errors += 1
        return posterior
