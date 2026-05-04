"""Ablation harness for the current ``sft_8b`` CaSiNo paper stack.

This module keeps the ablation variants out of the main Protocol-3 runner.
It exposes a generic turn-level teacher whose posterior can come from MC
sampling, a direct posterior prompt/model, incremental Bayes, rule likelihoods,
or a uniform prior.  The output layout mirrors ``casino_belief.evaluation.turn_eval_run``
so downstream result scripts can read both families of runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
import time
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from casino_belief.evaluation.hypotheses import HYPOTHESES, ITEMS
from casino_belief.evaluation.turn_agents import KeywordStrategyClassifier
from casino_belief.evaluation.turn_level_metrics import (
    DEAL_ACTIONS,
    TurnRecord,
    build_annotation_lookup,
    format_turn_level_summary,
    normalized_brier,
    turn_level_eval,
)
from casino_belief.belief.bayesian_agent import (
    DEFAULT_ACCEPT_FLOOR,
    DEFAULT_ACCEPT_MARGIN,
    DEFAULT_LAMBDA,
    pending_self_points,
    select_action,
    template_utterance,
)
from casino_belief.policy.menu import PRIORITY_POINTS, build_menu, points
from casino_belief.belief.posterior import N_ORDERINGS, ORDERINGS, get_posterior
from casino_belief.belief.prompts import build_user_prompt
from casino_belief.student.student_parser import parse_posterior_block
from casino_belief.student.student_prompts import format_posterior

logger = logging.getLogger("casino_belief.diagnostics.ablation.ablation")

DEFAULT_BASE_MODEL = (
    "/sc/arion/scratch/cuiz02/hf_cache/transformers/Meta-Llama-3.1-8B-Instruct"
)
DEFAULT_TEACHER_ADAPTER = "artifacts/training_metadata/teacher_lora_run/lora_best"
DEFAULT_OUTPUT_ROOT = Path("artifacts/results/ablation/main")

EVIDENCE_MODES = {
    "utterance_only",
    "utterance_plus_offers",
    "offers_only",
    "preference_utterances_only",
    "nonpreference_utterances_only",
}
PREFERENCE_LABELS = {"self-need", "other-need", "elicit-pref"}
NONPREFERENCE_LABELS = {
    "small-talk",
    "promote-coordination",
    "vouch-fair",
    "showing-empathy",
    "uv-part",
    "non-strategic",
    "no-need",
}

PROVIDER_CHOICES = {
    "mc_k16_full_context",
    "mc_k1_full_context",
    "direct_zero_shot",
    "direct_sft_groundtruth",
    "direct_sft_teacher",
    "incremental_bayes_utterance",
    "rule_likelihood",
    "uniform",
}


# ── Formatting + normalization ────────────────────────────────────────────


def normalize_posterior(raw: Sequence[float]) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64).flatten()
    if arr.shape != (N_ORDERINGS,):
        raise ValueError(f"posterior must have shape ({N_ORDERINGS},), got {arr.shape}")
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if total <= 0:
        return np.full(N_ORDERINGS, 1.0 / N_ORDERINGS, dtype=np.float64)
    return arr / total


def hypothesis_name(ordering: Sequence[str]) -> str:
    return " > ".join(ordering)


def parse_probability_response(text: str) -> Tuple[np.ndarray, List[str]]:
    """Parse direct posterior generations into a normalized six-vector.

    Accepted outputs:
      * a ``<posterior>`` block in the Day-8 format,
      * JSON ``{"posterior": [..six floats..]}``,
      * JSON ``{"probabilities": {"Food > Water > Firewood": 0.1, ...}}``.
    """
    errors: List[str] = []
    block_match = re.search(r"<posterior\s*>(.*?)</posterior\s*>", text or "", re.I | re.S)
    if block_match:
        posterior, errs = parse_posterior_block(block_match.group(1))
        if posterior is not None:
            return normalize_posterior(posterior), errs
        errors.extend(errs)

    raw = (text or "").strip()
    obj: Optional[Mapping[str, Any]] = None
    try:
        maybe = json.loads(raw)
        if isinstance(maybe, Mapping):
            obj = maybe
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                maybe = json.loads(m.group(0))
                if isinstance(maybe, Mapping):
                    obj = maybe
            except json.JSONDecodeError as exc:
                errors.append(f"json parse failed: {exc}")
    if obj is None:
        errors.append("no parseable posterior object")
        return np.full(N_ORDERINGS, 1.0 / N_ORDERINGS), errors

    if isinstance(obj.get("posterior"), list):
        try:
            return normalize_posterior(obj["posterior"]), errors
        except Exception as exc:
            errors.append(f"posterior list invalid: {exc}")

    probs = obj.get("probabilities") or obj.get("probs")
    if isinstance(probs, Mapping):
        out = np.zeros(N_ORDERINGS, dtype=np.float64)
        name_to_idx = {hypothesis_name(o).lower(): i for i, o in enumerate(ORDERINGS)}
        for key, value in probs.items():
            idx = name_to_idx.get(str(key).strip().lower())
            if idx is None:
                errors.append(f"unknown ordering key {key!r}")
                continue
            try:
                out[idx] = float(value)
            except (TypeError, ValueError):
                errors.append(f"bad probability for {key!r}: {value!r}")
        return normalize_posterior(out), errors

    errors.append("no supported posterior shape")
    return np.full(N_ORDERINGS, 1.0 / N_ORDERINGS), errors


def parse_score_response(text: str) -> Tuple[np.ndarray, List[str]]:
    """Parse incremental likelihood scores on a 0-100 neutral-at-50 scale."""
    errors: List[str] = []
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text or "", re.S)
        if not m:
            errors.append("no JSON score object")
            return np.full(N_ORDERINGS, 50.0), errors
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError as exc:
            errors.append(f"score JSON parse failed: {exc}")
            return np.full(N_ORDERINGS, 50.0), errors
    if not isinstance(obj, Mapping):
        errors.append("score JSON root is not an object")
        return np.full(N_ORDERINGS, 50.0), errors
    scores = obj.get("scores") or obj.get("evidence_scores") or obj.get("likelihood")
    if isinstance(scores, list):
        if len(scores) != N_ORDERINGS:
            errors.append(f"scores list has length {len(scores)}, expected {N_ORDERINGS}")
            return np.full(N_ORDERINGS, 50.0), errors
        try:
            return np.asarray(scores, dtype=np.float64), errors
        except (TypeError, ValueError) as exc:
            errors.append(f"scores list invalid: {exc}")
            return np.full(N_ORDERINGS, 50.0), errors
    if isinstance(scores, Mapping):
        out = np.full(N_ORDERINGS, 50.0, dtype=np.float64)
        name_to_idx = {hypothesis_name(o).lower(): i for i, o in enumerate(ORDERINGS)}
        for key, value in scores.items():
            idx = name_to_idx.get(str(key).strip().lower())
            if idx is None:
                continue
            try:
                out[idx] = float(value)
            except (TypeError, ValueError):
                errors.append(f"bad score for {key!r}: {value!r}")
        return out, errors
    errors.append("no supported score shape")
    return np.full(N_ORDERINGS, 50.0), errors


def render_submit_deal(turn: Mapping[str, Any], *, my_role: str) -> Optional[str]:
    if turn.get("text") != "Submit-Deal":
        return None
    td = turn.get("task_data") or {}
    if "issue2youget" not in td or "issue2theyget" not in td:
        return None
    proposer = turn.get("id")
    if proposer == my_role:
        mine = td["issue2youget"]
        theirs = td["issue2theyget"]
        who = "I proposed"
    else:
        mine = td["issue2theyget"]
        theirs = td["issue2youget"]
        who = "Opponent proposed"
    try:
        mine_txt = ", ".join(f"{it}={int(mine.get(it, 0))}" for it in ITEMS)
        theirs_txt = ", ".join(f"{it}={int(theirs.get(it, 0))}" for it in ITEMS)
    except (TypeError, ValueError):
        return None
    return f"{who}: my_share({mine_txt}); opponent_share({theirs_txt})."


def format_evidence_turns(turns: Sequence[Mapping[str, Any]], *, my_role: str) -> str:
    lines: List[str] = []
    for turn in turns:
        text = str(turn.get("text", "")).strip()
        if not text:
            continue
        speaker = "Me" if turn.get("id") == my_role else "Opponent"
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines) if lines else "(no usable evidence)"


class EvidenceRenderer:
    """Build provider-specific histories while keeping eval history intact."""

    def __init__(
        self,
        *,
        mode: str = "utterance_only",
        annotations_by_dialogue: Optional[Mapping[Any, Sequence[Any]]] = None,
    ) -> None:
        if mode not in EVIDENCE_MODES:
            raise ValueError(f"unknown evidence mode {mode!r}")
        self.mode = mode
        self.annotations_by_dialogue = annotations_by_dialogue or {}

    def _label_set_for_turn(
        self,
        turn: Mapping[str, Any],
        *,
        dialogue_id: Any,
    ) -> Optional[set[str]]:
        idx = turn.get("ablation_turn_index")
        if idx is None:
            return None
        # If turn carries labels from a pre-stamping pass, prefer them.
        raw_labels = turn.get("ablation_strategy_labels")
        if raw_labels is not None:
            return {str(x).strip() for x in raw_labels if str(x).strip()}
        anns = (
            self.annotations_by_dialogue.get(dialogue_id)
            or self.annotations_by_dialogue.get(str(dialogue_id))
            or []
        )
        if not anns:
            return None
        # Build a tiny lookup for this dialogue on demand. Histories are short,
        # and avoiding a global chat-log dependency keeps the renderer reusable.
        natural_i = 0
        # Fallback: direct text match against annotation rows.
        text = turn.get("text")
        for row in anns:
            if isinstance(row, (list, tuple)) and len(row) >= 2 and row[0] == text:
                return {x.strip() for x in str(row[1]).split(",") if x.strip()}
            natural_i += 1
        return None

    def transform_history(
        self,
        history: Sequence[Mapping[str, Any]],
        *,
        my_role: str,
        opp_role: str,
        dialogue_id: Any = None,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for turn in history:
            text = str(turn.get("text", "")).strip()
            if not text:
                continue
            is_action = text in DEAL_ACTIONS
            if is_action:
                rendered = render_submit_deal(turn, my_role=my_role)
                if rendered and self.mode in {"utterance_plus_offers", "offers_only"}:
                    d = dict(turn)
                    d["text"] = rendered
                    out.append(d)
                continue
            if self.mode == "offers_only":
                continue
            if self.mode in {"preference_utterances_only", "nonpreference_utterances_only"}:
                labels = self._label_set_for_turn(turn, dialogue_id=dialogue_id)
                if labels is None:
                    continue
                is_pref = bool(labels & PREFERENCE_LABELS)
                if self.mode == "preference_utterances_only" and not is_pref:
                    continue
                if self.mode == "nonpreference_utterances_only" and is_pref:
                    continue
            out.append(dict(turn))
        return out

    def newest_opponent_utterances(
        self,
        history: Sequence[Mapping[str, Any]],
        *,
        opp_role: str,
    ) -> List[str]:
        return [
            str(t.get("text", "")).strip()
            for t in history
            if t.get("id") == opp_role
            and str(t.get("text", "")).strip()
            and str(t.get("text", "")).strip() not in DEAL_ACTIONS
        ]


# ── Model wrappers ────────────────────────────────────────────────────────


class ChatModel:
    """Small HF+LoRA chat wrapper used by direct posterior/likelihood prompts."""

    def __init__(
        self,
        *,
        base_model: str,
        adapter_path: Optional[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        device_map: str = "auto",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from casino_belief.student.model_utils import choose_inference_dtype

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

            model = PeftModel.from_pretrained(
                model,
                adapter_path,
                autocast_adapter_dtype=False,
            )
        model = model.eval()
        model.config.use_cache = True
        self.model = model

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        assistant_prefix: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        add_generation_prompt = True
        if assistant_prefix is not None:
            messages.append({"role": "assistant", "content": assistant_prefix})
            add_generation_prompt = False
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        ).to(self.model.device)
        attention_mask = torch.ones_like(input_ids)
        temp = self.temperature if temperature is None else float(temperature)
        do_sample = temp > 0.0
        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens or self.max_new_tokens),
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temp
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
        new_tokens = output_ids[0, input_ids.shape[1] :]
        continuation = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return f"{assistant_prefix or ''}{continuation}".strip()


DIRECT_POSTERIOR_SYSTEM = """\
You infer an opponent's hidden priority ordering in the CaSiNo camping
negotiation. Return a calibrated probability distribution over exactly the six
orderings. Reply with one <posterior> block and no extra prose.
"""


LIKELIHOOD_SYSTEM = """\
You score how compatible one opponent utterance is with each hidden priority
ordering in the CaSiNo camping negotiation. Return JSON only:
{"scores":[s1,s2,s3,s4,s5,s6]}
Scores are on a 0-100 scale, where 50 means no evidence, higher means more
compatible, and lower means less compatible.
"""


def direct_posterior_prompt(
    *,
    evidence: str,
    my_priorities: Mapping[str, str],
    my_reasons: Mapping[str, str],
) -> str:
    orderings = "\n".join(
        f"{i + 1}. {hypothesis_name(o)}" for i, o in enumerate(ORDERINGS)
    )
    own = "\n".join(f"{k}: {my_priorities.get(k, '?')}" for k in ("High", "Medium", "Low"))
    reasons = "\n".join(
        f"{k}: {str(my_reasons.get(k, '')).strip() or '(none)'}"
        for k in ("High", "Medium", "Low")
    )
    return (
        "Your own priorities:\n"
        f"{own}\n\n"
        "Your own reasons:\n"
        f"{reasons}\n\n"
        "Dialogue evidence so far:\n"
        f"{evidence}\n\n"
        "Candidate opponent orderings:\n"
        f"{orderings}\n\n"
        "Output exactly:\n"
        "<posterior>\n"
        "p(Food > Water > Firewood)=...\n"
        "...\n"
        "</posterior>"
    )


def incremental_likelihood_prompt(utterance: str) -> str:
    orderings = "\n".join(
        f"{i + 1}. {hypothesis_name(o)}" for i, o in enumerate(ORDERINGS)
    )
    return (
        "Opponent utterance only, with no dialogue history:\n"
        f"\"{utterance}\"\n\n"
        "Candidate orderings in score-list order:\n"
        f"{orderings}\n\n"
        "Return JSON only with six scores in this exact order."
    )


# ── Posterior providers ───────────────────────────────────────────────────


class PosteriorProvider:
    def posterior(
        self,
        *,
        history: List[Mapping[str, Any]],
        my_role: str,
        opp_role: str,
        my_priorities: Mapping[str, str],
        my_reasons: Mapping[str, str],
        dialogue_id: Any = None,
        turn_index: Optional[int] = None,
    ) -> np.ndarray:
        raise NotImplementedError


class UniformProvider(PosteriorProvider):
    def posterior(self, **kwargs: Any) -> np.ndarray:
        return np.full(N_ORDERINGS, 1.0 / N_ORDERINGS, dtype=np.float64)


class MCPProvider(PosteriorProvider):
    def __init__(self, model_fn: Any, *, K: int, temperature: float, renderer: EvidenceRenderer) -> None:
        self.model_fn = model_fn
        self.K = int(K)
        self.temperature = float(temperature)
        self.renderer = renderer

    def posterior(self, **kwargs: Any) -> np.ndarray:
        hist = self.renderer.transform_history(
            kwargs["history"],
            my_role=kwargs["my_role"],
            opp_role=kwargs["opp_role"],
            dialogue_id=kwargs.get("dialogue_id"),
        )
        return normalize_posterior(get_posterior(
            dialogue_prefix=hist,
            speaker_priorities=kwargs["my_priorities"],
            model_fn=self.model_fn,
            speaker_reasons=kwargs["my_reasons"],
            me_role=kwargs["my_role"],
            K=self.K,
            temperature=self.temperature,
        ))


class DirectPosteriorProvider(PosteriorProvider):
    def __init__(self, model: ChatModel, *, renderer: EvidenceRenderer) -> None:
        self.model = model
        self.renderer = renderer
        self.parse_errors = 0

    def posterior(self, **kwargs: Any) -> np.ndarray:
        hist = self.renderer.transform_history(
            kwargs["history"],
            my_role=kwargs["my_role"],
            opp_role=kwargs["opp_role"],
            dialogue_id=kwargs.get("dialogue_id"),
        )
        evidence = format_evidence_turns(hist, my_role=kwargs["my_role"])
        prompt = direct_posterior_prompt(
            evidence=evidence,
            my_priorities=kwargs["my_priorities"],
            my_reasons=kwargs["my_reasons"],
        )
        raw = self.model.generate(
            system_prompt=DIRECT_POSTERIOR_SYSTEM,
            user_prompt=prompt,
        )
        posterior, errors = parse_probability_response(raw)
        if errors:
            self.parse_errors += 1
        return posterior


class IncrementalBayesProvider(PosteriorProvider):
    """Sequential update over prior opponent natural utterances only."""

    def __init__(
        self,
        model: ChatModel,
        *,
        likelihood_temperature: float = 25.0,
        likelihood_clip: Tuple[Optional[float], Optional[float]] = (-3.0, 3.0),
    ) -> None:
        self.model = model
        self.likelihood_temperature = float(likelihood_temperature)
        self.likelihood_clip = likelihood_clip
        self.parse_errors = 0

    def _log_likelihood(self, utterance: str) -> np.ndarray:
        raw = self.model.generate(
            system_prompt=LIKELIHOOD_SYSTEM,
            user_prompt=incremental_likelihood_prompt(utterance),
            max_new_tokens=96,
        )
        scores, errors = parse_score_response(raw)
        if errors:
            self.parse_errors += 1
        ll = (scores - 50.0) / self.likelihood_temperature
        lo, hi = self.likelihood_clip
        if lo is not None or hi is not None:
            ll = np.clip(
                ll,
                -np.inf if lo is None else float(lo),
                np.inf if hi is None else float(hi),
            )
        return ll

    def posterior(self, **kwargs: Any) -> np.ndarray:
        utterances = EvidenceRenderer().newest_opponent_utterances(
            kwargs["history"],
            opp_role=kwargs["opp_role"],
        )
        if not utterances:
            return np.full(N_ORDERINGS, 1.0 / N_ORDERINGS, dtype=np.float64)
        logp = np.full(N_ORDERINGS, -math.log(N_ORDERINGS), dtype=np.float64)
        for utt in utterances:
            logp += self._log_likelihood(utt)
            logp -= float(np.max(logp))
            logp -= math.log(float(np.exp(logp).sum()))
        return normalize_posterior(np.exp(logp))


class RuleLikelihoodProvider(PosteriorProvider):
    """Deterministic keyword/offer likelihood with the same sequential update."""

    NEED_PATTERNS = (
        r"\bneed(?:ed)?\s+(?P<item>food|water|firewood|wood)\b",
        r"\b(?P<item>food|water|firewood|wood)\s+(?:is|are)\s+(?:important|priority|essential)\b",
        r"\bprefer\s+(?P<item>food|water|firewood|wood)\b",
        r"\bwant\s+(?P<item>food|water|firewood|wood)\b",
    )
    GIVE_PATTERNS = (
        r"\byou\s+(?:can|may)\s+(?:have|take)\s+(?:all\s+)?(?:the\s+)?(?P<item>food|water|firewood|wood)\b",
        r"\bi\s+(?:do not|don't)\s+need\s+(?P<item>food|water|firewood|wood)\b",
    )

    def _canon(self, item: str) -> str:
        item = item.lower()
        return "Firewood" if item == "wood" else item.capitalize()

    def _scores_for_text(self, text: str) -> np.ndarray:
        t = text.lower()
        scores = np.zeros(N_ORDERINGS, dtype=np.float64)
        for pat in self.NEED_PATTERNS:
            for m in re.finditer(pat, t):
                item = self._canon(m.group("item"))
                for i, ordering in enumerate(ORDERINGS):
                    if ordering[0] == item:
                        scores[i] += 2.0
                    elif ordering[2] == item:
                        scores[i] -= 1.0
        for pat in self.GIVE_PATTERNS:
            for m in re.finditer(pat, t):
                item = self._canon(m.group("item"))
                for i, ordering in enumerate(ORDERINGS):
                    if ordering[2] == item:
                        scores[i] += 1.5
                    elif ordering[0] == item:
                        scores[i] -= 0.75
        return scores

    def _scores_for_offer(self, turn: Mapping[str, Any], *, opp_role: str) -> np.ndarray:
        if turn.get("text") != "Submit-Deal" or turn.get("id") != opp_role:
            return np.zeros(N_ORDERINGS, dtype=np.float64)
        td = turn.get("task_data") or {}
        share = td.get("issue2youget")
        if not isinstance(share, Mapping):
            return np.zeros(N_ORDERINGS, dtype=np.float64)
        try:
            kept = {item: int(share.get(item, 0)) for item in ITEMS}
        except (TypeError, ValueError):
            return np.zeros(N_ORDERINGS, dtype=np.float64)
        scores = np.zeros(N_ORDERINGS, dtype=np.float64)
        for i, ordering in enumerate(ORDERINGS):
            for item in ITEMS:
                centered = kept[item] - 1.5
                if ordering[0] == item:
                    scores[i] += 0.75 * centered
                elif ordering[2] == item:
                    scores[i] -= 0.50 * centered
        return scores

    def posterior(self, **kwargs: Any) -> np.ndarray:
        utterances = EvidenceRenderer().newest_opponent_utterances(
            kwargs["history"],
            opp_role=kwargs["opp_role"],
        )
        offer_turns = [
            t for t in kwargs["history"]
            if t.get("text") == "Submit-Deal" and t.get("id") == kwargs["opp_role"]
        ]
        if not utterances and not offer_turns:
            return np.full(N_ORDERINGS, 1.0 / N_ORDERINGS, dtype=np.float64)
        logp = np.full(N_ORDERINGS, -math.log(N_ORDERINGS), dtype=np.float64)
        for utt in utterances:
            logp += self._scores_for_text(utt)
            logp -= float(np.max(logp))
            logp -= math.log(float(np.exp(logp).sum()))
        for offer in offer_turns:
            logp += self._scores_for_offer(offer, opp_role=kwargs["opp_role"])
            logp -= float(np.max(logp))
            logp -= math.log(float(np.exp(logp).sum()))
        return normalize_posterior(np.exp(logp))


# ── Generic turn-level teacher ─────────────────────────────────────────────


class AblationTeacherTurnAgent:
    def __init__(
        self,
        posterior_provider: PosteriorProvider,
        *,
        lambda_: float = DEFAULT_LAMBDA,
        accept_margin: int = DEFAULT_ACCEPT_MARGIN,
        accept_floor: float = DEFAULT_ACCEPT_FLOOR,
        strategy_classifier: Optional[Any] = None,
    ) -> None:
        self.posterior_provider = posterior_provider
        self.lambda_ = float(lambda_)
        self.accept_margin = int(accept_margin)
        self.accept_floor = float(accept_floor)
        self.strategy_classifier = strategy_classifier or KeywordStrategyClassifier()
        self._summary = {"calls": 0}

    @property
    def summary(self) -> Dict[str, Any]:
        out = dict(self._summary)
        if hasattr(self.posterior_provider, "parse_errors"):
            out["provider_parse_errors"] = getattr(self.posterior_provider, "parse_errors")
        return out

    def predict_turn(
        self,
        *,
        history: List[Mapping[str, Any]],
        my_role: str,
        opp_role: str,
        my_priorities: Mapping[str, str],
        my_reasons: Mapping[str, str],
        pending_offer: Optional[Mapping[str, Any]],
        my_personality: Optional[Mapping[str, Any]] = None,
        dialogue_id: Any = None,
        turn_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        self._summary["calls"] += 1
        posterior = self.posterior_provider.posterior(
            history=history,
            my_role=my_role,
            opp_role=opp_role,
            my_priorities=my_priorities,
            my_reasons=my_reasons,
            dialogue_id=dialogue_id,
            turn_index=turn_index,
        )
        menu = build_menu(posterior, my_priorities, lambda_=self.lambda_, top_k=5)
        pending_pts: Optional[int] = None
        if pending_offer is not None and pending_offer.get("to_perspective"):
            pending_pts = pending_self_points(
                {**dict(pending_offer), "perspective": my_role},
                my_priorities,
            )
        decision = select_action(
            menu,
            pending_self_points=pending_pts,
            accept_margin=self.accept_margin,
            accept_floor=self.accept_floor,
        )
        utterance = ""
        if decision["counter_split"] is not None:
            utterance = template_utterance(decision["counter_split"].self_counts)
        strategy = None
        if utterance:
            strategy = list(self.strategy_classifier(utterance, list(history))) or None
        return {
            "accept": decision["accept"],
            "bid": decision["bid"],
            "utterance": utterance or None,
            "action": (
                "accept" if decision["action"] == "accept"
                else "reject" if decision["action"] == "reject"
                else "submit" if decision["action"] == "propose"
                else decision["action"]
            ),
            "lambda": self.lambda_,
            "strategy": strategy,
            "posterior": posterior.tolist(),
        }


# ── Diagnostics ───────────────────────────────────────────────────────────


def posterior_diagnostics(records: Sequence[TurnRecord]) -> Dict[str, Any]:
    rows = []
    by_turn: Dict[int, List[Dict[str, float]]] = {}
    for r in records:
        p = (r.pred or {}).get("posterior")
        true_idx = (r.true or {}).get("true_hypothesis_index")
        if p is None or true_idx is None:
            continue
        arr = normalize_posterior(p)
        pred_idx = int(np.argmax(arr))
        entropy = float(-(arr[arr > 0] * np.log2(arr[arr > 0])).sum())
        item = {
            "map_correct": float(pred_idx == int(true_idx)),
            "entropy_bits": entropy,
        }
        rows.append(item)
        by_turn.setdefault(int(r.turn_index), []).append(item)
    if not rows:
        return {"support": 0}
    return {
        "support": len(rows),
        "map_accuracy": float(np.mean([x["map_correct"] for x in rows])),
        "entropy_bits_mean": float(np.mean([x["entropy_bits"] for x in rows])),
        "by_turn_index": [
            {
                "turn_index": t,
                "support": len(vals),
                "map_accuracy": float(np.mean([x["map_correct"] for x in vals])),
                "entropy_bits_mean": float(np.mean([x["entropy_bits"] for x in vals])),
            }
            for t, vals in sorted(by_turn.items())
        ],
    }


def brier_by_strategy_diagnostics(records: Sequence[TurnRecord]) -> Dict[str, Any]:
    by_label: Dict[str, List[float]] = {}
    for r in records:
        p = (r.pred or {}).get("posterior")
        true_idx = (r.true or {}).get("true_hypothesis_index")
        labels = (r.true or {}).get("strategy")
        if p is None or true_idx is None or not labels:
            continue
        arr = normalize_posterior(p)
        val = normalized_brier(arr, int(true_idx))
        for label in labels:
            by_label.setdefault(str(label), []).append(val)
    return {
        label: {"support": len(vals), "brier_mean": float(np.mean(vals))}
        for label, vals in sorted(by_label.items())
    }


def _priorities_to_points(value2issue: Mapping[str, str]) -> Dict[str, int]:
    return {
        item: PRIORITY_POINTS[level]
        for level, item in value2issue.items()
        if level in PRIORITY_POINTS
    }


def pareto_efficiency_diagnostics(
    records: Sequence[TurnRecord],
    dialogues: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    d_by_id = {str(d.get("dialogue_id")): d for d in dialogues}
    vals: List[float] = []
    for r in records:
        bid = (r.pred or {}).get("bid")
        if bid is None:
            continue
        d = d_by_id.get(str(r.dialogue_id))
        if not d:
            continue
        try:
            pinfo = d["participant_info"]
            my_pr = pinfo[r.perspective]["value2issue"]
            opp_pr = pinfo[r.opp_role]["value2issue"]
            self_counts = {it: int(bid[i]) for i, it in enumerate(ITEMS)}
            opp_counts = {it: int(bid[i + 3]) for i, it in enumerate(ITEMS)}
        except Exception:
            continue
        my_pts = _priorities_to_points(my_pr)
        opp_pts = _priorities_to_points(opp_pr)
        u_self = sum(self_counts[it] * my_pts[it] for it in ITEMS)
        u_opp = sum(opp_counts[it] * opp_pts[it] for it in ITEMS)
        dominated = False
        for f, w, fw in product(range(4), repeat=3):
            cand_self = {"Food": f, "Water": w, "Firewood": fw}
            cand_opp = {it: 3 - cand_self[it] for it in ITEMS}
            cs = sum(cand_self[it] * my_pts[it] for it in ITEMS)
            co = sum(cand_opp[it] * opp_pts[it] for it in ITEMS)
            if cs >= u_self and co >= u_opp and (cs > u_self or co > u_opp):
                dominated = True
                break
        vals.append(0.0 if dominated else 1.0)
    return {
        "support": len(vals),
        "pareto_efficient_rate": float(np.mean(vals)) if vals else float("nan"),
    }


def stamp_dialogues_with_indices(
    dialogues: Sequence[Mapping[str, Any]],
    annotations_by_dialogue: Optional[Mapping[Any, Sequence[Any]]] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for d in dialogues:
        did = d.get("dialogue_id")
        anns = (
            (annotations_by_dialogue or {}).get(did)
            or (annotations_by_dialogue or {}).get(str(did))
            or d.get("annotations")
            or []
        )
        ann_lookup = build_annotation_lookup(anns, d.get("chat_logs", [])) if anns else {}
        chat = []
        for i, turn in enumerate(d.get("chat_logs", [])):
            t = dict(turn)
            t["dialogue_id"] = did
            t["ablation_turn_index"] = i
            if i in ann_lookup:
                t["ablation_strategy_labels"] = list(ann_lookup[i])
            chat.append(t)
        nd = dict(d)
        nd["chat_logs"] = chat
        out.append(nd)
    return out


def load_annotations_lookup(path: Optional[Path]) -> Dict[Any, Any]:
    if path is None:
        return {}
    data = json.load(path.open())
    return {d["dialogue_id"]: d.get("annotations", []) for d in data}


def filter_annotated_dialogues(
    dialogues: Sequence[Mapping[str, Any]],
    annotations_by_dialogue: Mapping[Any, Sequence[Any]],
) -> List[Mapping[str, Any]]:
    return [
        d for d in dialogues
        if annotations_by_dialogue.get(d.get("dialogue_id"))
        or annotations_by_dialogue.get(str(d.get("dialogue_id")))
        or d.get("annotations")
    ]


def annotation_support_counts(dialogues: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    annotated_dialogues = 0
    matched_natural_rows = 0
    natural_rows = 0
    for d in dialogues:
        has_any = False
        for turn in d.get("chat_logs", []):
            text = str(turn.get("text", "")).strip()
            if not text or text in DEAL_ACTIONS:
                continue
            natural_rows += 1
            if turn.get("ablation_strategy_labels") is not None:
                matched_natural_rows += 1
                has_any = True
        if has_any:
            annotated_dialogues += 1
    return {
        "annotated_dialogues": annotated_dialogues,
        "natural_rows": natural_rows,
        "matched_annotation_rows": matched_natural_rows,
    }


def build_provider(args: argparse.Namespace, renderer: EvidenceRenderer) -> PosteriorProvider:
    if args.provider == "uniform":
        return UniformProvider()
    if args.provider in {"mc_k16_full_context", "mc_k1_full_context"}:
        from casino_belief.student.predict import SftModelFn

        model = SftModelFn(
            base_model=args.base_model,
            adapter_path=args.adapter,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        return MCPProvider(
            model,
            K=16 if args.provider == "mc_k16_full_context" else 1,
            temperature=args.posterior_temperature,
            renderer=renderer,
        )
    if args.provider in {"direct_zero_shot", "direct_sft_groundtruth", "direct_sft_teacher"}:
        adapter = args.direct_adapter if args.provider != "direct_zero_shot" else args.adapter
        model = ChatModel(
            base_model=args.base_model,
            adapter_path=adapter,
            max_new_tokens=args.direct_max_new_tokens,
            temperature=args.temperature,
        )
        return DirectPosteriorProvider(model, renderer=renderer)
    if args.provider == "incremental_bayes_utterance":
        model = ChatModel(
            base_model=args.base_model,
            adapter_path=args.adapter,
            max_new_tokens=96,
            temperature=args.temperature,
        )
        return IncrementalBayesProvider(
            model,
            likelihood_temperature=args.likelihood_temperature,
            likelihood_clip=args.likelihood_clip,
        )
    if args.provider == "rule_likelihood":
        return RuleLikelihoodProvider()
    raise ValueError(f"unknown provider {args.provider!r}")


# ── CLI ───────────────────────────────────────────────────────────────────


def _setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger_ = logging.getLogger("casino_belief.diagnostics.ablation.ablation_run")
    logger_.setLevel(logging.INFO)
    logger_.propagate = False
    logger_.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger_.addHandler(fh)
    logger_.addHandler(sh)
    return logger_


def _parse_clip(raw: str) -> Tuple[Optional[float], Optional[float]]:
    if raw.lower() in {"none", "off", ""}:
        return (None, None)
    lo, hi = raw.split(",", 1)
    return (float(lo), float(hi))


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/casino/casino_test.json")
    p.add_argument("--annotations", default="external/casino_original/data/casino_ann.json")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--provider", choices=sorted(PROVIDER_CHOICES), required=True)
    p.add_argument("--evidence-mode", choices=sorted(EVIDENCE_MODES), default="utterance_only")
    p.add_argument("--annotated-only", action="store_true")
    p.add_argument("--max-dialogues", type=int, default=None)
    p.add_argument("--perspectives", default="mturk_agent_1")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--adapter", default=DEFAULT_TEACHER_ADAPTER)
    p.add_argument("--direct-adapter", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--posterior-temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--direct-max-new-tokens", type=int, default=256)
    p.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    p.add_argument("--accept-margin", type=int, default=5)
    p.add_argument("--accept-floor", type=float, default=0.50)
    p.add_argument("--likelihood-temperature", type=float, default=25.0)
    p.add_argument("--likelihood-clip", type=_parse_clip, default=(-3.0, 3.0))
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logging(out_dir / "ablation_eval.log")
    log.info("Args: %s", vars(args))

    dialogues = json.load(open(args.data))
    ann_lookup = load_annotations_lookup(Path(args.annotations) if args.annotations else None)
    if args.annotated_only:
        dialogues = list(filter_annotated_dialogues(dialogues, ann_lookup))
        log.info("Restricted to %d annotated dialogues.", len(dialogues))
    if args.max_dialogues is not None:
        dialogues = dialogues[: args.max_dialogues]
    dialogues = stamp_dialogues_with_indices(dialogues, ann_lookup)

    renderer = EvidenceRenderer(mode=args.evidence_mode, annotations_by_dialogue=ann_lookup)
    provider = build_provider(args, renderer)
    agent = AblationTeacherTurnAgent(
        provider,
        lambda_=args.lambda_,
        accept_margin=args.accept_margin,
        accept_floor=args.accept_floor,
    )
    records_path = out_dir / "turn_records.jsonl"
    if records_path.exists():
        records_path.unlink()
    records: List[TurnRecord] = []

    def _on_record(r: TurnRecord) -> None:
        records.append(r)
        with records_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(r), default=str) + "\n")

    perspectives = tuple(p.strip() for p in args.perspectives.split(",") if p.strip())
    t0 = time.time()
    result = turn_level_eval(
        dialogues=dialogues,
        agent=agent,
        perspectives=perspectives,
        annotations_by_dialogue=ann_lookup,
        on_record=_on_record,
    )
    elapsed = time.time() - t0
    result["posterior_diagnostics"] = posterior_diagnostics(records)
    result["brier_by_strategy_label"] = brier_by_strategy_diagnostics(records)
    result["pareto_efficiency"] = pareto_efficiency_diagnostics(records, dialogues)
    log.info("Eval finished in %.1fs", elapsed)
    log.info("\n%s", format_turn_level_summary(result))

    summary = {
        "config": vars(args),
        "n_dialogues": len(dialogues),
        "n_records": result["n_records"],
        "elapsed_seconds": elapsed,
        "accept": result["accept"],
        "bid_cosine": result["bid_cosine"],
        "strategy_macro_f1": result["strategy_macro_f1"],
        "brier": result["brier"],
        "brier_by_turn_index": result["brier_by_turn_index"],
        "posterior_diagnostics": result["posterior_diagnostics"],
        "brier_by_strategy_label": result["brier_by_strategy_label"],
        "pareto_efficiency": result["pareto_efficiency"],
        "annotation_support": annotation_support_counts(dialogues),
        "agent_summary": agent.summary,
    }
    with (out_dir / "turn_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("Wrote %s", out_dir / "turn_summary.json")
    log.info("Wrote %s", records_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
