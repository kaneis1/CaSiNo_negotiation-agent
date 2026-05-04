"""Student-schema ablation evaluator and A2d prefix-injection harness."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from casino_belief.evaluation.hypotheses import ITEMS
from casino_belief.evaluation.turn_agents import KeywordStrategyClassifier
from casino_belief.evaluation.turn_level_metrics import (
    TurnRecord,
    format_turn_level_summary,
    turn_level_eval,
)
from casino_belief.diagnostics.ablation.ablation import (
    DEFAULT_BASE_MODEL,
    DEFAULT_OUTPUT_ROOT,
    ChatModel,
    annotation_support_counts,
    brier_by_strategy_diagnostics,
    filter_annotated_dialogues,
    load_annotations_lookup,
    normalize_posterior,
    pareto_efficiency_diagnostics,
    posterior_diagnostics,
    stamp_dialogues_with_indices,
)
from casino_belief.training.build_ablation_sft_data import (
    ACTION_ONLY_SYSTEM_PROMPT,
    MAP_ONLY_SYSTEM_PROMPT,
    REVERSED_SYSTEM_PROMPT,
)
from casino_belief.training.build_distill_data import format_history, format_priorities, format_reasons
from casino_belief.belief.posterior import N_ORDERINGS, ORDERINGS
from casino_belief.student.student_parser import (
    VALID_INTENTS,
    normalize_selected_content,
    parse_posterior_block,
    parse_student_response,
)
from casino_belief.student.student_prompts import STUDENT_SYSTEM_PROMPT, build_student_user_prompt, format_posterior

logger = logging.getLogger("casino_belief.diagnostics.ablation.ablation_student")

DEFAULT_STUDENT_ADAPTER = "artifacts/training_metadata/day8_lora_run/lora_best"
SCHEMA_CHOICES = {"full", "action_only", "map_only", "reversed"}
PREFIX_MODES = {"none", "emitted", "correct", "adversarial", "uniform", "random"}


def _extract_tag(text: str, tag: str) -> Optional[str]:
    match = re.search(rf"<{re.escape(tag)}\s*>(.*?)</{re.escape(tag)}\s*>", text or "", re.S | re.I)
    if match:
        return match.group(1).strip()
    open_match = re.search(rf"<{re.escape(tag)}\s*>", text or "", re.I)
    if not open_match:
        return None
    tail = (text or "")[open_match.end() :]
    next_pos = None
    for other in ("posterior", "selected_intent", "selected_content", "utterance"):
        if other == tag:
            continue
        m = re.search(rf"<{re.escape(other)}\s*>", tail, re.I)
        if m is not None and (next_pos is None or m.start() < next_pos):
            next_pos = m.start()
    body = tail if next_pos is None else tail[:next_pos]
    return body.strip() or None


def _extract_json_blob(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(raw[start:], start=start):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(raw[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def _parse_selected_content(body: Optional[str], intent: Optional[str]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    if body is None:
        return None, ["missing selected_content"]
    raw = body.strip()
    if raw.lower() == "null":
        content = None
        errs: List[str] = []
    else:
        blob = _extract_json_blob(raw)
        if blob is None:
            return None, ["could not extract JSON object from <selected_content>"]
        content, errs = normalize_selected_content(blob)
    if intent == "submit" and content is None:
        errs.append("submit intent requires non-null selected_content")
    if intent in {"accept", "walkaway", "utter"} and content is not None:
        errs.append(f"{intent} intent should have selected_content=null")
    return content, errs


def _parse_map_posterior(body: Optional[str]) -> Tuple[Optional[List[float]], List[str]]:
    if body is None:
        return None, ["missing posterior"]
    match = re.search(r"(?:MAP\s*:\s*)?(.+?)\s*$", body.strip(), re.I | re.S)
    if not match:
        return None, ["missing MAP ordering"]
    parts = tuple(p.strip().capitalize() for p in match.group(1).split(">"))
    if len(parts) != 3 or set(parts) != set(ITEMS):
        return None, [f"MAP ordering is not a permutation of items: {match.group(1)!r}"]
    arr = np.zeros(N_ORDERINGS, dtype=np.float64)
    try:
        arr[ORDERINGS.index(parts)] = 1.0
    except ValueError:
        return None, [f"unknown MAP ordering: {parts!r}"]
    return arr.tolist(), []


def parse_student_response_schema(text: str, *, schema: str) -> Dict[str, Any]:
    """Parse full/action-only/MAP/reversed student generations."""
    if schema not in SCHEMA_CHOICES:
        raise ValueError(f"unknown student schema {schema!r}")
    if schema == "full":
        return parse_student_response(text)

    result: Dict[str, Any] = {
        "posterior": None,
        "selected_intent": None,
        "selected_content": None,
        "utterance": _extract_tag(text, "utterance"),
        "posterior_raw": _extract_tag(text, "posterior"),
        "selected_content_raw": _extract_tag(text, "selected_content"),
        "parse_error": None,
        "missing_tags": [],
        "posterior_errors": [],
        "intent_errors": [],
        "selected_content_errors": [],
    }
    required = ["selected_intent", "selected_content", "utterance"]
    if schema in {"map_only", "reversed"}:
        required.append("posterior")
    blocks = {tag: _extract_tag(text, tag) for tag in set(required)}
    result["missing_tags"] = [tag for tag in required if blocks.get(tag) is None]

    if blocks.get("selected_intent") is not None:
        intent = str(blocks["selected_intent"]).strip().lower()
        if intent in VALID_INTENTS:
            result["selected_intent"] = intent
        else:
            result["intent_errors"].append(f"selected_intent must be one of {sorted(VALID_INTENTS)}, got {intent!r}")

    if schema == "map_only":
        posterior, errs = _parse_map_posterior(blocks.get("posterior"))
        result["posterior"] = posterior
        result["posterior_errors"] = errs
    elif schema == "reversed":
        posterior, errs = parse_posterior_block(blocks.get("posterior") or "")
        result["posterior"] = posterior
        result["posterior_errors"] = errs

    content, errs = _parse_selected_content(
        blocks.get("selected_content"),
        result.get("selected_intent"),
    )
    result["selected_content"] = content
    result["selected_content_errors"] = errs

    parts: List[str] = []
    if result["missing_tags"]:
        parts.append(f"missing tags: {result['missing_tags']}")
    if result["posterior_errors"]:
        parts.append(f"posterior errors: {result['posterior_errors']}")
    if result["intent_errors"]:
        parts.append(f"intent errors: {result['intent_errors']}")
    if result["selected_content_errors"]:
        parts.append(f"selected_content errors: {result['selected_content_errors']}")
    if parts:
        result["parse_error"] = "; ".join(parts)
    return result


def system_prompt_for_schema(schema: str) -> str:
    if schema == "action_only":
        return ACTION_ONLY_SYSTEM_PROMPT
    if schema == "map_only":
        return MAP_ONLY_SYSTEM_PROMPT
    if schema == "reversed":
        return REVERSED_SYSTEM_PROMPT
    return STUDENT_SYSTEM_PROMPT


def posterior_prefix(posterior: Sequence[float]) -> str:
    arr = normalize_posterior(posterior)
    return "<posterior>\n" + format_posterior(arr, ORDERINGS) + "\n</posterior>\n"


def true_posterior_for_dialogue(
    dialogue: Mapping[str, Any],
    *,
    opp_role: str,
    smoothing: Optional[float] = None,
) -> np.ndarray:
    pri = dialogue["participant_info"][opp_role]["value2issue"]
    ordering = (pri["High"], pri["Medium"], pri["Low"])
    arr = np.zeros(N_ORDERINGS, dtype=np.float64)
    arr[ORDERINGS.index(ordering)] = 1.0
    if smoothing is not None:
        arr[:] = (1.0 - float(smoothing)) / (N_ORDERINGS - 1)
        arr[ORDERINGS.index(ordering)] = float(smoothing)
    return arr


def adversarial_posterior_for_dialogue(dialogue: Mapping[str, Any], *, opp_role: str) -> np.ndarray:
    pri = dialogue["participant_info"][opp_role]["value2issue"]
    reversed_order = (pri["Low"], pri["Medium"], pri["High"])
    arr = np.zeros(N_ORDERINGS, dtype=np.float64)
    arr[ORDERINGS.index(reversed_order)] = 1.0
    return arr


class StudentSchemaModel:
    def __init__(
        self,
        *,
        base_model: str,
        adapter_path: Optional[str],
        schema: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.schema = schema
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.chat = ChatModel(
            base_model=base_model,
            adapter_path=adapter_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        self.last_raw_response = ""
        self.last_parse: Dict[str, Any] = {}

    def _user_prompt(
        self,
        *,
        history: Sequence[Mapping[str, Any]],
        my_role: str,
        my_priorities: Mapping[str, str],
        my_reasons: Mapping[str, str],
        style: str,
    ) -> str:
        return build_student_user_prompt(
            self_priorities=format_priorities(my_priorities),
            self_reasons=format_reasons(my_reasons),
            history=format_history(history, perspective=my_role),
            style=style,
        )

    def predict(
        self,
        *,
        history: Sequence[Mapping[str, Any]],
        my_role: str,
        my_priorities: Mapping[str, str],
        my_reasons: Mapping[str, str],
        style: str,
        assistant_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        raw = self.chat.generate(
            system_prompt=system_prompt_for_schema(self.schema),
            user_prompt=self._user_prompt(
                history=history,
                my_role=my_role,
                my_priorities=my_priorities,
                my_reasons=my_reasons,
                style=style,
            ),
            assistant_prefix=assistant_prefix,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        self.last_raw_response = raw
        parsed = parse_student_response_schema(raw, schema=self.schema)
        self.last_parse = parsed
        return parsed


def _bid_from_content(content: Any) -> Optional[Dict[str, int]]:
    if not isinstance(content, Mapping):
        return None
    self_counts = content.get("self_counts")
    if isinstance(self_counts, Mapping):
        try:
            return {it: int(self_counts[it]) for it in ITEMS}
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _bid_tuple(parsed: Mapping[str, Any]) -> Optional[Tuple[int, int, int]]:
    bid = _bid_from_content(parsed.get("selected_content"))
    if bid is None:
        return None
    return tuple(int(bid[it]) for it in ITEMS)


class AblationStudentTurnAgent:
    def __init__(
        self,
        student_model: Any,
        *,
        schema: str,
        dialogues_by_id: Mapping[str, Mapping[str, Any]],
        style: str = "balanced",
        prefix_mode: str = "none",
        sanity_compare_prefix: bool = False,
        random_seed: int = 2026,
        strategy_classifier: Optional[Any] = None,
    ) -> None:
        self.student_model = student_model
        self.schema = schema
        self.dialogues_by_id = dialogues_by_id
        self.style = style
        self.prefix_mode = prefix_mode
        self.sanity_compare_prefix = bool(sanity_compare_prefix)
        self.rng = random.Random(random_seed)
        self.strategy_classifier = strategy_classifier or KeywordStrategyClassifier()
        self.last_raw_response = ""
        self.last_parse: Dict[str, Any] = {}
        self._summary: Dict[str, Any] = {
            "calls": 0,
            "parse_errors": 0,
            "posterior_ok": 0,
            "intent_ok": 0,
            "content_ok": 0,
            "bid_emitted": 0,
            "prefix_mode": prefix_mode,
            "prefix_calls": 0,
            "prefix_same_intent": 0,
            "prefix_same_bid": 0,
            "prefix_action_or_bid_changed": 0,
            "prefix_mean_posterior_l1": [],
        }

    @property
    def summary(self) -> Dict[str, Any]:
        out = dict(self._summary)
        vals = out.pop("prefix_mean_posterior_l1", [])
        out["prefix_posterior_l1_mean"] = float(np.mean(vals)) if vals else float("nan")
        calls = int(out.get("prefix_calls") or 0)
        if calls:
            out["prefix_same_intent_rate"] = out["prefix_same_intent"] / calls
            out["prefix_same_bid_rate"] = out["prefix_same_bid"] / calls
            out["prefix_action_or_bid_changed_rate"] = (
                out["prefix_action_or_bid_changed"] / calls
            )
            out["mechanically_confounded"] = (
                self.prefix_mode in {"emitted", "correct"}
                and out["prefix_action_or_bid_changed_rate"] > 0.10
            )
        else:
            out["prefix_same_intent_rate"] = float("nan")
            out["prefix_same_bid_rate"] = float("nan")
            out["prefix_action_or_bid_changed_rate"] = float("nan")
            out["mechanically_confounded"] = None
        return out

    def _dialogue(self, dialogue_id: Any) -> Mapping[str, Any]:
        return self.dialogues_by_id[str(dialogue_id)]

    def _posterior_for_mode(
        self,
        *,
        mode: str,
        dialogue_id: Any,
        opp_role: str,
        unconstrained: Optional[Mapping[str, Any]],
    ) -> Optional[np.ndarray]:
        if mode == "none":
            return None
        if mode == "emitted":
            p = (unconstrained or {}).get("posterior")
            return normalize_posterior(p if p is not None else [1.0 / N_ORDERINGS] * N_ORDERINGS)
        dialogue = self._dialogue(dialogue_id)
        if mode == "correct":
            return true_posterior_for_dialogue(dialogue, opp_role=opp_role)
        if mode == "adversarial":
            return adversarial_posterior_for_dialogue(dialogue, opp_role=opp_role)
        if mode == "uniform":
            return np.full(N_ORDERINGS, 1.0 / N_ORDERINGS, dtype=np.float64)
        if mode == "random":
            vals = [self.rng.expovariate(1.0) for _ in range(N_ORDERINGS)]
            return normalize_posterior(vals)
        raise ValueError(f"unknown prefix mode {mode!r}")

    def _record_prefix_comparison(
        self,
        *,
        unconstrained: Mapping[str, Any],
        constrained: Mapping[str, Any],
    ) -> None:
        self._summary["prefix_calls"] += 1
        if unconstrained.get("selected_intent") == constrained.get("selected_intent"):
            self._summary["prefix_same_intent"] += 1
        if _bid_tuple(unconstrained) == _bid_tuple(constrained):
            self._summary["prefix_same_bid"] += 1
        if (
            unconstrained.get("selected_intent") != constrained.get("selected_intent")
            or _bid_tuple(unconstrained) != _bid_tuple(constrained)
        ):
            self._summary["prefix_action_or_bid_changed"] += 1
        pu = unconstrained.get("posterior")
        pc = constrained.get("posterior")
        if pu is not None and pc is not None:
            self._summary["prefix_mean_posterior_l1"].append(
                float(np.abs(normalize_posterior(pu) - normalize_posterior(pc)).sum())
            )

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
        unconstrained: Optional[Dict[str, Any]] = None
        parsed: Dict[str, Any]

        if self.prefix_mode == "none":
            parsed = self.student_model.predict(
                history=history,
                my_role=my_role,
                my_priorities=my_priorities,
                my_reasons=my_reasons,
                style=self.style,
            )
        else:
            if self.prefix_mode == "emitted" or self.sanity_compare_prefix:
                unconstrained = self.student_model.predict(
                    history=history,
                    my_role=my_role,
                    my_priorities=my_priorities,
                    my_reasons=my_reasons,
                    style=self.style,
                )
            forced = self._posterior_for_mode(
                mode=self.prefix_mode,
                dialogue_id=dialogue_id,
                opp_role=opp_role,
                unconstrained=unconstrained,
            )
            prefix = posterior_prefix(forced if forced is not None else [1.0 / N_ORDERINGS] * N_ORDERINGS)
            parsed = self.student_model.predict(
                history=history,
                my_role=my_role,
                my_priorities=my_priorities,
                my_reasons=my_reasons,
                style=self.style,
                assistant_prefix=prefix,
            )
            if unconstrained is not None:
                self._record_prefix_comparison(
                    unconstrained=unconstrained,
                    constrained=parsed,
                )

        raw_response = str(getattr(self.student_model, "last_raw_response", ""))
        self.last_raw_response = raw_response
        self.last_parse = dict(parsed)
        if parsed.get("parse_error"):
            self._summary["parse_errors"] += 1
        if parsed.get("posterior") is not None:
            self._summary["posterior_ok"] += 1
        if parsed.get("selected_intent") is not None:
            self._summary["intent_ok"] += 1
        if parsed.get("selected_content") is not None:
            self._summary["content_ok"] += 1

        intent = parsed.get("selected_intent")
        accept: Optional[bool]
        if intent == "accept":
            accept = True
        elif intent in {"reject", "walkaway"}:
            accept = False
        else:
            accept = None
        bid = _bid_from_content(parsed.get("selected_content")) if intent in {"submit", "reject"} else None
        if bid is not None:
            self._summary["bid_emitted"] += 1
        utterance = str(parsed.get("utterance") or "").strip()
        strategy = None
        if utterance:
            try:
                strategy = list(self.strategy_classifier(utterance, list(history))) or None
            except Exception:
                logger.exception("strategy classifier failed on student utterance.")
        return {
            "accept": accept,
            "bid": bid,
            "utterance": utterance or None,
            "action": intent,
            "style": self.style,
            "strategy": strategy,
            "posterior": parsed.get("posterior"),
        }


def _setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger_ = logging.getLogger("casino_belief.diagnostics.ablation.ablation_student_run")
    logger_.setLevel(logging.INFO)
    logger_.propagate = False
    logger_.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for handler in (logging.FileHandler(log_path, mode="a"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        logger_.addHandler(handler)
    return logger_


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/casino/casino_test.json")
    p.add_argument("--annotations", default="external/casino_original/data/casino_ann.json")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--schema", choices=sorted(SCHEMA_CHOICES), required=True)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--adapter", default=DEFAULT_STUDENT_ADAPTER)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--style-token", default="balanced")
    p.add_argument("--prefix-mode", choices=sorted(PREFIX_MODES), default="none")
    p.add_argument("--sanity-compare-prefix", action="store_true")
    p.add_argument("--random-seed", type=int, default=2026)
    p.add_argument("--annotated-only", action="store_true")
    p.add_argument("--max-dialogues", type=int, default=None)
    p.add_argument("--perspectives", default="mturk_agent_1")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logging(out_dir / "student_ablation_eval.log")
    log.info("Args: %s", vars(args))

    dialogues = json.load(open(args.data))
    ann_lookup = load_annotations_lookup(Path(args.annotations) if args.annotations else None)
    if args.annotated_only:
        dialogues = list(filter_annotated_dialogues(dialogues, ann_lookup))
        log.info("Restricted to %d annotated dialogues.", len(dialogues))
    if args.max_dialogues is not None:
        dialogues = dialogues[: args.max_dialogues]
    dialogues = stamp_dialogues_with_indices(dialogues, ann_lookup)
    d_by_id = {str(d.get("dialogue_id")): d for d in dialogues}

    model = StudentSchemaModel(
        base_model=args.base_model,
        adapter_path=args.adapter,
        schema=args.schema,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    agent = AblationStudentTurnAgent(
        model,
        schema=args.schema,
        dialogues_by_id=d_by_id,
        style=args.style_token,
        prefix_mode=args.prefix_mode,
        sanity_compare_prefix=args.sanity_compare_prefix,
        random_seed=args.random_seed,
    )
    records: List[TurnRecord] = []
    records_path = out_dir / "turn_records.jsonl"
    if records_path.exists():
        records_path.unlink()

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
