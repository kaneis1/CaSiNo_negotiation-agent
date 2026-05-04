"""Run paper-locked structured opponent-inference evaluation.

Regime A produces Chawla-style k-prefix ranking metrics for Table 1.
Regime B produces turn-level self-consistency posteriors for Table 2.
Both use one locally loaded HF ``transformers.pipeline`` and append-only
checkpoints so the LSF job can resume cleanly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as md
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from casino_belief.evaluation.hypotheses import HYPOTHESES, ITEMS
from casino_belief.evaluation.metrics import ema, ndcg_at_3, top1
from casino_belief.evaluation.turn_level_metrics import build_annotation_lookup, normalized_brier
from casino_belief.baselines.structured_cot.opponent_inference import (
    DEAL_ACTIONS,
    build_ranking_messages,
    loose_action_consistent,
    parse_ranking_response,
    posterior_from_sample_indices,
    priorities_to_ordering,
    strict_action_consistent,
)


LOGGER = logging.getLogger("casino_belief.baselines.structured_cot.run_structured_opp_inference")

DEFAULT_OUTPUT_DIR = Path(
    "artifacts/results/posterior_quality/structured_cot_elicited_posterior"
)
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_REVISION = "6f6073b423013f6a7d4d9f39144961bfbfbc386b"
DEFAULT_SNAPSHOT_PATH = Path(
    "/sc/arion/scratch/cuiz02/hf_cache/transformers/"
    "models--meta-llama--Llama-3.3-70B-Instruct/snapshots/"
    "6f6073b423013f6a7d4d9f39144961bfbfbc386b"
)
DEFAULT_MATCHED_TURN_RECORDS = Path(
    "artifacts/results/protocol3/bayesian_teacher_full150/"
    "turn_records.jsonl"
)
DEFAULT_BASELINE_TURN_RECORDS = Path(
    "artifacts/results/protocol3/structured_cot_70b_full150/"
    "turn_records.jsonl"
)

DETERMINISTIC_KWARGS = {
    "do_sample": False,
    "max_new_tokens": 512,
    "return_full_text": False,
}
SELF_CONSISTENCY_KWARGS = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "num_return_sequences": 5,
    "max_new_tokens": 512,
    "return_full_text": False,
}
KPENALTY_WEIGHTS = {1: 5.0 / 15.0, 2: 4.0 / 15.0, 3: 3.0 / 15.0, 4: 2.0 / 15.0, 5: 1.0 / 15.0}


RegimeAKey = Tuple[Any, str, int]
RegimeBKey = Tuple[Any, str, int]


def _setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "run.log", mode="a"),
        ],
    )


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _package_version(name: str) -> Optional[str]:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def _load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _role_pair(perspective: str) -> str:
    return "mturk_agent_2" if perspective == "mturk_agent_1" else "mturk_agent_1"


def _is_natural(turn: Mapping[str, Any]) -> bool:
    return str(turn.get("text") or "").strip() not in DEAL_ACTIONS


def _stamp_turn(turn: Mapping[str, Any], dialogue_id: Any) -> Dict[str, Any]:
    out = dict(turn)
    out.setdefault("dialogue_id", dialogue_id)
    return out


def _dialogues_by_id(dialogues: Sequence[Mapping[str, Any]]) -> Dict[Any, Mapping[str, Any]]:
    return {d.get("dialogue_id"): d for d in dialogues}


def _coerce_self_counts_from_bid_vector(bid: Any) -> Optional[Dict[str, int]]:
    if bid is None:
        return None
    if isinstance(bid, Mapping):
        if all(item in bid for item in ITEMS):
            try:
                return {item: int(bid[item]) for item in ITEMS}
            except (TypeError, ValueError):
                return None
        return None
    if isinstance(bid, (list, tuple)) and len(bid) >= 3:
        try:
            return {item: int(float(bid[i])) for i, item in enumerate(ITEMS)}
        except (TypeError, ValueError):
            return None
    return None


def _accepted_offer_self_counts(rec: Mapping[str, Any]) -> Optional[Dict[str, int]]:
    pending = rec.get("pending_offer") or {}
    if not pending or not pending.get("to_perspective"):
        return None
    td = pending.get("task_data") or {}
    proposer = pending.get("proposer")
    perspective = rec.get("perspective")
    source = td.get("issue2youget") if proposer == perspective else td.get("issue2theyget")
    if not isinstance(source, Mapping):
        return None
    try:
        return {item: int(source.get(item, 0)) for item in ITEMS}
    except (TypeError, ValueError):
        return None


def _record_key_a(row: Mapping[str, Any]) -> RegimeAKey:
    return (row.get("dialogue_id"), str(row.get("perspective")), int(row.get("k")))


def _record_key_b(row: Mapping[str, Any]) -> RegimeBKey:
    return (row.get("dialogue_id"), str(row.get("perspective")), int(row.get("turn_index")))


def load_jsonl_by_key(path: Path, key_fn) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    out: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[key_fn(row)] = row
    return out


class PipelineGenerator:
    def __init__(
        self,
        *,
        model: str,
        revision: Optional[str],
    ) -> None:
        import torch
        import transformers

        kwargs: Dict[str, Any] = {
            "model": model,
            "model_kwargs": {"torch_dtype": torch.bfloat16},
            "device_map": "auto",
        }
        if revision:
            kwargs["revision"] = revision
        self.pipe = transformers.pipeline("text-generation", **kwargs)
        if self.pipe.tokenizer.pad_token_id is None:
            self.pipe.tokenizer.pad_token_id = self.pipe.tokenizer.eos_token_id
        model_obj = getattr(self.pipe, "model", None)
        model_config = getattr(model_obj, "config", None)
        self.resolved_name_or_path = getattr(model_config, "_name_or_path", None)

    def generate(self, messages: Sequence[Mapping[str, str]], kwargs: Mapping[str, Any]) -> List[str]:
        prompt = self.pipe.tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
        outputs = self.pipe(prompt, **dict(kwargs))
        if outputs and isinstance(outputs[0], list):
            outputs = outputs[0]
        texts: List[str] = []
        for out in outputs:
            if isinstance(out, Mapping):
                texts.append(str(out.get("generated_text", "")).strip())
            else:
                texts.append(str(out).strip())
        return texts


def _parse_ranking(raw_output: str) -> Dict[str, Any]:
    parsed = parse_ranking_response(raw_output)
    return {
        "valid": parsed.get("parse_error") is None,
        "parse_error": parsed.get("parse_error"),
        "ranking_raw": parsed.get("ranking_raw"),
        "ranking": parsed.get("ranking"),
        "confidence": parsed.get("confidence"),
        "ordering": parsed.get("ordering"),
        "hypothesis_index": parsed.get("hypothesis_index"),
        "missing_tags": parsed.get("missing_tags") or [],
        "ranking_errors": parsed.get("ranking_errors") or [],
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, default=_json_default) + "\n")
        f.flush()


def iter_regime_a_snapshots(
    dialogues: Sequence[Mapping[str, Any]],
    *,
    perspective: str,
    max_dialogues: Optional[int] = None,
    max_k: int = 5,
) -> Iterable[Dict[str, Any]]:
    """Yield Chawla-style first-k-opponent-natural-utterance snapshots."""
    selected = dialogues[:max_dialogues] if max_dialogues is not None else dialogues
    opp_role = _role_pair(perspective)
    for dialogue in selected:
        did = dialogue.get("dialogue_id")
        pinfo = dialogue.get("participant_info") or {}
        if perspective not in pinfo or opp_role not in pinfo:
            continue
        my_priorities = dict(pinfo[perspective].get("value2issue") or {})
        my_reasons = dict(pinfo[perspective].get("value2reason") or {})
        opp_priorities = dict(pinfo[opp_role].get("value2issue") or {})
        true_ordering = priorities_to_ordering(opp_priorities)

        prefix: List[Dict[str, Any]] = []
        opp_seen = 0
        yielded = set()
        for turn in dialogue.get("chat_logs") or []:
            if not _is_natural(turn):
                continue
            stamped = _stamp_turn(turn, did)
            prefix.append(stamped)
            if turn.get("id") != opp_role:
                continue
            opp_seen += 1
            if opp_seen > max_k:
                break
            yielded.add(opp_seen)
            yield {
                "dialogue_id": did,
                "perspective": perspective,
                "opp_role": opp_role,
                "k": opp_seen,
                "messages": build_ranking_messages(
                    history=prefix,
                    my_role=perspective,
                    opp_role=opp_role,
                    my_priorities=my_priorities,
                    my_reasons=my_reasons,
                ),
                "my_priorities": my_priorities,
                "true_ordering": true_ordering,
                "true_hypothesis_index": _hypothesis_index(true_ordering),
            }
        missing = [k for k in range(1, max_k + 1) if k not in yielded]
        if missing:
            LOGGER.debug("dialogue %s missing Regime A k values %s", did, missing)


def _hypothesis_index(ordering: Sequence[str]) -> Optional[int]:
    target = tuple(ordering)
    for idx, hyp in enumerate(HYPOTHESES):
        if tuple(hyp) == target:
            return idx
    return None


def load_turn_record_keys(path: Path, *, perspective: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"matched turn-records file not found: {path}")
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("perspective") == perspective:
                rows.append(row)
    rows.sort(key=lambda r: (str(r.get("dialogue_id")), int(r.get("turn_index"))))
    return rows


def iter_regime_b_snapshots(
    dialogues: Sequence[Mapping[str, Any]],
    *,
    matched_turn_records: Sequence[Mapping[str, Any]],
) -> Iterable[Dict[str, Any]]:
    dialogues_by_id = _dialogues_by_id(dialogues)
    for rec in matched_turn_records:
        did = rec.get("dialogue_id")
        perspective = str(rec.get("perspective"))
        opp_role = str(rec.get("opp_role") or _role_pair(perspective))
        turn_index = int(rec.get("turn_index"))
        dialogue = dialogues_by_id.get(did)
        if dialogue is None:
            continue
        pinfo = dialogue.get("participant_info") or {}
        if perspective not in pinfo or opp_role not in pinfo:
            continue
        my_priorities = dict(pinfo[perspective].get("value2issue") or {})
        my_reasons = dict(pinfo[perspective].get("value2reason") or {})
        true_ordering = list((rec.get("true") or {}).get("true_ordering") or priorities_to_ordering(pinfo[opp_role].get("value2issue") or {}))
        history = [
            _stamp_turn(t, did)
            for t in (dialogue.get("chat_logs") or [])[:turn_index]
        ]
        yield {
            "dialogue_id": did,
            "perspective": perspective,
            "opp_role": opp_role,
            "turn_index": turn_index,
            "messages": build_ranking_messages(
                history=history,
                my_role=perspective,
                opp_role=opp_role,
                my_priorities=my_priorities,
                my_reasons=my_reasons,
            ),
            "my_priorities": my_priorities,
            "true_ordering": true_ordering,
            "true_hypothesis_index": (rec.get("true") or {}).get("true_hypothesis_index"),
        }


def run_regime_a(
    *,
    generator: PipelineGenerator,
    snapshots: Sequence[Mapping[str, Any]],
    output_path: Path,
    existing: Dict[RegimeAKey, Dict[str, Any]],
) -> Dict[str, int]:
    stats = {"dispatched": 0, "skipped": 0}
    for idx, snap in enumerate(snapshots, start=1):
        if idx % 25 == 1:
            LOGGER.info("Regime A snapshot %d/%d", idx, len(snapshots))
        key = (snap["dialogue_id"], str(snap["perspective"]), int(snap["k"]))
        if key in existing:
            stats["skipped"] += 1
            continue
        t0 = time.time()
        outputs = generator.generate(snap["messages"], DETERMINISTIC_KWARGS)
        elapsed = time.time() - t0
        raw = outputs[0] if outputs else ""
        row = {
            "regime": "A",
            "dialogue_id": snap["dialogue_id"],
            "perspective": snap["perspective"],
            "opp_role": snap["opp_role"],
            "k": int(snap["k"]),
            "elapsed_seconds": elapsed,
            "my_priorities": snap["my_priorities"],
            "true_ordering": snap["true_ordering"],
            "true_hypothesis_index": snap["true_hypothesis_index"],
            "raw_output": raw,
            **_parse_ranking(raw),
        }
        _write_jsonl(output_path, [row])
        existing[key] = row
        stats["dispatched"] += 1
    return stats


def run_regime_b(
    *,
    generator: PipelineGenerator,
    snapshots: Sequence[Mapping[str, Any]],
    output_path: Path,
    existing: Dict[RegimeBKey, Dict[str, Any]],
) -> Dict[str, int]:
    stats = {"dispatched": 0, "skipped": 0}
    for idx, snap in enumerate(snapshots, start=1):
        if idx % 25 == 1:
            LOGGER.info("Regime B snapshot %d/%d", idx, len(snapshots))
        key = (snap["dialogue_id"], str(snap["perspective"]), int(snap["turn_index"]))
        if key in existing:
            stats["skipped"] += 1
            continue
        t0 = time.time()
        outputs = generator.generate(snap["messages"], SELF_CONSISTENCY_KWARGS)
        elapsed = time.time() - t0
        samples = []
        for sample_idx in range(5):
            raw = outputs[sample_idx] if sample_idx < len(outputs) else ""
            samples.append({
                "sample_idx": sample_idx,
                "raw_output": raw,
                **_parse_ranking(raw),
            })
        row = {
            "regime": "B",
            "dialogue_id": snap["dialogue_id"],
            "perspective": snap["perspective"],
            "opp_role": snap["opp_role"],
            "turn_index": int(snap["turn_index"]),
            "elapsed_seconds": elapsed,
            "my_priorities": snap["my_priorities"],
            "true_ordering": snap["true_ordering"],
            "true_hypothesis_index": snap["true_hypothesis_index"],
            "samples": samples,
        }
        _write_jsonl(output_path, [row])
        existing[key] = row
        stats["dispatched"] += 1
    return stats


def _parse_rate_a(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    valid = sum(1 for r in rows if r.get("valid"))
    return {"valid": valid, "total": total, "rate": valid / total if total else float("nan")}


def _parse_rate_b(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    samples = [s for r in rows for s in (r.get("samples") or [])]
    total = len(samples)
    valid = sum(1 for s in samples if s.get("valid"))
    return {"valid": valid, "total": total, "rate": valid / total if total else float("nan")}


def summarize_regime_a(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_k: Dict[int, List[Mapping[str, Any]]] = {k: [] for k in range(1, 6)}
    for row in rows:
        k = int(row.get("k"))
        if k in by_k:
            by_k[k].append(row)

    per_k: Dict[str, Dict[str, float]] = {}
    for k in range(1, 6):
        vals_ema = []
        vals_top1 = []
        vals_ndcg = []
        for row in by_k[k]:
            pred = list(row.get("ordering") or [])
            true = list(row.get("true_ordering") or [])
            if row.get("valid") and len(pred) == 3 and len(true) == 3:
                vals_ema.append(ema(pred, true))
                vals_top1.append(top1(pred, true))
                vals_ndcg.append(ndcg_at_3(pred, true))
            else:
                vals_ema.append(0.0)
                vals_top1.append(0.0)
                vals_ndcg.append(0.0)
        per_k[str(k)] = {
            "support": len(by_k[k]),
            "ema": float(np.mean(vals_ema)) if vals_ema else float("nan"),
            "top1": float(np.mean(vals_top1)) if vals_top1 else float("nan"),
            "ndcg": float(np.mean(vals_ndcg)) if vals_ndcg else float("nan"),
        }

    def weighted(metric: str) -> float:
        vals = []
        for k, weight in KPENALTY_WEIGHTS.items():
            value = per_k[str(k)][metric]
            if np.isnan(value):
                return float("nan")
            vals.append(weight * value)
        return float(sum(vals))

    return {
        "support_total": len(rows),
        "support_target_note": (
            "Strict natural-opponent-utterance prefixes in data/casino/casino_test.json "
            "have supports 150/150/150/149/147 for mturk_agent_1."
        ),
        "parse": _parse_rate_a(rows),
        "per_k": per_k,
        "kpenalty": {
            "ema": weighted("ema"),
            "top1": weighted("top1"),
            "ndcg": weighted("ndcg"),
        },
    }


def _posterior_for_b_row(row: Mapping[str, Any]) -> np.ndarray:
    sample_indices = [
        int(s["hypothesis_index"]) if s.get("valid") and s.get("hypothesis_index") is not None else None
        for s in (row.get("samples") or [])
    ]
    return posterior_from_sample_indices(sample_indices)


def summarize_regime_b(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    brier_vals: List[float] = []
    by_turn: Dict[int, List[float]] = {}
    for row in rows:
        true_idx = row.get("true_hypothesis_index")
        if true_idx is None:
            continue
        posterior = _posterior_for_b_row(row)
        score = normalized_brier(posterior, int(true_idx))
        brier_vals.append(score)
        by_turn.setdefault(int(row.get("turn_index")), []).append(score)

    turn_curve = []
    turn_curve_n10 = []
    for turn_index in sorted(by_turn):
        vals = by_turn[turn_index]
        item = {
            "turn_index": turn_index,
            "mean": float(np.mean(vals)),
            "support": len(vals),
        }
        turn_curve.append(item)
        if len(vals) >= 10 and 0 <= turn_index <= 16:
            turn_curve_n10.append(item)

    return {
        "support": len(rows),
        "parse": _parse_rate_b(rows),
        "brier": {
            "mean": float(np.mean(brier_vals)) if brier_vals else float("nan"),
            "support": len(brier_vals),
        },
        "brier_by_turn_index": turn_curve,
        "brier_by_turn_index_n_ge_10_0_16": turn_curve_n10,
    }


def load_baseline_actions(path: Path, *, perspective: str) -> Dict[RegimeBKey, Dict[str, Any]]:
    out: Dict[RegimeBKey, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("perspective") != perspective:
                continue
            key = (rec.get("dialogue_id"), str(rec.get("perspective")), int(rec.get("turn_index")))
            pred = rec.get("pred") or {}
            counts = _coerce_self_counts_from_bid_vector(pred.get("bid"))
            action_source = "pred_bid"
            if counts is None and pred.get("accept") is True:
                counts = _accepted_offer_self_counts(rec)
                action_source = "accepted_pending_offer"
            out[key] = {
                "counts": counts,
                "action": pred.get("action"),
                "accept": pred.get("accept"),
                "source": action_source if counts is not None else None,
            }
    return out


def summarize_action_consistency(
    rows_b: Sequence[Mapping[str, Any]],
    *,
    baseline_actions: Mapping[RegimeBKey, Mapping[str, Any]],
) -> Dict[str, Any]:
    strict_vals: List[bool] = []
    loose_vals: List[bool] = []
    missing_action = 0
    invalid_map = 0
    source_counts: Dict[str, int] = {}
    for row in rows_b:
        key = (row.get("dialogue_id"), str(row.get("perspective")), int(row.get("turn_index")))
        action = baseline_actions.get(key) or {}
        counts = action.get("counts")
        if counts is None:
            missing_action += 1
            continue
        posterior = _posterior_for_b_row(row)
        map_idx = int(np.argmax(posterior))
        ordering = list(HYPOTHESES[map_idx])
        if len(ordering) != 3:
            invalid_map += 1
            continue
        source = str(action.get("source") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        strict_vals.append(
            strict_action_consistent(
                offer_self_counts=counts,
                my_priorities=row.get("my_priorities") or {},
                predicted_ordering=ordering,
                lambda_=1.0,
            )
        )
        loose_vals.append(
            loose_action_consistent(
                offer_self_counts=counts,
                predicted_ordering=ordering,
            )
        )
    strict = float(np.mean(strict_vals)) if strict_vals else float("nan")
    loose = float(np.mean(loose_vals)) if loose_vals else float("nan")
    return {
        "strict_rate": strict,
        "loose_rate": loose,
        "strict_mismatch_rate": 1.0 - strict if strict_vals else float("nan"),
        "loose_mismatch_rate": 1.0 - loose if loose_vals else float("nan"),
        "support": len(strict_vals),
        "baseline_action_missing_or_unparseable": missing_action,
        "invalid_map_ranking": invalid_map,
        "action_sources": source_counts,
        "definition_note": (
            "Uses the existing 70B structured-CoT live baseline turn_records "
            "as the action source; Regime B supplies the MAP ranking."
        ),
    }


def _annotation_lookups(
    dialogues: Sequence[Mapping[str, Any]],
    annotations_path: Optional[Path],
) -> Dict[Any, Dict[int, List[str]]]:
    if annotations_path is None or not annotations_path.exists():
        return {}
    annotations = _load_json(annotations_path)
    ann_by_id = {d.get("dialogue_id"): d.get("annotations", []) for d in annotations}
    return {
        d.get("dialogue_id"): build_annotation_lookup(
            ann_by_id.get(d.get("dialogue_id")) or [],
            d.get("chat_logs") or [],
        )
        for d in dialogues
    }


def _deceptive_tags_seen(
    *,
    dialogue: Mapping[str, Any],
    perspective: str,
    turn_index: int,
    ann_lookup: Mapping[int, Sequence[str]],
) -> List[str]:
    opp_role = _role_pair(perspective)
    seen = set()
    for idx, turn in enumerate(dialogue.get("chat_logs") or []):
        if idx > turn_index:
            break
        if turn.get("id") != opp_role or not _is_natural(turn):
            continue
        tags = {str(t).strip().lower() for t in ann_lookup.get(idx, [])}
        seen.update(tags & {"uv-part", "vouch-fair"})
    return sorted(seen)


def write_clean_records(
    *,
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
    data_path: Path,
    annotations_path: Optional[Path],
    output_path: Path,
) -> None:
    dialogues = _load_json(data_path)
    by_id = _dialogues_by_id(dialogues)
    ann = _annotation_lookups(dialogues, annotations_path)
    with output_path.open("w") as f:
        for row in rows_a:
            clean = dict(row)
            clean.pop("raw_output", None)
            f.write(json.dumps(clean, default=_json_default) + "\n")
        for row in rows_b:
            clean = dict(row)
            did = clean.get("dialogue_id")
            clean["deceptive_tags_seen"] = _deceptive_tags_seen(
                dialogue=by_id.get(did) or {},
                perspective=str(clean.get("perspective")),
                turn_index=int(clean.get("turn_index")),
                ann_lookup=ann.get(did, {}),
            )
            for sample in clean.get("samples") or []:
                sample.pop("raw_output", None)
            f.write(json.dumps(clean, default=_json_default) + "\n")


def write_headline_csv(summary: Mapping[str, Any], path: Path) -> None:
    a = summary.get("regime_a") or {}
    b = summary.get("regime_b") or {}
    ac = summary.get("action_consistency") or {}
    rows = [
        {
            "paper_location": "Table 1 70B row",
            "metric": "top1_kpenalty_percent",
            "value": 100.0 * (a.get("kpenalty") or {}).get("top1", float("nan")),
            "support": a.get("support_total"),
            "note": "weights (5,4,3,2,1)/15 over k=1..5",
        },
        {
            "paper_location": "Table 1 70B row",
            "metric": "ndcg3_kpenalty_percent",
            "value": 100.0 * (a.get("kpenalty") or {}).get("ndcg", float("nan")),
            "support": a.get("support_total"),
            "note": "weights (5,4,3,2,1)/15 over k=1..5",
        },
        {
            "paper_location": "Table 2 70B row",
            "metric": "brier_mean",
            "value": (b.get("brier") or {}).get("mean"),
            "support": (b.get("brier") or {}).get("support"),
            "note": "6-way self-consistency posterior over HYPOTHESES",
        },
        {
            "paper_location": "Table 2 sanity",
            "metric": "ema_k5_percent",
            "value": 100.0 * ((a.get("per_k") or {}).get("5") or {}).get("ema", float("nan")),
            "support": ((a.get("per_k") or {}).get("5") or {}).get("support"),
            "note": "sanity check for draft EMA",
        },
        {
            "paper_location": "Section 4.1",
            "metric": "action_consistency_strict_percent",
            "value": 100.0 * ac.get("strict_rate", float("nan")),
            "support": ac.get("support"),
            "note": "MAP ranking; U_self + 1.0 * U_opp argmax",
        },
        {
            "paper_location": "Section 4.1",
            "metric": "action_consistency_loose_percent",
            "value": 100.0 * ac.get("loose_rate", float("nan")),
            "support": ac.get("support"),
            "note": "directional opponent allocation consistency",
        },
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["paper_location", "metric", "value", "support", "note"],
        )
        writer.writeheader()
        writer.writerows(rows)


def aggregate_outputs(
    *,
    output_dir: Path,
    data_path: Path,
    annotations_path: Optional[Path],
    baseline_turn_records: Path,
    perspective: str,
) -> Dict[str, Any]:
    rows_a = list(load_jsonl_by_key(output_dir / "partial_regime_a.jsonl", _record_key_a).values())
    rows_b = list(load_jsonl_by_key(output_dir / "partial_regime_b.jsonl", _record_key_b).values())
    rows_a.sort(key=lambda r: (str(r.get("dialogue_id")), int(r.get("k"))))
    rows_b.sort(key=lambda r: (str(r.get("dialogue_id")), int(r.get("turn_index"))))
    baseline_actions = load_baseline_actions(baseline_turn_records, perspective=perspective)

    summary = {
        "protocol_lock": {
            "split": str(data_path),
            "perspective": perspective,
            "matched_turn_records_target": 1054,
            "regime_b_support": len(rows_b),
            "regime_a_natural_prefix_support": len(rows_a),
        },
        "regime_a": summarize_regime_a(rows_a),
        "regime_b": summarize_regime_b(rows_b),
        "action_consistency": summarize_action_consistency(
            rows_b,
            baseline_actions=baseline_actions,
        ),
        "hypotheses_order": [" > ".join(h) for h in HYPOTHESES],
    }
    write_clean_records(
        rows_a=rows_a,
        rows_b=rows_b,
        data_path=data_path,
        annotations_path=annotations_path,
        output_path=output_dir / "inference_records.jsonl",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n"
    )
    write_headline_csv(summary, output_dir / "headline_numbers.csv")
    LOGGER.info("wrote aggregation outputs under %s", output_dir)
    return summary


def _write_run_config(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    generator: Optional[PipelineGenerator],
) -> None:
    config = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model": args.model,
        "model_revision": args.model_revision,
        "snapshot_path": str(args.snapshot_path) if args.snapshot_path else None,
        "resolved_name_or_path": (
            generator.resolved_name_or_path if generator is not None else None
        ),
        "generation_kwargs": {
            "regime_a": DETERMINISTIC_KWARGS,
            "regime_b": SELF_CONSISTENCY_KWARGS,
        },
        "dataset_sha256": _hash_file(args.data),
        "git_sha": _git_sha(),
        "lsf_job_id": os.environ.get("LSB_JOBID"),
        "versions": {
            "transformers": _package_version("transformers"),
            "accelerate": _package_version("accelerate"),
            "torch": _package_version("torch"),
        },
        "env": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        },
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, default=_json_default) + "\n"
    )


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/casino/casino_test.json"))
    p.add_argument("--annotations", type=Path, default=Path("external/casino_original/data/casino_ann.json"))
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--model-revision", default=DEFAULT_REVISION)
    p.add_argument("--snapshot-path", type=Path, default=DEFAULT_SNAPSHOT_PATH)
    p.add_argument("--matched-turn-records", type=Path, default=DEFAULT_MATCHED_TURN_RECORDS)
    p.add_argument("--baseline-turn-records", type=Path, default=DEFAULT_BASELINE_TURN_RECORDS)
    p.add_argument("--perspective", default="mturk_agent_1")
    p.add_argument("--max-dialogues", type=int, default=150)
    p.add_argument("--smoke-dialogues", type=int, default=10)
    p.add_argument("--smoke-min-parse-rate", type=float, default=0.95)
    p.add_argument("--regime-a-expected-ema-k5", type=float, default=0.2222)
    p.add_argument("--regime-a-ema-tolerance", type=float, default=0.005)
    p.add_argument("--aggregate-only", action="store_true")
    p.add_argument("--skip-regime-b-gate", action="store_true",
                   help="Deprecated compatibility flag; Regime A EMA drift is non-blocking unless --strict-regime-a-gate is set.")
    p.add_argument("--strict-regime-a-gate", action="store_true",
                   help="Stop before Regime B if EMA@5 drifts from --regime-a-expected-ema-k5 beyond tolerance.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    _setup_logging(args.output_dir)

    if args.aggregate_only:
        aggregate_outputs(
            output_dir=args.output_dir,
            data_path=args.data,
            annotations_path=args.annotations,
            baseline_turn_records=args.baseline_turn_records,
            perspective=args.perspective,
        )
        return 0

    if args.snapshot_path and not args.snapshot_path.exists():
        LOGGER.error("snapshot path does not exist: %s", args.snapshot_path)
        return 2

    dialogues = _load_json(args.data)
    selected = dialogues[: args.max_dialogues] if args.max_dialogues else dialogues
    matched_records = load_turn_record_keys(args.matched_turn_records, perspective=args.perspective)
    matched_records = [
        r for r in matched_records
        if r.get("dialogue_id") in {d.get("dialogue_id") for d in selected}
    ]

    LOGGER.info("loading model %s revision=%s", args.model, args.model_revision)
    generator = PipelineGenerator(model=args.model, revision=args.model_revision)
    LOGGER.info("resolved model _name_or_path=%s", generator.resolved_name_or_path)
    _write_run_config(args=args, output_dir=args.output_dir, generator=generator)

    path_a = args.output_dir / "partial_regime_a.jsonl"
    path_b = args.output_dir / "partial_regime_b.jsonl"
    existing_a = load_jsonl_by_key(path_a, _record_key_a)
    existing_b = load_jsonl_by_key(path_b, _record_key_b)
    LOGGER.info("loaded existing A=%d B=%d", len(existing_a), len(existing_b))

    if args.smoke_dialogues:
        smoke = selected[: args.smoke_dialogues]
        smoke_a = list(iter_regime_a_snapshots(smoke, perspective=args.perspective))
        LOGGER.info("running Regime A smoke: %d dialogues, %d snapshots", len(smoke), len(smoke_a))
        run_regime_a(
            generator=generator,
            snapshots=smoke_a,
            output_path=path_a,
            existing=existing_a,
        )
        smoke_ids = {d.get("dialogue_id") for d in smoke}
        smoke_rows = [r for r in existing_a.values() if r.get("dialogue_id") in smoke_ids]
        smoke_rate = _parse_rate_a(smoke_rows)["rate"]
        LOGGER.info("Regime A smoke parse rate: %.3f", smoke_rate)
        if smoke_rate < args.smoke_min_parse_rate:
            LOGGER.error(
                "smoke parse rate %.3f below gate %.3f; stopping",
                smoke_rate,
                args.smoke_min_parse_rate,
            )
            return 3

    full_a = list(iter_regime_a_snapshots(selected, perspective=args.perspective))
    LOGGER.info("running Regime A full: %d snapshots", len(full_a))
    stats_a = run_regime_a(
        generator=generator,
        snapshots=full_a,
        output_path=path_a,
        existing=existing_a,
    )
    LOGGER.info("Regime A stats: %s", stats_a)
    summary_a = summarize_regime_a(list(existing_a.values()))
    ema_k5 = (summary_a.get("per_k") or {}).get("5", {}).get("ema")
    LOGGER.info("Regime A EMA@5 %.4f", ema_k5)
    ema_drifted = (
        ema_k5 is not None
        and not np.isnan(float(ema_k5))
        and abs(float(ema_k5) - args.regime_a_expected_ema_k5) > args.regime_a_ema_tolerance
    )
    if ema_drifted:
        message = (
            "Regime A EMA@5 %.4f drifted from expected %.4f by more than %.4f; "
            "continuing to Regime B because the sanity gate is warning-only"
        )
        if args.strict_regime_a_gate:
            aggregate_outputs(
                output_dir=args.output_dir,
                data_path=args.data,
                annotations_path=args.annotations,
                baseline_turn_records=args.baseline_turn_records,
                perspective=args.perspective,
            )
            LOGGER.error(
                message.replace("continuing to Regime B because the sanity gate is warning-only", "not launching Regime B"),
                ema_k5,
                args.regime_a_expected_ema_k5,
                args.regime_a_ema_tolerance,
            )
            return 4
        LOGGER.warning(
            message,
            ema_k5,
            args.regime_a_expected_ema_k5,
            args.regime_a_ema_tolerance,
        )

    full_b = list(iter_regime_b_snapshots(selected, matched_turn_records=matched_records))
    LOGGER.info("running Regime B full: %d snapshots", len(full_b))
    stats_b = run_regime_b(
        generator=generator,
        snapshots=full_b,
        output_path=path_b,
        existing=existing_b,
    )
    LOGGER.info("Regime B stats: %s", stats_b)

    summary = aggregate_outputs(
        output_dir=args.output_dir,
        data_path=args.data,
        annotations_path=args.annotations,
        baseline_turn_records=args.baseline_turn_records,
        perspective=args.perspective,
    )
    LOGGER.info("summary protocol lock: %s", summary.get("protocol_lock"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
