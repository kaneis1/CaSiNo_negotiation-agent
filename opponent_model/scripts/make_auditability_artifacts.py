#!/usr/bin/env python3
"""Create auditability demonstrations from held-out turn records.

This script is intentionally analysis-only: it reads saved Protocol-3 records,
recomputes deterministic menu recommendations from exposed posteriors, and
writes paper-ready diagnostic artifacts. It does not load or run any model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from opponent_model.turn_level_metrics import (
    coerce_bid_vector,
    cosine_similarity,
    normalized_brier,
)
from sft_8b.bayesian_agent import (
    DEFAULT_ACCEPT_FLOOR,
    DEFAULT_ACCEPT_MARGIN,
    pending_self_points as compute_pending_self_points,
    select_action,
)
from sft_8b.menu import build_menu
from sft_8b.posterior import N_ORDERINGS, ORDERINGS
from sft_8b.prompts import ITEMS


DEFAULT_STUDENT_RECORDS = "opponent_model/results/turn_eval_student_balanced_full150/turn_records.jsonl"
DEFAULT_BAYESIAN_MENU_RECORDS = "opponent_model/results/turn_eval_bayesian_lambda1.0_m5_f0.50_full150/turn_records.jsonl"
DEFAULT_CORRECT_PREFIX_RECORDS = "opponent_model/results/ablation_neurips2026/a2d_correct_prefix/turn_records.jsonl"
DEFAULT_ADVERSARIAL_PREFIX_RECORDS = "opponent_model/results/ablation_neurips2026/a2d_adversarial_prefix/turn_records.jsonl"
DEFAULT_DATA = "data/casino_test.json"
DEFAULT_OUTPUT_DIR = "opponent_model/results/day11_auditability"

DEAL_ACTIONS = {"Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away"}
CASE_TAGS = (
    "belief wrong / policy consistent",
    "belief right / policy inconsistent",
    "belief wrong / lucky action",
    "full failure",
)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summary_path_for_records(records_path: Path) -> Path:
    return records_path.with_name("turn_summary.json")


def maybe_load_summary(records_path: Path) -> Dict[str, Any]:
    path = summary_path_for_records(records_path)
    if not path.exists():
        return {}
    return load_json(path)


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def record_key(record: Mapping[str, Any]) -> Tuple[str, str, int]:
    return (
        str(record.get("dialogue_id")),
        str(record.get("perspective")),
        int(record.get("turn_index")),
    )


def index_records(records: Iterable[Mapping[str, Any]]) -> Dict[Tuple[str, str, int], Mapping[str, Any]]:
    out: Dict[Tuple[str, str, int], Mapping[str, Any]] = {}
    for record in records:
        out[record_key(record)] = record
    return out


def dialogue_lookup(dialogues: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    return {str(d.get("dialogue_id", i)): d for i, d in enumerate(dialogues)}


def ordering_label(index: Optional[int]) -> str:
    if index is None or index < 0 or index >= len(ORDERINGS):
        return "unknown"
    return " > ".join(ORDERINGS[index])


def normalize_posterior(posterior: Any) -> Optional[np.ndarray]:
    if posterior is None:
        return None
    arr = np.asarray(posterior, dtype=float).flatten()
    if arr.shape != (N_ORDERINGS,) or np.any(arr < 0):
        return None
    total = float(arr.sum())
    if total <= 0.0:
        return None
    return arr / total


def posterior_entropy_bits(posterior: Sequence[float]) -> float:
    arr = np.asarray(posterior, dtype=float)
    nz = arr[arr > 0.0]
    entropy = float(-np.sum(nz * np.log2(nz))) if nz.size else 0.0
    return 0.0 if abs(entropy) < 1e-12 else entropy


def posterior_confidence(posterior: Sequence[float]) -> float:
    arr = np.asarray(posterior, dtype=float)
    return float(np.max(arr)) if arr.size else float("nan")


def posterior_stats(posterior: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(posterior, dtype=float)
    map_index = int(np.argmax(arr))
    return {
        "map_index": map_index,
        "map_ordering": ordering_label(map_index),
        "confidence": posterior_confidence(arr),
        "entropy_bits": posterior_entropy_bits(arr),
    }


def one_hot(index: int) -> np.ndarray:
    arr = np.zeros(N_ORDERINGS, dtype=float)
    arr[int(index)] = 1.0
    return arr


def _bid_dict_from_vector(vec: Optional[np.ndarray]) -> Optional[Dict[str, int]]:
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=float).flatten()
    if arr.shape[0] < 3:
        return None
    return {item: int(round(float(arr[i]))) for i, item in enumerate(ITEMS)}


def _split_bid_vector(self_counts: Mapping[str, Any]) -> np.ndarray:
    self_arr = np.array([int(self_counts.get(item, 0)) for item in ITEMS], dtype=float)
    opp_arr = np.array([3 - int(self_counts.get(item, 0)) for item in ITEMS], dtype=float)
    return np.concatenate([self_arr, opp_arr])


def canonical_action_type(pred_or_decision: Mapping[str, Any], *, is_human: bool = False) -> str:
    if is_human:
        if pred_or_decision.get("accept") is True:
            return "accept"
        if pred_or_decision.get("accept") is False:
            return "reject"
        if pred_or_decision.get("bid") is not None:
            return "bid"
        return "utter"

    action = str(pred_or_decision.get("action") or "").lower()
    accept = pred_or_decision.get("accept")
    if accept is True or action == "accept":
        return "accept"
    if accept is False or action in {"reject", "walk-away", "walk_away"}:
        return "reject"
    if action in {"submit", "propose", "bid"} or pred_or_decision.get("bid") is not None:
        return "bid"
    return "utter"


def action_alignment(
    student_type: str,
    reference_type: str,
    *,
    student_bid: Optional[np.ndarray],
    reference_bid: Optional[np.ndarray],
    bid_close_threshold: float,
) -> Optional[bool]:
    if not student_type or not reference_type:
        return None
    if student_type != reference_type:
        return False
    if reference_type != "bid":
        return True
    if student_bid is None or reference_bid is None:
        return False
    return cosine_similarity(student_bid, reference_bid) >= float(bid_close_threshold)


def menu_decision(
    posterior: Sequence[float],
    *,
    my_priorities: Mapping[str, str],
    pending_offer: Optional[Mapping[str, Any]],
    lambda_: float,
    accept_margin: int,
    accept_floor: float,
) -> Dict[str, Any]:
    menu = build_menu(posterior, my_priorities, lambda_=lambda_, top_k=1)
    pending_points = (
        compute_pending_self_points(pending_offer, my_priorities)
        if pending_offer is not None else None
    )
    decision = select_action(
        menu,
        pending_self_points=pending_points,
        accept_margin=accept_margin,
        accept_floor=accept_floor,
    )
    action_type = canonical_action_type(decision)
    bid_vec = (
        _split_bid_vector(decision["bid"])
        if decision.get("bid") is not None else None
    )
    return {
        "decision": decision,
        "action_type": action_type,
        "bid_vector": bid_vec,
        "bid": _bid_dict_from_vector(bid_vec),
        "top_u_self": int(menu[0].u_self),
        "top_exp_u_opp": float(menu[0].exp_u_opp),
        "top_score": float(menu[0].score),
        "pending_self_points": pending_points,
    }


def compact_text(text: Any, *, limit: int = 160) -> str:
    s = " ".join(str(text or "").split())
    s = s.encode("ascii", "ignore").decode("ascii")
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 3)].rstrip() + "..."


def context_snippet(
    dialogue: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    window: int = 4,
) -> str:
    logs = list(dialogue.get("chat_logs") or [])
    turn_index = int(record.get("turn_index", 0))
    rows = []
    for idx in range(max(0, turn_index - window), min(len(logs), turn_index + 1)):
        turn = logs[idx]
        speaker = turn.get("id")
        text = compact_text(turn.get("text"), limit=120)
        rows.append(f"t{idx} {speaker}: {text}")
    return " | ".join(rows)


def annotation_events(
    dialogue: Mapping[str, Any],
    *,
    perspective: str,
    records: Sequence[Mapping[str, Any]],
    limit: int = 2,
) -> List[Dict[str, Any]]:
    if not records:
        return []
    turns = {int(r.get("turn_index")) for r in records}
    max_turn = max(turns)
    events: List[Dict[str, Any]] = []
    keywords = ("need", "want", "prefer", "important", "take", "reject", "no ")
    items = tuple(item.lower() for item in ITEMS)
    for idx, turn in enumerate(dialogue.get("chat_logs") or []):
        if idx > max_turn:
            break
        text = str(turn.get("text") or "")
        lower = text.lower()
        if turn.get("id") == perspective:
            continue
        if text in DEAL_ACTIONS:
            label = text
        elif any(item.lower() in lower for item in items) and any(k in lower for k in keywords):
            label = compact_text(text, limit=72)
        else:
            continue
        events.append({"turn_index": idx, "label": label})
        if len(events) >= limit:
            break
    return events


def case_tags_for_row(row: Mapping[str, Any]) -> List[str]:
    belief_correct = row.get("belief_correct")
    menu_alignment = row.get("menu_alignment")
    human_agreement = row.get("human_agreement")
    correct_menu_alignment = row.get("correct_menu_alignment")

    tags: List[str] = []
    if belief_correct is False and menu_alignment is True:
        tags.append("belief wrong / policy consistent")
    if belief_correct is True and menu_alignment is False:
        tags.append("belief right / policy inconsistent")
    if (
        belief_correct is False
        and (human_agreement is True or correct_menu_alignment is True)
    ):
        tags.append("belief wrong / lucky action")
    if (
        belief_correct is False
        and menu_alignment is False
        and human_agreement is False
        and correct_menu_alignment is False
    ):
        tags.append("full failure")
    if not tags and belief_correct is True and menu_alignment is True:
        tags.append("belief right / policy consistent")
    if not tags:
        tags.append("other")
    return tags


def primary_case_tag(tags: Sequence[str]) -> str:
    priority = (
        "full failure",
        "belief right / policy inconsistent",
        "belief wrong / lucky action",
        "belief wrong / policy consistent",
        "belief right / policy consistent",
        "other",
    )
    for tag in priority:
        if tag in tags:
            return tag
    return tags[0] if tags else "other"


def analyze_student_records(
    records: Sequence[Mapping[str, Any]],
    dialogues_by_id: Mapping[str, Mapping[str, Any]],
    *,
    lambda_: float,
    accept_margin: int,
    accept_floor: float,
    bid_close_threshold: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        pred = record.get("pred") or {}
        true = record.get("true") or {}
        posterior = normalize_posterior(pred.get("posterior"))
        true_idx = true.get("true_hypothesis_index")
        if posterior is None or true_idx is None:
            continue

        did = str(record.get("dialogue_id"))
        dialogue = dialogues_by_id.get(did)
        if dialogue is None:
            raise KeyError(f"dialogue_id {did!r} from records not found in data")
        perspective = str(record.get("perspective"))
        my_priorities = (
            (dialogue.get("participant_info") or {})
            .get(perspective, {})
            .get("value2issue")
        )
        if not my_priorities:
            raise KeyError(f"missing priorities for {did}/{perspective}")

        stats = posterior_stats(posterior)
        true_idx_int = int(true_idx)
        student_menu = menu_decision(
            posterior,
            my_priorities=my_priorities,
            pending_offer=record.get("pending_offer"),
            lambda_=lambda_,
            accept_margin=accept_margin,
            accept_floor=accept_floor,
        )
        correct_menu = menu_decision(
            one_hot(true_idx_int),
            my_priorities=my_priorities,
            pending_offer=record.get("pending_offer"),
            lambda_=lambda_,
            accept_margin=accept_margin,
            accept_floor=accept_floor,
        )

        student_type = canonical_action_type(pred)
        human_type = canonical_action_type(true, is_human=True)
        student_bid = coerce_bid_vector(pred.get("bid"), target_self=True)
        human_bid = coerce_bid_vector(true.get("bid"), target_self=True)

        human_agreement = action_alignment(
            student_type,
            human_type,
            student_bid=student_bid,
            reference_bid=human_bid,
            bid_close_threshold=bid_close_threshold,
        )
        menu_alignment = action_alignment(
            student_type,
            student_menu["action_type"],
            student_bid=student_bid,
            reference_bid=student_menu["bid_vector"],
            bid_close_threshold=bid_close_threshold,
        )
        correct_menu_alignment = action_alignment(
            student_type,
            correct_menu["action_type"],
            student_bid=student_bid,
            reference_bid=correct_menu["bid_vector"],
            bid_close_threshold=bid_close_threshold,
        )

        audit_supported = bool(
            human_type != "utter"
            or student_type != "utter"
            or record.get("pending_offer") is not None
        )
        belief_correct = stats["map_index"] == true_idx_int
        row: Dict[str, Any] = {
            "dialogue_id": did,
            "perspective": perspective,
            "turn_index": int(record.get("turn_index")),
            "turn_text": record.get("turn_text"),
            "audit_supported": audit_supported,
            "belief_correct": belief_correct,
            "true_index": true_idx_int,
            "true_ordering": ordering_label(true_idx_int),
            "map_index": stats["map_index"],
            "map_ordering": stats["map_ordering"],
            "confidence": stats["confidence"],
            "entropy_bits": stats["entropy_bits"],
            "brier": normalized_brier(posterior, true_idx_int),
            "student_action": pred.get("action"),
            "student_action_type": student_type,
            "student_bid": _bid_dict_from_vector(student_bid),
            "human_action_type": human_type,
            "human_bid": _bid_dict_from_vector(human_bid),
            "human_agreement": human_agreement,
            "menu_action_type": student_menu["action_type"],
            "menu_bid": student_menu["bid"],
            "menu_alignment": menu_alignment,
            "correct_menu_action_type": correct_menu["action_type"],
            "correct_menu_bid": correct_menu["bid"],
            "correct_menu_alignment": correct_menu_alignment,
            "pending_self_points": student_menu["pending_self_points"],
            "student_menu_top_u_self": student_menu["top_u_self"],
            "student_menu_top_score": student_menu["top_score"],
            "correct_menu_top_u_self": correct_menu["top_u_self"],
            "context_snippet": context_snippet(dialogue, record),
        }
        tags = case_tags_for_row(row)
        row["case_tags"] = tags
        row["primary_case_tag"] = primary_case_tag(tags)
        rows.append(row)
    return rows


def decomposition_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    audit_rows = [r for r in rows if r.get("audit_supported")]
    tag_counts = Counter()
    for row in audit_rows:
        for tag in row.get("case_tags") or ():
            tag_counts[str(tag)] += 1
    return {
        "support": len(audit_rows),
        "belief_correct": sum(1 for r in audit_rows if r.get("belief_correct") is True),
        "belief_wrong": sum(1 for r in audit_rows if r.get("belief_correct") is False),
        "human_agreement_supported": sum(1 for r in audit_rows if r.get("human_agreement") is not None),
        "human_agreement_rate": mean_bool(r.get("human_agreement") for r in audit_rows),
        "menu_alignment_rate": mean_bool(r.get("menu_alignment") for r in audit_rows),
        "correct_menu_alignment_rate": mean_bool(r.get("correct_menu_alignment") for r in audit_rows),
        "tag_counts": dict(tag_counts),
    }


def mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(sum(vals) / len(vals)) if vals else None


def mean_bool(values: Iterable[Any]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return float(sum(1 for v in vals if v) / len(vals)) if vals else None


def _sort_case_candidates(rows: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            -float(r.get("confidence") or 0.0),
            float(r.get("entropy_bits") or 0.0),
            str(r.get("dialogue_id")),
            int(r.get("turn_index") or 0),
        ),
    )


def _case_identity(row: Mapping[str, Any]) -> Tuple[str, str, int]:
    return (str(row["dialogue_id"]), str(row["perspective"]), int(row["turn_index"]))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _menu_delta(row: Mapping[str, Any]) -> bool:
    return (
        row.get("menu_alignment") != row.get("correct_menu_alignment")
        or row.get("menu_action_type") != row.get("correct_menu_action_type")
        or row.get("menu_bid") != row.get("correct_menu_bid")
    )


def _showcase_score(row: Mapping[str, Any], label: str) -> float:
    confidence = float(row.get("confidence") or 0.0)
    context = str(row.get("context_snippet") or "")
    no_recent_reject = "Reject-Deal" not in context and "Walk-Away" not in context
    action_delta = row.get("menu_action_type") != row.get("correct_menu_action_type")
    score = confidence
    if label == "right belief / wrong action (policy failure)":
        if row.get("student_action_type") == "reject" and row.get("human_action_type") == "accept":
            score += 20.0
        if row.get("menu_action_type") == "accept":
            score += 3.0
        if no_recent_reject:
            score += 2.0
        return score
    if _menu_delta(row):
        score += 10.0
    if action_delta:
        score += 5.0
    if label == "wrong belief / lucky action (planner risk)":
        # Prefer the clean latent-risk cases over generic end-of-dialogue
        # accepts: wrong posterior says reject, correct posterior says accept.
        if row.get("menu_action_type") == "reject" and row.get("correct_menu_action_type") == "accept":
            score += 8.0
        # Lower-confidence wrong beliefs make the uncertainty visible in text.
        score += max(0.0, 1.0 - confidence)
        return score
    if no_recent_reject:
        score += 2.0
    return score


def select_belief_policy_cases(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_per_tag: int = 2,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    used: set[Tuple[str, str, int]] = set()

    showcase_specs = [
        (
            "wrong belief / lucky action (planner risk)",
            max(1, max_per_tag),
            lambda r: (
                r.get("audit_supported")
                and r.get("belief_correct") is False
                and r.get("human_agreement") is True
                and r.get("menu_alignment") is False
                and r.get("correct_menu_alignment") is True
            ),
        ),
        (
            "right belief / wrong action (policy failure)",
            min(2, max(1, max_per_tag)),
            lambda r: (
                r.get("audit_supported")
                and r.get("belief_correct") is True
                and r.get("human_agreement") is False
                and r.get("menu_alignment") is False
            ),
        ),
        (
            "wrong belief / wrong action (belief changes menu)",
            1,
            lambda r: (
                r.get("audit_supported")
                and r.get("belief_correct") is False
                and r.get("human_agreement") is False
                and r.get("menu_alignment") != r.get("correct_menu_alignment")
            ),
        ),
    ]

    for label, quota, predicate in showcase_specs:
        candidates = [r for r in rows if predicate(r)]
        candidates = sorted(
            candidates,
            key=lambda r: (
                -_showcase_score(r, label),
                _safe_int(r.get("dialogue_id")),
                _safe_int(r.get("turn_index")),
            ),
        )
        count = 0
        for row in candidates:
            key = _case_identity(row)
            if key in used:
                continue
            out = dict(row)
            out["selected_for"] = label
            selected.append(out)
            used.add(key)
            count += 1
            if count >= quota:
                break

    if len(selected) >= 2:
        return selected

    # Tiny synthetic test fixtures may not contain the showcase categories.
    # Fall back to the broad diagnostic tags so helper behavior remains stable.
    for tag in CASE_TAGS:
        candidates = [
            r for r in rows
            if r.get("audit_supported") and tag in (r.get("case_tags") or ())
        ]
        for row in _sort_case_candidates(candidates):
            key = _case_identity(row)
            if key in used:
                continue
            out = dict(row)
            out["selected_for"] = tag
            selected.append(out)
            used.add(key)
            if sum(1 for s in selected if s["selected_for"] == tag) >= max_per_tag:
                break
    return selected


def _trajectory_groups(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], List[Mapping[str, Any]]]:
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dialogue_id"]), str(row["perspective"]))].append(row)
    for key in list(groups):
        groups[key] = sorted(groups[key], key=lambda r: int(r["turn_index"]))
    return groups


def _first_correct_position(group: Sequence[Mapping[str, Any]]) -> Optional[int]:
    for i, row in enumerate(group):
        if row.get("belief_correct") is True:
            return i
    return None


def select_trajectory_cases(
    rows: Sequence[Mapping[str, Any]],
    dialogues_by_id: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    groups = _trajectory_groups(rows)
    selected: List[Dict[str, Any]] = []
    used: set[Tuple[str, str]] = set()

    def add_case(label: str, candidates: Iterable[Tuple[Tuple[str, str], List[Mapping[str, Any]], float]]) -> None:
        if any(c["label"] == label for c in selected):
            return
        for key, group, score in sorted(candidates, key=lambda x: (-x[2], x[0])):
            if key in used:
                continue
            did, perspective = key
            dialogue = dialogues_by_id.get(did, {})
            selected.append({
                "label": label,
                "dialogue_id": did,
                "perspective": perspective,
                "records": group,
                "events": annotation_events(dialogue, perspective=perspective, records=group),
            })
            used.add(key)
            break

    fast = []
    slow = []
    revise = []
    fallback = []
    for key, group in groups.items():
        if len(group) < 3:
            continue
        correct = [bool(r.get("belief_correct")) for r in group]
        first = _first_correct_position(group)
        final_correct = correct[-1]
        confidence = mean(r.get("confidence") for r in group) or 0.0
        turn_indices = [int(r.get("turn_index")) for r in group]
        maps = [int(r.get("map_index")) for r in group]
        if (
            first is not None
            and correct[0] is False
            and turn_indices[first] <= 4
            and all(correct[first:])
            and len(set(maps[: first + 1])) > 1
        ):
            initial_conf = float(group[0].get("confidence") or 0.0)
            first_conf = float(group[first].get("confidence") or 0.0)
            flips = sum(1 for i in range(1, len(maps)) if maps[i] != maps[i - 1])
            certainty_penalty = 0.7 if first_conf >= 0.999 else 0.0
            score = flips * 2.0 + first_conf + initial_conf - certainty_penalty
            fast.append((key, group, score))
        if first is not None and first >= 3 and final_correct:
            slow.append((key, group, confidence + first / 10.0))
        if correct[0] is False and final_correct and len(set(maps)) > 1:
            wrong_conf = max(float(r.get("confidence") or 0.0) for r in group if r.get("belief_correct") is False)
            revise.append((key, group, wrong_conf + len(set(maps)) / 10.0))
        fallback.append((key, group, confidence))

    add_case("fast correct", fast)
    add_case("slow correct", slow)
    add_case("wrong then correct", revise)
    for label in ("fast correct", "slow correct", "wrong then correct"):
        add_case(label, fallback)
    return selected[:3]


def action_or_bid_changed(
    base: Mapping[str, Any],
    other: Mapping[str, Any],
) -> bool:
    base_pred = base.get("pred") or {}
    other_pred = other.get("pred") or {}
    if canonical_action_type(base_pred) != canonical_action_type(other_pred):
        return True
    base_bid = coerce_bid_vector(base_pred.get("bid"), target_self=True)
    other_bid = coerce_bid_vector(other_pred.get("bid"), target_self=True)
    if base_bid is None and other_bid is None:
        return False
    if (base_bid is None) != (other_bid is None):
        return True
    return not bool(np.array_equal(base_bid, other_bid))


def prefix_change_rows(
    baseline_records: Sequence[Mapping[str, Any]],
    prefix_records: Sequence[Mapping[str, Any]],
    *,
    prefix_label: str,
    analysis_by_key: Mapping[Tuple[str, str, int], Mapping[str, Any]],
    bid_close_threshold: float,
) -> List[Dict[str, Any]]:
    baseline = index_records(baseline_records)
    prefix = index_records(prefix_records)
    rows: List[Dict[str, Any]] = []
    for key in sorted(set(baseline) & set(prefix)):
        base_rec = baseline[key]
        pref_rec = prefix[key]
        if not action_or_bid_changed(base_rec, pref_rec):
            continue
        base_pred = base_rec.get("pred") or {}
        pref_pred = pref_rec.get("pred") or {}
        analysis = analysis_by_key.get(key, {})
        pref_type = canonical_action_type(pref_pred)
        human_type = canonical_action_type(pref_rec.get("true") or {}, is_human=True)
        pref_bid = coerce_bid_vector(pref_pred.get("bid"), target_self=True)
        human_bid = coerce_bid_vector((pref_rec.get("true") or {}).get("bid"), target_self=True)
        pref_human_agreement = action_alignment(
            pref_type,
            human_type,
            student_bid=pref_bid,
            reference_bid=human_bid,
            bid_close_threshold=bid_close_threshold,
        )
        base_human_agreement = analysis.get("human_agreement")
        improvement = None
        if base_human_agreement is not None and pref_human_agreement is not None:
            improvement = (
                "improved" if pref_human_agreement and not base_human_agreement
                else "worsened" if base_human_agreement and not pref_human_agreement
                else "unchanged"
            )
        rows.append({
            "prefix_label": prefix_label,
            "dialogue_id": key[0],
            "perspective": key[1],
            "turn_index": key[2],
            "baseline_action": base_pred.get("action"),
            "prefix_action": pref_pred.get("action"),
            "baseline_action_type": canonical_action_type(base_pred),
            "prefix_action_type": pref_type,
            "baseline_bid": _bid_dict_from_vector(coerce_bid_vector(base_pred.get("bid"), target_self=True)),
            "prefix_bid": _bid_dict_from_vector(pref_bid),
            "human_action_type": human_type,
            "baseline_human_agreement": base_human_agreement,
            "prefix_human_agreement": pref_human_agreement,
            "agreement_change": improvement,
            "belief_correct_baseline": analysis.get("belief_correct"),
            "baseline_map_ordering": analysis.get("map_ordering"),
            "true_ordering": analysis.get("true_ordering"),
            "turn_text": compact_text(base_rec.get("turn_text"), limit=160),
            "context_snippet": analysis.get("context_snippet"),
        })
    return rows


def confidence_bins(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    support = [
        r for r in rows
        if r.get("audit_supported")
        and r.get("human_agreement") is not None
        and r.get("confidence") is not None
    ]
    if not support:
        return []
    ranked = sorted(enumerate(support), key=lambda pair: (float(pair[1]["confidence"]), pair[0]))
    bin_by_id: Dict[int, str] = {}
    n = len(ranked)
    for rank, (_, row) in enumerate(ranked):
        if rank < n / 3.0:
            bin_by_id[id(row)] = "low"
        elif rank < 2.0 * n / 3.0:
            bin_by_id[id(row)] = "medium"
        else:
            bin_by_id[id(row)] = "high"

    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in support:
        status = "MAP correct" if row.get("belief_correct") else "MAP wrong"
        grouped[(status, bin_by_id[id(row)])].append(row)

    out: List[Dict[str, Any]] = []
    for status in ("MAP correct", "MAP wrong"):
        for name in ("low", "medium", "high"):
            bucket = grouped.get((status, name), [])
            out.append({
                "map_status": status,
                "confidence_bin": name,
                "n": len(bucket),
                "mean_confidence": mean(r.get("confidence") for r in bucket),
                "mean_entropy_bits": mean(r.get("entropy_bits") for r in bucket),
                "human_agreement_rate": mean_bool(r.get("human_agreement") for r in bucket),
                "menu_alignment_rate": mean_bool(r.get("menu_alignment") for r in bucket),
                "confidence_min": min((float(r["confidence"]) for r in bucket), default=None),
                "confidence_max": max((float(r["confidence"]) for r in bucket), default=None),
            })
    return out


def _jsonish(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _jsonish(row.get(name)) for name in fieldnames})


def write_belief_policy_cases_md(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Belief/Policy Audit Cases",
        "",
        "These examples separate the exposed posterior from the emitted action.",
        "Human agreement is a behavioral reference, while menu alignment is the deterministic Bayesian menu recommendation under the student's own posterior.",
        "The selector prioritizes turns where the student-posterior menu and the correct-posterior menu disagree, because those are the turns where the belief audit is actionable.",
        "",
    ]
    for row in rows:
        lines.extend([
            f"## {row.get('selected_for')}: dialogue {row['dialogue_id']} turn {row['turn_index']}",
            "",
            f"- True ordering: {row['true_ordering']}",
            f"- Student MAP: {row['map_ordering']} (confidence {float(row['confidence']):.3f}, entropy {float(row['entropy_bits']):.3f} bits)",
            f"- Student action: {row['student_action_type']} {row.get('student_bid') or ''}",
            f"- Human reference: {row['human_action_type']} {row.get('human_bid') or ''}; agreement={row.get('human_agreement')}",
            f"- Student-posterior menu: {row['menu_action_type']} {row.get('menu_bid') or ''}; aligned={row.get('menu_alignment')}",
            f"- Correct-posterior menu: {row['correct_menu_action_type']} {row.get('correct_menu_bid') or ''}; aligned={row.get('correct_menu_alignment')}",
            f"- Context: {row.get('context_snippet')}",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_section_draft(
    path: Path,
    *,
    summary: Mapping[str, Any],
    selected_cases: Sequence[Mapping[str, Any]],
    prefix_rows: Sequence[Mapping[str, Any]],
) -> None:
    decomp = summary.get("belief_policy_decomposition") or {}
    tag_counts = decomp.get("tag_counts") or {}
    bins = summary.get("overconfidence_bins") or []
    correct_rate = summary.get("correct_prefix_action_change_rate")
    adversarial_rate = summary.get("adversarial_prefix_action_change_rate")
    map_acc = summary.get("map_accuracy")
    mean_entropy = summary.get("mean_entropy")
    mean_conf = summary.get("mean_confidence")
    brier = summary.get("mean_brier")

    def _change_counts(prefix_label: str) -> Counter:
        return Counter(
            r.get("agreement_change")
            for r in prefix_rows
            if r.get("prefix_label") == prefix_label
        )

    correct_counts = _change_counts("correct_prefix")
    adversarial_counts = _change_counts("adversarial_prefix")

    lines = [
        "# Auditability Demonstrations",
        "",
        "The posterior is useful not only as an aggregate calibration metric, but as an audit interface. We therefore inspect individual held-out turns from the distilled student and compare three objects: the student's exposed belief, the student's emitted action, and the deterministic Bayesian menu recommendation induced by that belief.",
        "",
        f"Across {summary.get('n_student_turns')} student turns, MAP accuracy is {map_acc:.3f}, mean confidence is {mean_conf:.3f}, mean entropy is {mean_entropy:.3f} bits, and mean class-normalized Brier is {brier:.3f}. The audit-supported subset contains {decomp.get('support')} formal or structured decision turns.",
        "",
        "## Belief versus policy errors",
        "",
        "We use human agreement as a behavioral reference and menu alignment as a planner reference. This avoids treating every human decision as uniquely optimal while still making policy errors inspectable.",
        "",
        f"- Belief wrong / policy consistent: {tag_counts.get('belief wrong / policy consistent', 0)} turns.",
        f"- Belief right / policy inconsistent: {tag_counts.get('belief right / policy inconsistent', 0)} turns.",
        f"- Belief wrong / lucky action: {tag_counts.get('belief wrong / lucky action', 0)} turns.",
        f"- Full failure: {tag_counts.get('full failure', 0)} turns.",
        "",
        "Representative cases are listed in `belief_policy_cases.md`; the selected cases prioritize turns where the student-posterior menu and the correct-posterior menu disagree. Figure `posterior_trajectories.png` shows three posterior trajectories where the belief converges quickly, converges slowly, or corrects after initially favoring the wrong ordering.",
        "",
        "## Posterior correction reveals weak coupling",
        "",
        f"Correct-prefix posterior intervention changes the student's action or bid on only {correct_rate:.3%} of turns. Adversarial-prefix intervention changes action or bid on {adversarial_rate:.3%} of turns. This is a limitation rather than a controllability result: the student exposes accurate beliefs, but the exposed belief text is only weakly causally coupled to control.",
        "",
        f"In joined changed cases, correct-prefix behavioral agreement improved in {correct_counts.get('improved', 0)}, worsened in {correct_counts.get('worsened', 0)}, and was unchanged in {correct_counts.get('unchanged', 0)}. Adversarial-prefix agreement improved in {adversarial_counts.get('improved', 0)}, worsened in {adversarial_counts.get('worsened', 0)}, and was unchanged in {adversarial_counts.get('unchanged', 0)}. We therefore do not claim that posterior correction reliably improves decisions; instead, the audit identifies a concrete distillation failure mode.",
        "",
        "## Confidence as an audit signal",
        "",
        "High-confidence wrong beliefs are especially audit-critical because they combine low apparent uncertainty with a wrong MAP ordering. The confidence-bin diagnostic reports human-agreement rates separately for MAP-correct and MAP-wrong turns.",
        "",
    ]
    for row in bins:
        if row.get("n", 0) == 0:
            continue
        rate = row.get("human_agreement_rate")
        rate_s = "n/a" if rate is None else f"{rate:.3f}"
        lines.append(
            f"- {row['map_status']}, {row['confidence_bin']} confidence: n={row['n']}, human-agreement rate={rate_s}."
        )
    lines.extend([
        "",
        "Suggested paper framing: the posterior makes failures diagnosable. It reveals when the model is wrong because of belief, when it is wrong because of policy, and when distillation has failed to couple belief to action.",
        "",
    ])
    if selected_cases:
        lines.append("Selected case anchors:")
        for row in selected_cases[:6]:
            lines.append(
                f"- {row.get('selected_for')}: dialogue {row['dialogue_id']} turn {row['turn_index']} ({row['map_ordering']} vs. true {row['true_ordering']})."
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_posterior_trajectories(
    path: Path,
    trajectory_cases: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    if not trajectory_cases:
        plt.figure(figsize=(7.0, 3.0))
        plt.text(0.5, 0.5, "No trajectory cases selected", ha="center", va="center")
        plt.axis("off")
        plt.savefig(path, dpi=220)
        plt.close()
        return

    fig, axes = plt.subplots(len(trajectory_cases), 1, figsize=(8.4, 2.8 * len(trajectory_cases)), sharex=False)
    if len(trajectory_cases) == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, N_ORDERINGS))
    for ax, case in zip(axes, trajectory_cases):
        group = list(case["records"])
        xs = [int(r["turn_index"]) for r in group]
        true_idx = int(group[0]["true_index"])
        posteriors = []
        for r in group:
            # The analysis row stores enough scalar diagnostics for case
            # selection; recover full posterior from an optional private field.
            posteriors.append(np.asarray(r["_posterior"], dtype=float))
        arr = np.vstack(posteriors)
        for i, ordering in enumerate(ORDERINGS):
            lw = 2.4 if i == true_idx else 1.2
            alpha = 1.0 if i == true_idx else 0.65
            ax.plot(xs, arr[:, i], marker="o", linewidth=lw, alpha=alpha, color=colors[i], label=ordering_label(i))
        for event in case.get("events") or []:
            ax.axvline(int(event["turn_index"]), color="0.35", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.text(
                int(event["turn_index"]),
                1.02,
                compact_text(event["label"], limit=46),
                rotation=20,
                ha="left",
                va="bottom",
                fontsize=7,
            )
        ax.set_ylim(-0.02, 1.12)
        ax.set_ylabel("posterior")
        ax.grid(alpha=0.2)
        ax.set_title(
            f"{case['label']}: dialogue {case['dialogue_id']} {case['perspective']} (true {ordering_label(true_idx)})",
            loc="left",
            fontsize=10,
        )
    axes[-1].set_xlabel("Dialogue turn index")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_confidence_diagnostic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    support = [
        r for r in rows
        if r.get("audit_supported")
        and r.get("human_agreement") is not None
        and r.get("confidence") is not None
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if not support:
        ax.text(0.5, 0.5, "No confidence diagnostic support", ha="center", va="center")
        ax.axis("off")
    else:
        for status, color in ((True, "#1f77b4"), (False, "#d62728")):
            bucket = [r for r in support if r.get("belief_correct") is status]
            xs = [float(r["confidence"]) for r in bucket]
            ys = [1.0 if r.get("human_agreement") else 0.0 for r in bucket]
            jitter = [((int(r["turn_index"]) % 7) - 3) * 0.018 for r in bucket]
            ax.scatter(
                xs,
                [y + j for y, j in zip(ys, jitter)],
                s=28,
                alpha=0.65,
                color=color,
                label="MAP correct" if status else "MAP wrong",
            )
        ax.set_xlabel("Posterior confidence (max probability)")
        ax.set_ylabel("Human agreement")
        ax.set_yticks([0, 1], ["no", "yes"])
        ax.set_ylim(-0.18, 1.18)
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _with_private_posteriors(
    rows: List[Dict[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    posterior_by_key = {
        record_key(record): normalize_posterior((record.get("pred") or {}).get("posterior"))
        for record in records
    }
    for row in rows:
        key = (str(row["dialogue_id"]), str(row["perspective"]), int(row["turn_index"]))
        posterior = posterior_by_key.get(key)
        row["_posterior"] = posterior.tolist() if posterior is not None else [1.0 / N_ORDERINGS] * N_ORDERINGS
    return rows


def public_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in row.items() if not str(k).startswith("_")}


def build_metrics_summary(
    *,
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    correct_summary: Mapping[str, Any],
    adversarial_summary: Mapping[str, Any],
    overconfidence_bins: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    posterior_rows = [r for r in rows if r.get("confidence") is not None]
    correct_agent = correct_summary.get("agent_summary") or {}
    adversarial_agent = adversarial_summary.get("agent_summary") or {}
    return {
        "student_records": str(args.student_records),
        "bayesian_menu_records": str(args.bayesian_menu_records),
        "teacher_records": str(args.bayesian_menu_records),
        "teacher_records_note": "Compatibility alias; this path is the Bayesian menu agent run.",
        "correct_prefix_records": str(args.correct_prefix_records),
        "adversarial_prefix_records": str(args.adversarial_prefix_records),
        "data": str(args.data),
        "n_student_turns": len(rows),
        "map_accuracy": mean_bool(r.get("belief_correct") for r in posterior_rows),
        "mean_entropy": mean(r.get("entropy_bits") for r in posterior_rows),
        "mean_confidence": mean(r.get("confidence") for r in posterior_rows),
        "mean_brier": mean(r.get("brier") for r in posterior_rows),
        "belief_policy_decomposition": decomposition_counts(rows),
        "correct_prefix_action_change_rate": correct_agent.get("prefix_action_or_bid_changed_rate"),
        "correct_prefix_action_or_bid_changed": correct_agent.get("prefix_action_or_bid_changed"),
        "adversarial_prefix_action_change_rate": adversarial_agent.get("prefix_action_or_bid_changed_rate"),
        "adversarial_prefix_action_or_bid_changed": adversarial_agent.get("prefix_action_or_bid_changed"),
        "overconfidence_bins": list(overconfidence_bins),
        "lambda": args.lambda_,
        "accept_margin": args.accept_margin,
        "accept_floor": args.accept_floor,
        "bid_close_threshold": args.bid_close_threshold,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-records", type=Path, default=Path(DEFAULT_STUDENT_RECORDS))
    parser.add_argument("--bayesian-menu-records", type=Path, default=Path(DEFAULT_BAYESIAN_MENU_RECORDS))
    parser.add_argument("--correct-prefix-records", type=Path, default=Path(DEFAULT_CORRECT_PREFIX_RECORDS))
    parser.add_argument("--adversarial-prefix-records", type=Path, default=Path(DEFAULT_ADVERSARIAL_PREFIX_RECORDS))
    parser.add_argument("--data", type=Path, default=Path(DEFAULT_DATA))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    parser.add_argument("--accept-margin", type=int, default=DEFAULT_ACCEPT_MARGIN)
    parser.add_argument("--accept-floor", type=float, default=DEFAULT_ACCEPT_FLOOR)
    parser.add_argument("--bid-close-threshold", type=float, default=0.90)
    parser.add_argument("--max-cases-per-tag", type=int, default=3)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    student_records = load_jsonl(args.student_records)
    correct_prefix_records = load_jsonl(args.correct_prefix_records)
    adversarial_prefix_records = load_jsonl(args.adversarial_prefix_records)
    dialogues = load_json(args.data)
    dialogues_by_id = dialogue_lookup(dialogues)
    correct_summary = maybe_load_summary(args.correct_prefix_records)
    adversarial_summary = maybe_load_summary(args.adversarial_prefix_records)

    analysis_rows = analyze_student_records(
        student_records,
        dialogues_by_id,
        lambda_=args.lambda_,
        accept_margin=args.accept_margin,
        accept_floor=args.accept_floor,
        bid_close_threshold=args.bid_close_threshold,
    )
    analysis_rows = _with_private_posteriors(analysis_rows, student_records)
    analysis_by_key = {
        (str(r["dialogue_id"]), str(r["perspective"]), int(r["turn_index"])): r
        for r in analysis_rows
    }

    selected_cases = select_belief_policy_cases(
        analysis_rows,
        max_per_tag=args.max_cases_per_tag,
    )
    trajectory_cases = select_trajectory_cases(analysis_rows, dialogues_by_id)
    correct_changes = prefix_change_rows(
        student_records,
        correct_prefix_records,
        prefix_label="correct_prefix",
        analysis_by_key=analysis_by_key,
        bid_close_threshold=args.bid_close_threshold,
    )
    adversarial_changes = prefix_change_rows(
        student_records,
        adversarial_prefix_records,
        prefix_label="adversarial_prefix",
        analysis_by_key=analysis_by_key,
        bid_close_threshold=args.bid_close_threshold,
    )
    prefix_rows = correct_changes + adversarial_changes
    bins = confidence_bins(analysis_rows)
    summary = build_metrics_summary(
        args=args,
        rows=analysis_rows,
        correct_summary=correct_summary,
        adversarial_summary=adversarial_summary,
        overconfidence_bins=bins,
    )
    summary["joined_prefix_action_or_bid_changed"] = {
        "correct_prefix": len(correct_changes),
        "adversarial_prefix": len(adversarial_changes),
        "note": "Joined baseline-vs-prefix records used for case examples; headline rates come from the prefix harness summaries.",
    }
    summary["posterior_correction_case_agreement_changes"] = {
        label: dict(Counter(
            row.get("agreement_change")
            for row in prefix_rows
            if row.get("prefix_label") == label
        ))
        for label in ("correct_prefix", "adversarial_prefix")
    }

    case_fields = [
        "selected_for", "dialogue_id", "perspective", "turn_index",
        "belief_correct", "true_ordering", "map_ordering", "confidence",
        "entropy_bits", "student_action_type", "student_bid",
        "human_action_type", "human_bid", "human_agreement",
        "menu_action_type", "menu_bid", "menu_alignment",
        "correct_menu_action_type", "correct_menu_bid",
        "correct_menu_alignment", "case_tags", "context_snippet",
    ]
    correction_fields = [
        "prefix_label", "dialogue_id", "perspective", "turn_index",
        "baseline_action", "prefix_action", "baseline_action_type",
        "prefix_action_type", "baseline_bid", "prefix_bid",
        "human_action_type", "baseline_human_agreement",
        "prefix_human_agreement", "agreement_change",
        "belief_correct_baseline", "baseline_map_ordering",
        "true_ordering", "turn_text", "context_snippet",
    ]
    diagnostic_fields = [
        "dialogue_id", "perspective", "turn_index", "audit_supported",
        "belief_correct", "true_ordering", "map_ordering", "confidence",
        "entropy_bits", "brier", "student_action_type",
        "human_action_type", "human_agreement", "menu_action_type",
        "menu_alignment", "correct_menu_action_type",
        "correct_menu_alignment", "primary_case_tag", "context_snippet",
    ]

    outputs = {
        "auditability_section_draft.md": args.output_dir / "auditability_section_draft.md",
        "belief_policy_cases.md": args.output_dir / "belief_policy_cases.md",
        "belief_policy_cases.csv": args.output_dir / "belief_policy_cases.csv",
        "posterior_trajectories.png": args.output_dir / "posterior_trajectories.png",
        "posterior_correction_cases.csv": args.output_dir / "posterior_correction_cases.csv",
        "entropy_confidence_diagnostic.csv": args.output_dir / "entropy_confidence_diagnostic.csv",
        "entropy_confidence_plot.png": args.output_dir / "entropy_confidence_plot.png",
        "auditability_metrics_summary.json": args.output_dir / "auditability_metrics_summary.json",
        "artifact_manifest.json": args.output_dir / "artifact_manifest.json",
    }

    public_cases = [public_row(r) for r in selected_cases]
    public_analysis = [public_row(r) for r in analysis_rows]
    write_csv(outputs["belief_policy_cases.csv"], public_cases, case_fields)
    write_belief_policy_cases_md(outputs["belief_policy_cases.md"], public_cases)
    write_csv(outputs["posterior_correction_cases.csv"], prefix_rows, correction_fields)
    write_csv(outputs["entropy_confidence_diagnostic.csv"], public_analysis, diagnostic_fields)
    write_json(outputs["auditability_metrics_summary.json"], summary)
    write_section_draft(
        outputs["auditability_section_draft.md"],
        summary=summary,
        selected_cases=public_cases,
        prefix_rows=prefix_rows,
    )
    plot_posterior_trajectories(outputs["posterior_trajectories.png"], trajectory_cases)
    plot_confidence_diagnostic(outputs["entropy_confidence_plot.png"], analysis_rows)

    manifest = {
        "description": "Auditability demonstrations for posterior diagnostics in CaSiNo negotiation.",
        "outputs": {name: str(path) for name, path in outputs.items()},
        "source_paths": {
            "student_records": str(args.student_records),
            "bayesian_menu_records": str(args.bayesian_menu_records),
            "correct_prefix_records": str(args.correct_prefix_records),
            "adversarial_prefix_records": str(args.adversarial_prefix_records),
            "data": str(args.data),
        },
        "n_selected_belief_policy_cases": len(public_cases),
        "n_prefix_changed_cases": len(prefix_rows),
        "trajectory_cases": [
            {
                "label": c["label"],
                "dialogue_id": c["dialogue_id"],
                "perspective": c["perspective"],
            }
            for c in trajectory_cases
        ],
    }
    write_json(outputs["artifact_manifest.json"], manifest)

    print(f"Wrote auditability artifacts to {args.output_dir}")
    print(f"Student turns: {summary['n_student_turns']}")
    print(f"MAP accuracy: {summary['map_accuracy']:.3f}")
    print(f"Correct-prefix action/bid change rate: {summary['correct_prefix_action_change_rate']:.3%}")
    print(f"Adversarial-prefix action/bid change rate: {summary['adversarial_prefix_action_change_rate']:.3%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
