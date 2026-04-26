#!/usr/bin/env python3
"""Build Day 10 SVO subgroup tables for matched/mismatched lambda runs.

The matched run is the normal ``--lambda-from-svo`` condition. The mismatched
run is the swapped-lambda counterfactual, where proself participants receive
the prosocial lambda and prosocial participants receive the proself lambda.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Optional

from opponent_model.hypotheses import ITEMS
from sft_8b.menu import points


ROLES = ("mturk_agent_1", "mturk_agent_2")
ROW_ORDER = ("proself", "prosocial", "unclassified", "classified_all", "overall")


def _load_jsonl(path: Optional[Path]) -> list[dict[str, Any]]:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_dialogues(path: Path) -> dict[str, Mapping[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {str(d.get("dialogue_id", i)): d for i, d in enumerate(data)}


def _svo(dialogue: Mapping[str, Any], role: str) -> str:
    return str(
        ((dialogue.get("participant_info") or {})
         .get(role, {})
         .get("personality", {})
         .get("svo", "unclassified"))
    ).strip().lower()


def _priorities(dialogue: Mapping[str, Any], role: str) -> Mapping[str, str]:
    return (dialogue.get("participant_info") or {})[role]["value2issue"]


def _outcome_points(dialogue: Mapping[str, Any], role: str) -> float:
    return float(
        ((dialogue.get("participant_info") or {})
         .get(role, {})
         .get("outcomes", {})
         .get("points_scored"))
    )


def _row_keys(svo: str) -> tuple[str, ...]:
    keys = [svo, "overall"]
    if svo in {"proself", "prosocial"}:
        keys.append("classified_all")
    return tuple(keys)


def _counts_from_bid(bid: Any, offset: int) -> Optional[dict[str, int]]:
    if not isinstance(bid, (list, tuple)) or len(bid) != 6:
        return None
    try:
        return {item: int(float(bid[offset + i])) for i, item in enumerate(ITEMS)}
    except (TypeError, ValueError):
        return None


def _counts_from_task_data(turn: Mapping[str, Any]) -> Optional[tuple[dict[str, int], dict[str, int]]]:
    td = turn.get("task_data") or {}
    you = td.get("issue2youget")
    they = td.get("issue2theyget")
    if not isinstance(you, Mapping) or not isinstance(they, Mapping):
        return None
    try:
        self_counts = {item: int(float(you.get(item, 0))) for item in ITEMS}
        opp_counts = {item: int(float(they.get(item, 0))) for item in ITEMS}
    except (TypeError, ValueError):
        return None
    return self_counts, opp_counts


def _max_joint_points(dialogue: Mapping[str, Any]) -> int:
    best = -1
    mt1_priorities = _priorities(dialogue, "mturk_agent_1")
    mt2_priorities = _priorities(dialogue, "mturk_agent_2")
    for split in product(range(4), repeat=3):
        mt1_counts = {item: int(split[i]) for i, item in enumerate(ITEMS)}
        mt2_counts = {item: 3 - mt1_counts[item] for item in ITEMS}
        joint = points(mt1_counts, mt1_priorities) + points(mt2_counts, mt2_priorities)
        best = max(best, int(joint))
    return best


def _is_integrative(joint_points: float, max_joint_points: float) -> bool:
    return float(joint_points) >= float(max_joint_points) - 1e-9


def _summ(vals: Iterable[float]) -> dict[str, Any]:
    arr = [float(v) for v in vals]
    return {
        "n": len(arr),
        "mean": float(mean(arr)) if arr else float("nan"),
        "std": float(stdev(arr)) if len(arr) > 1 else float("nan"),
    }


def _binary_f1(pairs: Iterable[tuple[bool, bool]]) -> dict[str, Any]:
    pairs = list(pairs)
    if not pairs:
        return {"f1": float("nan"), "support": 0, "accuracy": float("nan")}
    tp = fp = fn = tn = 0
    for pred, gold in pairs:
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif not pred and gold:
            fn += 1
        else:
            tn += 1
    if tp == 0:
        f1 = 0.0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "f1": float(f1),
        "support": len(pairs),
        "accuracy": float((tp + tn) / len(pairs)),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def _welch(a: list[float], b: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "mean_diff_a_minus_b": (
            float(mean(a) - mean(b)) if a and b else float("nan")
        ),
        "scipy_available": False,
    }
    try:
        from scipy import stats  # type: ignore
    except Exception as exc:  # pragma: no cover - env-dependent
        out["scipy_error"] = repr(exc)
        return out
    out["scipy_available"] = True
    if len(a) >= 2 and len(b) >= 2:
        res = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        out["welch_t"] = float(res.statistic) if math.isfinite(float(res.statistic)) else None
        out["welch_p"] = float(res.pvalue) if math.isfinite(float(res.pvalue)) else None
    return out


def _empty_cell() -> dict[str, list[Any]]:
    return {
        "human_self_points": [],
        "human_joint_points": [],
        "human_first_offer_integrative": [],
        "agent_self_points": [],
        "agent_joint_points": [],
        "agent_first_offer_integrative": [],
        "match_accept_pairs": [],
        "match_accept_correct": [],
        "mismatch_accept_pairs": [],
        "mismatch_accept_correct": [],
    }


def _collect_human(
    cells: dict[str, dict[str, list[Any]]],
    dialogues: Mapping[str, Mapping[str, Any]],
    *,
    perspective: str,
) -> list[dict[str, Any]]:
    debug: list[dict[str, Any]] = []
    for did, dialogue in dialogues.items():
        svo = _svo(dialogue, perspective)
        joint = sum(_outcome_points(dialogue, role) for role in ROLES)
        max_joint = _max_joint_points(dialogue)
        for key in _row_keys(svo):
            cells[key]["human_self_points"].append(_outcome_points(dialogue, perspective))
            cells[key]["human_joint_points"].append(joint)

        first_submit = None
        for turn_index, turn in enumerate(dialogue.get("chat_logs") or []):
            if turn.get("id") == perspective and turn.get("text") == "Submit-Deal":
                first_submit = (turn_index, turn)
                break
        if first_submit is None:
            continue
        turn_index, turn = first_submit
        counts = _counts_from_task_data(turn)
        if counts is None:
            continue
        self_counts, opp_counts = counts
        opp_role = "mturk_agent_2" if perspective == "mturk_agent_1" else "mturk_agent_1"
        self_pts = float(points(self_counts, _priorities(dialogue, perspective)))
        opp_pts = float(points(opp_counts, _priorities(dialogue, opp_role)))
        joint_pts = self_pts + opp_pts
        integrative = 1.0 if _is_integrative(joint_pts, max_joint) else 0.0
        for key in _row_keys(svo):
            cells[key]["human_first_offer_integrative"].append(integrative)
        debug.append({
            "source": "human",
            "dialogue_id": did,
            "turn_index": turn_index,
            "svo": svo,
            "self_points": self_pts,
            "joint_points": joint_pts,
            "max_joint_points": max_joint,
            "integrative": bool(integrative),
        })
    return debug


def _collect_agent_bids(
    cells: dict[str, dict[str, list[Any]]],
    records: list[Mapping[str, Any]],
    dialogues: Mapping[str, Mapping[str, Any]],
    *,
    perspective: str,
) -> list[dict[str, Any]]:
    first_seen: set[str] = set()
    debug: list[dict[str, Any]] = []
    for row in records:
        if row.get("perspective") != perspective:
            continue
        dialogue = dialogues.get(str(row.get("dialogue_id")))
        if dialogue is None:
            continue
        pred = row.get("pred") or {}
        bid = pred.get("bid")
        if bid is None:
            continue
        self_counts = _counts_from_bid(bid, 0)
        opp_counts = _counts_from_bid(bid, 3)
        if self_counts is None or opp_counts is None:
            continue
        svo = _svo(dialogue, perspective)
        opp_role = row.get("opp_role")
        if not isinstance(opp_role, str):
            continue
        self_pts = float(points(self_counts, _priorities(dialogue, perspective)))
        opp_pts = float(points(opp_counts, _priorities(dialogue, opp_role)))
        joint_pts = self_pts + opp_pts
        max_joint = _max_joint_points(dialogue)
        for key in _row_keys(svo):
            cells[key]["agent_self_points"].append(self_pts)
            cells[key]["agent_joint_points"].append(joint_pts)
        did = str(row.get("dialogue_id"))
        if did not in first_seen:
            first_seen.add(did)
            integrative = 1.0 if _is_integrative(joint_pts, max_joint) else 0.0
            for key in _row_keys(svo):
                cells[key]["agent_first_offer_integrative"].append(integrative)
            debug.append({
                "source": "agent_matched",
                "dialogue_id": did,
                "turn_index": row.get("turn_index"),
                "svo": svo,
                "lambda": pred.get("lambda"),
                "self_points": self_pts,
                "joint_points": joint_pts,
                "max_joint_points": max_joint,
                "integrative": bool(integrative),
                "bid": bid,
            })
    return debug


def _collect_accept(
    cells: dict[str, dict[str, list[Any]]],
    records: list[Mapping[str, Any]],
    dialogues: Mapping[str, Mapping[str, Any]],
    *,
    perspective: str,
    condition: str,
) -> None:
    pair_key = f"{condition}_accept_pairs"
    correct_key = f"{condition}_accept_correct"
    for row in records:
        if row.get("perspective") != perspective:
            continue
        dialogue = dialogues.get(str(row.get("dialogue_id")))
        if dialogue is None:
            continue
        pred = row.get("pred") or {}
        gold_accept = (row.get("true") or {}).get("accept")
        pred_accept = pred.get("accept")
        pending = row.get("pending_offer")
        eligible = (
            gold_accept is not None
            and pred_accept is not None
            and pending is not None
            and bool(pending.get("to_perspective", False))
        )
        if not eligible:
            continue
        pair = (bool(pred_accept), bool(gold_accept))
        correct = 1.0 if pair[0] == pair[1] else 0.0
        for key in _row_keys(_svo(dialogue, perspective)):
            cells[key][pair_key].append(pair)
            cells[key][correct_key].append(correct)


def _metric_test(agent_vals: list[float], human_vals: list[float]) -> dict[str, Any]:
    test = _welch(agent_vals, human_vals)
    return {
        "agent_minus_human": test["mean_diff_a_minus_b"],
        "welch_p": test.get("welch_p"),
    }


def _build_rows(cells: Mapping[str, Mapping[str, list[Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_key in ROW_ORDER:
        cell = cells.get(row_key, _empty_cell())
        human_self = [float(x) for x in cell["human_self_points"]]
        agent_self = [float(x) for x in cell["agent_self_points"]]
        human_joint = [float(x) for x in cell["human_joint_points"]]
        agent_joint = [float(x) for x in cell["agent_joint_points"]]
        human_int = [float(x) for x in cell["human_first_offer_integrative"]]
        agent_int = [float(x) for x in cell["agent_first_offer_integrative"]]
        match_correct = [float(x) for x in cell["match_accept_correct"]]
        mismatch_correct = [float(x) for x in cell["mismatch_accept_correct"]]
        match_f1 = _binary_f1(cell["match_accept_pairs"])
        mismatch_f1 = _binary_f1(cell["mismatch_accept_pairs"])
        match_test = _welch(match_correct, mismatch_correct)
        self_test = _metric_test(agent_self, human_self)
        joint_test = _metric_test(agent_joint, human_joint)
        int_test = _metric_test(agent_int, human_int)
        rows.append({
            "row": row_key,
            "human_self_mean": _summ(human_self)["mean"],
            "human_self_n": len(human_self),
            "agent_self_mean": _summ(agent_self)["mean"],
            "agent_self_n": len(agent_self),
            "self_agent_minus_human": self_test["agent_minus_human"],
            "self_welch_p": self_test["welch_p"],
            "human_joint_mean": _summ(human_joint)["mean"],
            "human_joint_n": len(human_joint),
            "agent_joint_mean": _summ(agent_joint)["mean"],
            "agent_joint_n": len(agent_joint),
            "joint_agent_minus_human": joint_test["agent_minus_human"],
            "joint_welch_p": joint_test["welch_p"],
            "human_first_offer_integrative_rate": _summ(human_int)["mean"],
            "human_first_offer_integrative_n": len(human_int),
            "agent_first_offer_integrative_rate": _summ(agent_int)["mean"],
            "agent_first_offer_integrative_n": len(agent_int),
            "integrative_agent_minus_human": int_test["agent_minus_human"],
            "integrative_welch_p": int_test["welch_p"],
            "match_accept_f1": match_f1["f1"],
            "match_accept_n": match_f1["support"],
            "mismatch_accept_f1": mismatch_f1["f1"],
            "mismatch_accept_n": mismatch_f1["support"],
            "match_minus_mismatch_accept_f1": (
                float(match_f1["f1"] - mismatch_f1["f1"])
                if math.isfinite(float(match_f1["f1"])) and math.isfinite(float(mismatch_f1["f1"]))
                else float("nan")
            ),
            "match_accuracy": match_f1["accuracy"],
            "mismatch_accuracy": mismatch_f1["accuracy"],
            "match_minus_mismatch_accuracy": match_test["mean_diff_a_minus_b"],
            "match_vs_mismatch_accuracy_welch_p": match_test.get("welch_p"),
        })
    return rows


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def _write_csv(rows: list[Mapping[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_md(rows: list[Mapping[str, Any]], path: Path, metadata: Mapping[str, Any]) -> None:
    lines = [
        "# Day 10 SVO Subgroup Table",
        "",
        f"Matched records: `{metadata['match_records']}`",
        f"Mismatched records: `{metadata.get('mismatch_records') or 'not provided'}`",
        f"Data: `{metadata['data']}`",
        "",
        "First-offer integrativeness = first formal/predicted offer reaches the dialogue's max feasible joint points.",
        "Match-vs-mismatch p-values are Welch tests on accept-decision correctness indicators.",
        "",
        "| Row | Human self | Agent self | Human joint | Agent joint | Human first-offer int. | Agent first-offer int. | Match accept-F1 | Mismatch accept-F1 | Match - mismatch | p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join([
                str(row["row"]),
                f"{_fmt(row['human_self_mean'])} (n={row['human_self_n']})",
                f"{_fmt(row['agent_self_mean'])} (n={row['agent_self_n']})",
                f"{_fmt(row['human_joint_mean'])} (n={row['human_joint_n']})",
                f"{_fmt(row['agent_joint_mean'])} (n={row['agent_joint_n']})",
                f"{_fmt(row['human_first_offer_integrative_rate'])} (n={row['human_first_offer_integrative_n']})",
                f"{_fmt(row['agent_first_offer_integrative_rate'])} (n={row['agent_first_offer_integrative_n']})",
                f"{_fmt(row['match_accept_f1'])} (n={row['match_accept_n']})",
                f"{_fmt(row['mismatch_accept_f1'])} (n={row['mismatch_accept_n']})",
                _fmt(row["match_minus_mismatch_accept_f1"]),
                _fmt(row["match_vs_mismatch_accuracy_welch_p"]),
            ])
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match-records", type=Path, required=True)
    ap.add_argument("--mismatch-records", type=Path, default=None)
    ap.add_argument("--data", type=Path, default=Path("data/casino_test.json"))
    ap.add_argument("--perspective", default="mturk_agent_1")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    dialogues = _load_dialogues(args.data)
    match_records = _load_jsonl(args.match_records)
    mismatch_records = _load_jsonl(args.mismatch_records)
    cells: dict[str, dict[str, list[Any]]] = defaultdict(_empty_cell)
    debug = []
    debug.extend(_collect_human(cells, dialogues, perspective=args.perspective))
    debug.extend(_collect_agent_bids(cells, match_records, dialogues, perspective=args.perspective))
    _collect_accept(cells, match_records, dialogues, perspective=args.perspective, condition="match")
    _collect_accept(cells, mismatch_records, dialogues, perspective=args.perspective, condition="mismatch")

    rows = _build_rows(cells)
    overall = next(row for row in rows if row["row"] == "overall")
    headline = {
        "prediction": "matched_accept_f1_gt_mismatched_accept_f1",
        "match_accept_f1": overall["match_accept_f1"],
        "mismatch_accept_f1": overall["mismatch_accept_f1"],
        "diff": overall["match_minus_mismatch_accept_f1"],
        "welch_p_on_accuracy": overall["match_vs_mismatch_accuracy_welch_p"],
        "pass_pre_registered_direction_p_lt_0.05": (
            isinstance(overall["match_vs_mismatch_accuracy_welch_p"], (int, float))
            and math.isfinite(float(overall["match_vs_mismatch_accuracy_welch_p"]))
            and float(overall["match_minus_mismatch_accept_f1"]) > 0
            and float(overall["match_vs_mismatch_accuracy_welch_p"]) < 0.05
        ),
    }
    report = {
        "metadata": {
            "match_records": str(args.match_records),
            "mismatch_records": str(args.mismatch_records) if args.mismatch_records else None,
            "data": str(args.data),
            "perspective": args.perspective,
            "rows": ROW_ORDER,
        },
        "headline_match_vs_mismatch": headline,
        "rows": rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    csv_path = args.output_dir / "table1_svo_subgroup.csv"
    md_path = args.output_dir / "table1_svo_subgroup.md"
    debug_path = args.output_dir / "first_offer_debug.jsonl"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(rows, csv_path)
    _write_md(rows, md_path, report["metadata"])
    with debug_path.open("w", encoding="utf-8") as f:
        for row in debug:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {summary_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {debug_path}")
    print(json.dumps(headline, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
