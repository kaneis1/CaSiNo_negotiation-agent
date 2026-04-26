#!/usr/bin/env python3
"""Build the Day 10 SVO x integrative-potential Table 1.

Rows are the 2 x 3 cells requested for Claim 2:

    SVO category (proself/prosocial) x IP tercile (low/mid/high)

For each cell we report human outcome self-/joint-points, SVO-conditioned
agent bid self-/joint-points, accept-F1 for the agent, and an oracle human
accept-F1. Welch p-values test agent-vs-human *differences* for point metrics
and accept correctness. They are not formal equivalence tests; use TOST with a
pre-registered margin if the paper needs the word "equivalent."
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
from typing import Any, Dict, Iterable, Mapping, Optional

from opponent_model.hypotheses import ITEMS
from sft_8b.menu import points


ROLES = ("mturk_agent_1", "mturk_agent_2")
SVO_ORDER = ("proself", "prosocial")
IP_ORDER = ("low_ip", "mid_ip", "high_ip")


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_dialogues(path: Path) -> Dict[str, Mapping[str, Any]]:
    with path.open() as f:
        data = json.load(f)
    return {str(d.get("dialogue_id", i)): d for i, d in enumerate(data)}


def _counts_from_bid(bid: Any, offset: int) -> Optional[Dict[str, int]]:
    if not isinstance(bid, (list, tuple)) or len(bid) != 6:
        return None
    try:
        return {item: int(float(bid[offset + i])) for i, item in enumerate(ITEMS)}
    except (TypeError, ValueError):
        return None


def _max_joint_points(
    mt1_priorities: Mapping[str, str],
    mt2_priorities: Mapping[str, str],
) -> int:
    best = -1
    for split in product(range(4), repeat=3):
        mt1_counts = {item: int(split[i]) for i, item in enumerate(ITEMS)}
        mt2_counts = {item: 3 - mt1_counts[item] for item in ITEMS}
        joint = points(mt1_counts, mt1_priorities) + points(mt2_counts, mt2_priorities)
        best = max(best, int(joint))
    return best


def _ip_bin(max_joint: int) -> str:
    # Natural terciles for the discrete CaSiNo 3-item utility space on this split.
    if max_joint <= 36:
        return "low_ip"
    if max_joint <= 39:
        return "mid_ip"
    return "high_ip"


def _dialogue_ip(dialogue: Mapping[str, Any]) -> Dict[str, Any]:
    pinfo = dialogue.get("participant_info") or {}
    max_joint = _max_joint_points(
        pinfo["mturk_agent_1"]["value2issue"],
        pinfo["mturk_agent_2"]["value2issue"],
    )
    return {"max_joint_points": max_joint, "ip_bin": _ip_bin(max_joint)}


def _svo(dialogue: Mapping[str, Any], role: str) -> str:
    return str(
        ((dialogue.get("participant_info") or {})
         .get(role, {})
         .get("personality", {})
         .get("svo", "unclassified"))
    ).strip().lower()


def _outcome_points(dialogue: Mapping[str, Any], role: str) -> float:
    return float(
        ((dialogue.get("participant_info") or {})
         .get(role, {})
         .get("outcomes", {})
         .get("points_scored"))
    )


def _summ(vals: Iterable[float]) -> Dict[str, Any]:
    arr = list(vals)
    return {
        "n": len(arr),
        "mean": float(mean(arr)) if arr else float("nan"),
        "std": float(stdev(arr)) if len(arr) > 1 else float("nan"),
    }


def _binary_f1(pairs: list[tuple[bool, bool]]) -> Dict[str, Any]:
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


def _welch(agent_vals: list[float], human_vals: list[float]) -> Dict[str, Any]:
    out = {
        "mean_diff_agent_minus_human": (
            float(mean(agent_vals) - mean(human_vals))
            if agent_vals and human_vals else float("nan")
        ),
        "scipy_available": False,
    }
    try:
        from scipy import stats  # type: ignore
    except Exception as exc:  # pragma: no cover - env-dependent
        out["scipy_error"] = repr(exc)
        return out
    out["scipy_available"] = True
    if len(agent_vals) >= 2 and len(human_vals) >= 2:
        res = stats.ttest_ind(agent_vals, human_vals, equal_var=False, nan_policy="omit")
        out["welch_t"] = float(res.statistic) if math.isfinite(float(res.statistic)) else None
        out["welch_p"] = float(res.pvalue) if math.isfinite(float(res.pvalue)) else None
    return out


def _cell() -> Dict[str, list[Any]]:
    return {
        "human_self_points": [],
        "human_joint_points": [],
        "human_accept_correct": [],
        "agent_self_points": [],
        "agent_joint_points": [],
        "agent_accept_pairs": [],
        "agent_accept_correct": [],
    }


def _collect_human(
    dialogues: Mapping[str, Mapping[str, Any]],
    *,
    perspective: str,
) -> Dict[tuple[str, str], Dict[str, list[Any]]]:
    cells: Dict[tuple[str, str], Dict[str, list[Any]]] = defaultdict(_cell)
    for dialogue in dialogues.values():
        svo = _svo(dialogue, perspective)
        if svo not in SVO_ORDER:
            continue
        ip = _dialogue_ip(dialogue)["ip_bin"]
        joint = sum(_outcome_points(dialogue, role) for role in ROLES)
        key = (svo, ip)
        cells[key]["human_self_points"].append(_outcome_points(dialogue, perspective))
        cells[key]["human_joint_points"].append(joint)
    return cells


def _merge_cells(
    base: Dict[tuple[str, str], Dict[str, list[Any]]],
    extra: Dict[tuple[str, str], Dict[str, list[Any]]],
) -> Dict[tuple[str, str], Dict[str, list[Any]]]:
    for key, block in extra.items():
        cell = base[key]
        for metric, values in block.items():
            cell[metric].extend(values)
    return base


def _collect_agent(
    records: list[Mapping[str, Any]],
    dialogues: Mapping[str, Mapping[str, Any]],
    *,
    perspective: str,
) -> Dict[tuple[str, str], Dict[str, list[Any]]]:
    cells: Dict[tuple[str, str], Dict[str, list[Any]]] = defaultdict(_cell)
    for row in records:
        if row.get("perspective") != perspective:
            continue
        dialogue = dialogues.get(str(row.get("dialogue_id")))
        if dialogue is None:
            continue
        svo = _svo(dialogue, perspective)
        if svo not in SVO_ORDER:
            continue
        ip = _dialogue_ip(dialogue)["ip_bin"]
        key = (svo, ip)
        pred = row.get("pred") or {}
        bid = pred.get("bid")
        if bid is not None:
            self_counts = _counts_from_bid(bid, 0)
            opp_counts = _counts_from_bid(bid, 3)
            if self_counts is not None and opp_counts is not None:
                pinfo = dialogue.get("participant_info") or {}
                self_pts = float(points(self_counts, pinfo[perspective]["value2issue"]))
                opp_pts = float(points(opp_counts, pinfo[row["opp_role"]]["value2issue"]))
                cells[key]["agent_self_points"].append(self_pts)
                cells[key]["agent_joint_points"].append(self_pts + opp_pts)

        gold_accept = (row.get("true") or {}).get("accept")
        pred_accept = pred.get("accept")
        pending = row.get("pending_offer")
        eligible = (
            gold_accept is not None
            and pred_accept is not None
            and pending is not None
            and bool(pending.get("to_perspective", False))
        )
        if eligible:
            pair = (bool(pred_accept), bool(gold_accept))
            cells[key]["agent_accept_pairs"].append(pair)
            cells[key]["agent_accept_correct"].append(1.0 if pair[0] == pair[1] else 0.0)
            # Oracle human row: the human gold action compared with itself.
            cells[key]["human_accept_correct"].append(1.0)
    return cells


def _row(
    *,
    svo: str,
    ip: str,
    cell: Mapping[str, list[Any]],
) -> Dict[str, Any]:
    human_self = [float(x) for x in cell.get("human_self_points", [])]
    human_joint = [float(x) for x in cell.get("human_joint_points", [])]
    agent_self = [float(x) for x in cell.get("agent_self_points", [])]
    agent_joint = [float(x) for x in cell.get("agent_joint_points", [])]
    human_accept_correct = [float(x) for x in cell.get("human_accept_correct", [])]
    agent_accept_correct = [float(x) for x in cell.get("agent_accept_correct", [])]
    agent_f1 = _binary_f1(list(cell.get("agent_accept_pairs", [])))
    human_f1 = {
        "f1": 1.0 if human_accept_correct else float("nan"),
        "support": len(human_accept_correct),
        "accuracy": 1.0 if human_accept_correct else float("nan"),
    }
    self_test = _welch(agent_self, human_self)
    joint_test = _welch(agent_joint, human_joint)
    accept_test = _welch(agent_accept_correct, human_accept_correct)
    return {
        "svo": svo,
        "ip_tercile": ip,
        "human_self_mean": _summ(human_self)["mean"],
        "human_self_n": len(human_self),
        "agent_self_mean": _summ(agent_self)["mean"],
        "agent_self_n": len(agent_self),
        "self_welch_p": self_test.get("welch_p"),
        "self_agent_minus_human": self_test.get("mean_diff_agent_minus_human"),
        "human_joint_mean": _summ(human_joint)["mean"],
        "human_joint_n": len(human_joint),
        "agent_joint_mean": _summ(agent_joint)["mean"],
        "agent_joint_n": len(agent_joint),
        "joint_welch_p": joint_test.get("welch_p"),
        "joint_agent_minus_human": joint_test.get("mean_diff_agent_minus_human"),
        "human_accept_f1": human_f1["f1"],
        "human_accept_n": human_f1["support"],
        "agent_accept_f1": agent_f1["f1"],
        "agent_accept_n": agent_f1["support"],
        "accept_accuracy_welch_p": accept_test.get("welch_p"),
        "accept_accuracy_agent_minus_human": accept_test.get("mean_diff_agent_minus_human"),
    }


def _fmt_num(value: Any, digits: int = 3) -> str:
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
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(rows: list[Mapping[str, Any]], path: Path, *, metadata: Mapping[str, Any]) -> None:
    header = [
        "| SVO | IP tercile | Human self | Agent self | p | Human joint | Agent joint | p | Human accept-F1 | Agent accept-F1 | p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines = [
        "# Day 10 Table 1: SVO x Integrative Potential",
        "",
        f"Records: `{metadata['records']}`",
        f"Data: `{metadata['data']}`",
        "",
        "Welch p-values test agent-vs-human mean differences. They are not formal equivalence tests.",
        "",
    ]
    lines.extend(header)
    for row in rows:
        lines.append(
            "| "
            + " | ".join([
                str(row["svo"]),
                str(row["ip_tercile"]).replace("_ip", ""),
                f"{_fmt_num(row['human_self_mean'])} (n={row['human_self_n']})",
                f"{_fmt_num(row['agent_self_mean'])} (n={row['agent_self_n']})",
                _fmt_num(row["self_welch_p"]),
                f"{_fmt_num(row['human_joint_mean'])} (n={row['human_joint_n']})",
                f"{_fmt_num(row['agent_joint_mean'])} (n={row['agent_joint_n']})",
                _fmt_num(row["joint_welch_p"]),
                f"{_fmt_num(row['human_accept_f1'])} (n={row['human_accept_n']})",
                f"{_fmt_num(row['agent_accept_f1'])} (n={row['agent_accept_n']})",
                _fmt_num(row["accept_accuracy_welch_p"]),
            ])
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--records",
        type=Path,
        default=Path("opponent_model/results/turn_eval_bayesian_svo_lambda_m5_f0.50_full150/turn_records.jsonl"),
    )
    ap.add_argument("--data", type=Path, default=Path("data/casino_test.json"))
    ap.add_argument("--perspective", default="mturk_agent_1")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("opponent_model/results/day10_svo_ip_table"),
    )
    args = ap.parse_args()

    dialogues = _load_dialogues(args.data)
    records = _load_jsonl(args.records)
    cells = _collect_human(dialogues, perspective=args.perspective)
    cells = _merge_cells(cells, _collect_agent(records, dialogues, perspective=args.perspective))

    rows = [
        _row(svo=svo, ip=ip, cell=cells[(svo, ip)])
        for svo in SVO_ORDER
        for ip in IP_ORDER
    ]
    pvals = [
        row[key]
        for row in rows
        for key in ("self_welch_p", "joint_welch_p", "accept_accuracy_welch_p")
        if row.get(key) is not None
    ]
    nonsig = [
        p for p in pvals
        if isinstance(p, (int, float)) and math.isfinite(float(p)) and float(p) >= 0.05
    ]
    report = {
        "metadata": {
            "records": str(args.records),
            "data": str(args.data),
            "perspective": args.perspective,
            "ip_definition": (
                "low/mid/high bins from max feasible joint points: <=36, 37-39, >=40"
            ),
            "human_accept_f1_definition": (
                "oracle gold action compared with itself on the same eligible accept turns"
            ),
            "statistical_note": (
                "Welch p-values are difference tests, not formal equivalence tests."
            ),
        },
        "rows": rows,
        "claim_check": {
            "welch_tests_count": len(pvals),
            "welch_tests_non_significant_at_0.05": len(nonsig),
            "all_reported_welch_tests_non_significant": len(nonsig) == len(pvals),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    csv_path = args.output_dir / "table1_svo_ip.csv"
    md_path = args.output_dir / "table1_svo_ip.md"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(rows, csv_path)
    _write_md(rows, md_path, metadata=report["metadata"])
    print(f"Wrote {summary_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(json.dumps(report["claim_check"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
