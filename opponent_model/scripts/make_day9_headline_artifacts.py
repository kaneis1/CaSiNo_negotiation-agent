#!/usr/bin/env python3
"""Create Day 9 headline spreadsheet and Brier trajectory figure."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import matplotlib.pyplot as plt


MODEL_PATHS = {
    "baseline_live": Path("opponent_model/results/turn_eval_structured_cot_p3_live_70b_m1_150/turn_summary.json"),
    "teacher": Path("opponent_model/results/turn_eval_bayesian_lambda1.0_m5_f0.50_full150/turn_summary.json"),
    "student": Path("opponent_model/results/turn_eval_student_balanced_full150/turn_summary.json"),
}
EXTRACTED_PATH = Path("opponent_model/results/day9_extracted_bid_analysis/summary.json")
BRIER_REFERENCE = 1.0 / 6.0


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _safe(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return value


def _metric_rows(model: str, summary: Mapping[str, Any], source_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metric, block, key in [
        ("accept_f1", summary.get("accept") or {}, "f1"),
        ("accept_precision", summary.get("accept") or {}, "precision"),
        ("accept_recall", summary.get("accept") or {}, "recall"),
        ("accept_accuracy", summary.get("accept") or {}, "accuracy"),
        ("native_bid_cosine", summary.get("bid_cosine") or {}, "mean"),
        ("strategy_macro_f1", summary.get("strategy_macro_f1") or {}, "macro_f1"),
        ("brier_mean", summary.get("brier") or {}, "mean"),
    ]:
        rows.append({
            "section": "summary_metric",
            "model": model,
            "metric": metric,
            "turn_index": "",
            "value": _safe(block.get(key)),
            "support": block.get("support", ""),
            "source_path": str(source_path),
            "note": "",
        })
    return rows


def _brier_rows(model: str, summary: Mapping[str, Any], source_path: Path) -> List[Dict[str, Any]]:
    return [
        {
            "section": "brier_by_turn_index",
            "model": model,
            "metric": "brier",
            "turn_index": row["turn_index"],
            "value": row["mean"],
            "support": row["support"],
            "source_path": str(source_path),
            "note": "",
        }
        for row in (summary.get("brier_by_turn_index") or [])
    ]


def _extracted_rows(extracted: Mapping[str, Any], source_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model, block in (extracted.get("agents") or {}).items():
        for support_name in ("native", "extracted"):
            support = block.get(support_name) or {}
            rows.extend([
                {
                    "section": f"{support_name}_bid",
                    "model": model,
                    "metric": "bid_cosine",
                    "turn_index": "",
                    "value": support.get("bid_cosine_mean"),
                    "support": support.get("scored_overlap"),
                    "source_path": str(source_path),
                    "note": f"gold_submit_turns={support.get('gold_submit_turns')}; predicted_bid_turns={support.get('predicted_bid_turns')}",
                },
                {
                    "section": f"{support_name}_bid",
                    "model": model,
                    "metric": "coverage_vs_gold",
                    "turn_index": "",
                    "value": support.get("coverage_vs_gold"),
                    "support": support.get("gold_submit_turns"),
                    "source_path": str(source_path),
                    "note": "",
                },
            ])
    three_way = ((extracted.get("extracted_intersections") or {}).get("three_way") or {})
    for model, value in (three_way.get("bid_cosine_mean") or {}).items():
        rows.append({
            "section": "extracted_three_way_bid",
            "model": model,
            "metric": "bid_cosine",
            "turn_index": "",
            "value": value,
            "support": three_way.get("count"),
            "source_path": str(source_path),
            "note": "three-way shared extracted support",
        })
    return rows


def _write_spreadsheet(rows: Iterable[Mapping[str, Any]], out_path: Path) -> None:
    fieldnames = [
        "section", "model", "metric", "turn_index", "value", "support",
        "source_path", "note",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _series(
    summary: Mapping[str, Any],
    *,
    min_support: int = 1,
) -> Dict[int, float]:
    return {
        int(row["turn_index"]): float(row["mean"])
        for row in (summary.get("brier_by_turn_index") or [])
        if int(row.get("support", 0)) >= min_support
    }


def _support_by_turn(summary: Mapping[str, Any]) -> Dict[int, int]:
    return {
        int(row["turn_index"]): int(row.get("support", 0))
        for row in (summary.get("brier_by_turn_index") or [])
    }


def _plot_brier(
    summaries: Mapping[str, Mapping[str, Any]],
    out_path: Path,
    *,
    min_support: int = 1,
) -> Dict[str, Any]:
    student = _series(summaries["student"], min_support=min_support)
    teacher = _series(summaries["teacher"], min_support=min_support)
    turns = sorted(set(student) & set(teacher))
    student_support = _support_by_turn(summaries["student"])
    teacher_support = _support_by_turn(summaries["teacher"])

    bad_student = [
        {"turn_index": t, "student": student[t], "reference": BRIER_REFERENCE}
        for t in turns
        if t in student and student[t] > BRIER_REFERENCE
    ]
    not_between = [
        {"turn_index": t, "teacher": teacher[t], "student": student[t], "reference": BRIER_REFERENCE}
        for t in turns
        if t in student and t in teacher and not (teacher[t] <= student[t] <= BRIER_REFERENCE)
    ]

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(turns, [teacher.get(t, float("nan")) for t in turns], marker="o", linewidth=2.0, label="Bayesian teacher")
    plt.plot(turns, [student.get(t, float("nan")) for t in turns], marker="s", linewidth=2.0, label="Distilled student")
    plt.axhline(BRIER_REFERENCE, color="black", linestyle="--", linewidth=1.6, label="Baseline reference (1/6)")
    plt.xlabel("Turn index")
    plt.ylabel("Normalized Brier score")
    plt.ylim(0.0, 0.18)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

    return {
        "min_support": min_support,
        "reference_brier": BRIER_REFERENCE,
        "turn_indices": turns,
        "student_support_by_turn": {str(t): student_support.get(t) for t in turns},
        "teacher_support_by_turn": {str(t): teacher_support.get(t) for t in turns},
        "student_max_brier": max(student.values()) if student else None,
        "student_never_worse_than_reference": not bad_student,
        "student_between_teacher_and_reference_all_turns": not not_between,
        "student_worse_than_reference_turns": bad_student,
        "student_not_between_teacher_and_reference_turns": not_between,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=Path("opponent_model/results/day9_headline_artifacts"))
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {name: _load_json(path) for name, path in MODEL_PATHS.items()}

    rows: List[Dict[str, Any]] = []
    for name, path in MODEL_PATHS.items():
        rows.extend(_metric_rows(name, summaries[name], path))
        rows.extend(_brier_rows(name, summaries[name], path))
    rows.extend([
        {
            "section": "brier_reference",
            "model": "baseline_reference",
            "metric": "brier",
            "turn_index": row["turn_index"],
            "value": BRIER_REFERENCE,
            "support": "",
            "source_path": "constant: uniform 6-way posterior",
            "note": "Flat dashed baseline reference for plot",
        }
        for row in (summaries["student"].get("brier_by_turn_index") or [])
    ])

    if EXTRACTED_PATH.exists():
        rows.extend(_extracted_rows(_load_json(EXTRACTED_PATH), EXTRACTED_PATH))

    spreadsheet_path = args.output_dir / "headline_numbers.csv"
    figure_path = args.output_dir / "brier_trajectory.png"
    all_turn_figure_path = args.output_dir / "brier_trajectory_all_turns.png"
    checks_path = args.output_dir / "headline_checks.json"
    _write_spreadsheet(rows, spreadsheet_path)
    trimmed_checks = _plot_brier(summaries, figure_path, min_support=10)
    all_turn_checks = _plot_brier(summaries, all_turn_figure_path, min_support=1)
    checks = {
        "main_plot": trimmed_checks,
        "diagnostic_all_turns": all_turn_checks,
        "caption_note": (
            "Main plot keeps turn indices with support n >= 10 for both "
            "teacher and student; all-turn plot is diagnostic."
        ),
    }
    checks_path.write_text(json.dumps(checks, indent=2), encoding="utf-8")

    print(f"Wrote {spreadsheet_path}")
    print(f"Wrote {figure_path}")
    print(f"Wrote {all_turn_figure_path}")
    print(f"Wrote {checks_path}")
    if not trimmed_checks["student_never_worse_than_reference"]:
        print("ERROR: student Brier exceeds 1/6 reference on the main trimmed plot.")
        return 2
    if not all_turn_checks["student_never_worse_than_reference"]:
        print("ERROR: student Brier exceeds 1/6 reference on at least one turn.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
