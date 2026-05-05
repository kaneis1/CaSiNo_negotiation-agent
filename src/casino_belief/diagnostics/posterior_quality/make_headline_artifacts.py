#!/usr/bin/env python3
"""Create Day 9 headline spreadsheet and Brier trajectory figure."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt


MODEL_PATHS = {
    "baseline_live": Path("artifacts/results/protocol3/structured_cot_70b_full150/turn_summary.json"),
    "teacher": Path("artifacts/results/protocol3/bayesian_teacher_full150/turn_summary.json"),
    "student": Path("artifacts/results/protocol3/distilled_student_balanced_full150/turn_summary.json"),
}
TURN_RECORD_PATHS = {
    "teacher": Path("artifacts/results/protocol3/bayesian_teacher_full150/turn_records.jsonl"),
    "student": Path("artifacts/results/protocol3/distilled_student_balanced_full150/turn_records.jsonl"),
}
EXTRACTED_PATH = Path("artifacts/results/bid_coverage/extracted_bid_analysis/summary.json")
# turn_level_metrics.normalized_brier uses the class-mean multiclass Brier:
# mean_k (p_k - 1{k=true})^2. For a uniform six-way posterior this is 5/36.
BRIER_REFERENCE = 5.0 / 36.0
BOOTSTRAP_SEED = 20260504
BOOTSTRAP_REPLICATES = 2000


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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


def _normalized_brier(posterior: Sequence[float], true_index: int) -> float:
    n = len(posterior)
    if n == 0 or true_index < 0 or true_index >= n:
        return float("nan")
    total = 0.0
    for i, prob in enumerate(posterior):
        target = 1.0 if i == true_index else 0.0
        total += (float(prob) - target) ** 2
    return total / n


def _brier_by_dialogue(records: Iterable[Mapping[str, Any]]) -> Dict[Any, Dict[int, List[float]]]:
    by_dialogue: Dict[Any, Dict[int, List[float]]] = {}
    for record in records:
        pred = record.get("pred") or {}
        true = record.get("true") or {}
        posterior = pred.get("posterior")
        true_index = true.get("true_hypothesis_index")
        if posterior is None or true_index is None:
            continue
        brier = _normalized_brier(posterior, int(true_index))
        if math.isnan(brier):
            continue
        dialogue_id = record.get("dialogue_id")
        turn_index = int(record["turn_index"])
        by_dialogue.setdefault(dialogue_id, {}).setdefault(turn_index, []).append(brier)
    return by_dialogue


def _percentile(values: Sequence[float], q: float) -> float | None:
    clean = sorted(v for v in values if not math.isnan(v))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * q
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return clean[int(rank)]
    weight = rank - lo
    return clean[lo] * (1.0 - weight) + clean[hi] * weight


def _bootstrap_brier_ci(
    records_by_model: Mapping[str, Dict[Any, Dict[int, List[float]]]],
    turns: Sequence[int],
    *,
    n_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    model_dialogue_sets = [set(by_dialogue) for by_dialogue in records_by_model.values()]
    dialogue_ids = sorted(set.intersection(*model_dialogue_sets))
    rng = random.Random(seed)
    bootstrap_values: Dict[str, Dict[int, List[float]]] = {
        model: {turn: [] for turn in turns}
        for model in records_by_model
    }

    for _ in range(n_replicates):
        sampled_dialogue_ids = [rng.choice(dialogue_ids) for _ in dialogue_ids]
        for model, by_dialogue in records_by_model.items():
            sums = {turn: 0.0 for turn in turns}
            counts = {turn: 0 for turn in turns}
            for dialogue_id in sampled_dialogue_ids:
                for turn, values in by_dialogue.get(dialogue_id, {}).items():
                    if turn not in sums:
                        continue
                    sums[turn] += sum(values)
                    counts[turn] += len(values)
            for turn in turns:
                if counts[turn]:
                    bootstrap_values[model][turn].append(sums[turn] / counts[turn])

    intervals: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for model, by_turn in bootstrap_values.items():
        intervals[model] = {}
        for turn, values in by_turn.items():
            intervals[model][turn] = {
                "lower": _percentile(values, 0.025),
                "upper": _percentile(values, 0.975),
                "valid_replicates": len(values),
            }

    return {
        "intervals": intervals,
        "metadata": {
            "seed": seed,
            "n_replicates": n_replicates,
            "bootstrap_unit": "dialogue_id",
            "n_dialogues": len(dialogue_ids),
            "ci_method": "percentile",
            "ci_level": 0.95,
        },
    }


def _plot_brier(
    summaries: Mapping[str, Mapping[str, Any]],
    records_by_model: Mapping[str, Dict[Any, Dict[int, List[float]]]],
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
    bootstrap = _bootstrap_brier_ci(records_by_model, turns)
    intervals = bootstrap["intervals"]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    teacher_color = "#1f1f1f"
    student_color = "#b33a3a"
    reference_color = "#9a9a9a"
    teacher_ci = intervals["teacher"]
    student_ci = intervals["student"]

    ax.fill_between(
        turns,
        [teacher_ci[t]["lower"] for t in turns],
        [teacher_ci[t]["upper"] for t in turns],
        color=teacher_color,
        alpha=0.12,
        linewidth=0,
        label="_nolegend_",
    )
    ax.fill_between(
        turns,
        [student_ci[t]["lower"] for t in turns],
        [student_ci[t]["upper"] for t in turns],
        color=student_color,
        alpha=0.14,
        linewidth=0,
        label="_nolegend_",
    )

    ax.plot(
        turns,
        [teacher.get(t, float("nan")) for t in turns],
        color=teacher_color,
        linewidth=2.4,
        label="Bayesian teacher",
    )
    ax.plot(
        turns,
        [student.get(t, float("nan")) for t in turns],
        color=student_color,
        linewidth=2.4,
        label="Distilled student",
    )
    ax.axhline(
        BRIER_REFERENCE,
        color=reference_color,
        linestyle="--",
        linewidth=1.3,
        label="Uniform reference (5/36)",
    )
    ax.set_xlabel("Turn index")
    ax.set_ylabel("Normalized Brier score")
    max_ci = max(
        value["upper"]
        for model_intervals in (teacher_ci, student_ci)
        for value in model_intervals.values()
        if value["upper"] is not None
    )
    ax.set_ylim(0.0, max(0.18, math.ceil((max_ci + 0.005) / 0.02) * 0.02))
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.grid(axis="x", color="#f1f1f1", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#bcbcbc")
    ax.spines["bottom"].set_color("#bcbcbc")
    ax.tick_params(axis="both", colors="#3a3a3a", labelsize=9)
    legend = ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#dddddd",
        fontsize=9,
    )
    for line in legend.get_lines():
        line.set_linewidth(2.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

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
        "bootstrap": bootstrap["metadata"],
        "bootstrap_ci_by_turn": {
            model: {
                str(turn): {
                    "lower": intervals[model][turn]["lower"],
                    "upper": intervals[model][turn]["upper"],
                    "valid_replicates": intervals[model][turn]["valid_replicates"],
                }
                for turn in turns
            }
            for model in ("teacher", "student")
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/results/posterior_quality"))
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {name: _load_json(path) for name, path in MODEL_PATHS.items()}
    records_by_model = {
        name: _brier_by_dialogue(_load_jsonl(path))
        for name, path in TURN_RECORD_PATHS.items()
    }

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
            "note": "Flat dashed class-mean Brier reference for plot",
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
    trimmed_checks = _plot_brier(summaries, records_by_model, figure_path, min_support=10)
    all_turn_checks = _plot_brier(summaries, records_by_model, all_turn_figure_path, min_support=1)
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
        print("WARNING: student Brier exceeds the 5/36 uniform reference on the main trimmed plot.")
    if not all_turn_checks["student_never_worse_than_reference"]:
        print("WARNING: student Brier exceeds the 5/36 uniform reference on at least one turn.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
