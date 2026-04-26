#!/usr/bin/env python3
"""Compare fixed-style student strategy distributions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from opponent_model.turn_level_metrics import CASINO_STRATEGIES


DEFAULT_RUNS = {
    "balanced": Path("opponent_model/results/turn_eval_student_balanced_full150/turn_records.jsonl"),
    "cooperative": Path("opponent_model/results/turn_eval_student_cooperative_full150/turn_records.jsonl"),
    "competitive": Path("opponent_model/results/turn_eval_student_competitive_full150/turn_records.jsonl"),
}


def _load_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _run_summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    turn_count = 0
    tagged_turn_count = 0
    for row in records:
        turn_count += 1
        tags = list((row.get("pred") or {}).get("strategy") or [])
        if tags:
            tagged_turn_count += 1
        for tag in tags:
            if tag in CASINO_STRATEGIES:
                counts[tag] += 1
    incidence = {
        strategy: counts[strategy] / turn_count if turn_count else float("nan")
        for strategy in CASINO_STRATEGIES
    }
    total_tags = sum(counts.values())
    distribution = {
        strategy: counts[strategy] / total_tags if total_tags else float("nan")
        for strategy in CASINO_STRATEGIES
    }
    return {
        "turns": turn_count,
        "tagged_turns": tagged_turn_count,
        "total_strategy_tags": total_tags,
        "counts": dict(counts),
        "incidence_per_turn": incidence,
        "distribution_over_tags": distribution,
    }


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    eps = 1e-12
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m)))


def _global_chi_square(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"scipy_available": False}
    table = np.asarray([
        [int((summary.get("counts") or {}).get(strategy, 0)) for strategy in CASINO_STRATEGIES]
        for summary in summaries.values()
    ])
    out["table"] = table.tolist()
    nonzero_cols = table.sum(axis=0) > 0
    out["tested_strategies"] = [
        strategy for strategy, keep in zip(CASINO_STRATEGIES, nonzero_cols) if bool(keep)
    ]
    test_table = table[:, nonzero_cols]
    if test_table.shape[1] < 2:
        out["error"] = "fewer than two nonzero strategy columns"
        return out
    try:
        from scipy import stats  # type: ignore
    except Exception as exc:  # pragma: no cover - env-dependent
        out["scipy_error"] = repr(exc)
        return out
    out["scipy_available"] = True
    res = stats.chi2_contingency(test_table)
    out["chi2"] = float(res.statistic)
    out["p"] = float(res.pvalue)
    out["dof"] = int(res.dof)
    return out


def _write_csv(rows: list[Mapping[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot(summaries: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels = list(summaries)
    x = np.arange(len(CASINO_STRATEGIES))
    width = 0.24
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for i, label in enumerate(labels):
        vals = [
            float((summaries[label].get("incidence_per_turn") or {}).get(strategy, 0.0))
            for strategy in CASINO_STRATEGIES
        ]
        ax.bar(x + (i - (len(labels) - 1) / 2) * width, vals, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(CASINO_STRATEGIES, rotation=35, ha="right")
    ax.set_ylabel("Strategy incidence per turn")
    ax.set_title("Fixed-Style Student Strategy Distributions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for label, path in DEFAULT_RUNS.items():
        ap.add_argument(f"--{label}-records", type=Path, default=path)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("opponent_model/results/day10_style_variant_strategy_distributions"),
    )
    args = ap.parse_args()

    paths = {
        "balanced": args.balanced_records,
        "cooperative": args.cooperative_records,
        "competitive": args.competitive_records,
    }
    summaries = {label: _run_summary(_load_records(path)) for label, path in paths.items()}
    chi = _global_chi_square(summaries)
    pairwise: dict[str, Any] = {}
    labels = list(paths)
    for a, b in combinations(labels, 2):
        pa = np.asarray([
            float((summaries[a]["distribution_over_tags"] or {}).get(strategy, 0.0))
            for strategy in CASINO_STRATEGIES
        ])
        pb = np.asarray([
            float((summaries[b]["distribution_over_tags"] or {}).get(strategy, 0.0))
            for strategy in CASINO_STRATEGIES
        ])
        max_abs_incidence_diff = max(
            abs(float(summaries[a]["incidence_per_turn"][s]) - float(summaries[b]["incidence_per_turn"][s]))
            for s in CASINO_STRATEGIES
        )
        pairwise[f"{a}_vs_{b}"] = {
            "js_divergence_bits": _js_divergence(pa, pb),
            "max_abs_incidence_diff": float(max_abs_incidence_diff),
        }

    strongest = max(
        (v["max_abs_incidence_diff"] for v in pairwise.values()),
        default=float("nan"),
    )
    gate = {
        "global_chi_square_p_lt_0.05": (
            isinstance(chi.get("p"), (int, float)) and float(chi["p"]) < 0.05
        ),
        "max_abs_incidence_diff_ge_0.02": (
            isinstance(strongest, (int, float)) and math.isfinite(float(strongest)) and strongest >= 0.02
        ),
        "claim3_style_token_load_bearing": (
            isinstance(chi.get("p"), (int, float))
            and float(chi["p"]) < 0.05
            and isinstance(strongest, (int, float))
            and math.isfinite(float(strongest))
            and strongest >= 0.02
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, summary in summaries.items():
        row = {"style": label, "turns": summary["turns"], "total_strategy_tags": summary["total_strategy_tags"]}
        row.update({f"incidence_{s}": summary["incidence_per_turn"][s] for s in CASINO_STRATEGIES})
        rows.append(row)
    report = {
        "metadata": {
            "records": {label: str(path) for label, path in paths.items()},
            "strategies": CASINO_STRATEGIES,
            "frequency_definition": "multi-label strategy incidence per evaluated turn",
            "gate": "global chi-square p<0.05 and max pairwise incidence difference >=0.02",
        },
        "summaries": summaries,
        "global_chi_square": chi,
        "pairwise": pairwise,
        "gate": gate,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(rows, args.output_dir / "strategy_incidence_by_style.csv")
    _plot(summaries, args.output_dir / "strategy_incidence_by_style.png")
    print(f"Wrote {args.output_dir / 'summary.json'}")
    print(f"Wrote {args.output_dir / 'strategy_incidence_by_style.csv'}")
    print(f"Wrote {args.output_dir / 'strategy_incidence_by_style.png'}")
    print(json.dumps(gate, indent=2))
    return 0 if gate["claim3_style_token_load_bearing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
