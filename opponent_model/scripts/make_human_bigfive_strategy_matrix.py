#!/usr/bin/env python3
"""Compute the human Big Five x strategy-frequency target matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping

import numpy as np

from opponent_model.turn_level_metrics import (
    CASINO_STRATEGIES,
    DEAL_ACTIONS,
    build_annotation_lookup,
)


BIG_FIVE_TRAITS = (
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "emotional-stability",
    "openness-to-experiences",
)


def _load_dialogues(path: Path) -> list[Mapping[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _participant_rows(dialogues: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dialogue in dialogues:
        did = dialogue.get("dialogue_id")
        chat_logs = dialogue.get("chat_logs") or []
        ann_lookup = build_annotation_lookup(dialogue.get("annotations") or [], chat_logs)
        for role, pinfo in (dialogue.get("participant_info") or {}).items():
            bigfive = ((pinfo or {}).get("personality") or {}).get("big-five") or {}
            if not all(t in bigfive for t in BIG_FIVE_TRAITS):
                continue
            counts: Counter[str] = Counter()
            utterance_count = 0
            for idx, turn in enumerate(chat_logs):
                if turn.get("id") != role:
                    continue
                text = str(turn.get("text") or "")
                if text in DEAL_ACTIONS:
                    continue
                utterance_count += 1
                for tag in ann_lookup.get(idx, []):
                    if tag in CASINO_STRATEGIES:
                        counts[tag] += 1
            if utterance_count == 0:
                continue
            row: dict[str, Any] = {
                "dialogue_id": did,
                "role": role,
                "utterance_count": utterance_count,
            }
            for trait in BIG_FIVE_TRAITS:
                row[trait] = float(bigfive[trait])
            for strategy in CASINO_STRATEGIES:
                row[f"freq_{strategy}"] = counts[strategy] / utterance_count
                row[f"count_{strategy}"] = counts[strategy]
            rows.append(row)
    return rows


def _corr(x: list[float], y: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(x), "pearson_r": float("nan"), "pearson_p": None}
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return out
    try:
        from scipy import stats  # type: ignore
    except Exception as exc:  # pragma: no cover - env-dependent
        out["scipy_error"] = repr(exc)
        out["pearson_r"] = float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])
        return out
    res = stats.pearsonr(x, y)
    out["pearson_r"] = float(res.statistic)
    out["pearson_p"] = float(res.pvalue)
    spearman = stats.spearmanr(x, y)
    out["spearman_r"] = float(spearman.statistic) if math.isfinite(float(spearman.statistic)) else None
    out["spearman_p"] = float(spearman.pvalue) if math.isfinite(float(spearman.pvalue)) else None
    return out


def _summary(vals: list[float]) -> dict[str, Any]:
    return {
        "n": len(vals),
        "mean": float(mean(vals)) if vals else float("nan"),
        "std": float(stdev(vals)) if len(vals) > 1 else float("nan"),
    }


def _write_csv(rows: list[Mapping[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_matrix_csv(matrix: list[Mapping[str, Any]], path: Path) -> None:
    fieldnames = ["trait", *CASINO_STRATEGIES]
    by_trait = {row["trait"]: row for row in matrix}
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trait in BIG_FIVE_TRAITS:
            row = {"trait": trait}
            for strategy in CASINO_STRATEGIES:
                row[strategy] = by_trait[(trait, strategy)]["pearson_r"]
            writer.writerow(row)


def _plot_heatmap(matrix: list[Mapping[str, Any]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    by_trait = {row["trait"]: row for row in matrix}
    arr = np.asarray([
        [float(by_trait[(trait, strategy)]["pearson_r"]) for strategy in CASINO_STRATEGIES]
        for trait in BIG_FIVE_TRAITS
    ])
    fig, ax = plt.subplots(figsize=(12, 4.8))
    im = ax.imshow(arr, vmin=-0.35, vmax=0.35, cmap="coolwarm")
    ax.set_xticks(range(len(CASINO_STRATEGIES)))
    ax.set_xticklabels(CASINO_STRATEGIES, rotation=35, ha="right")
    ax.set_yticks(range(len(BIG_FIVE_TRAITS)))
    ax.set_yticklabels(BIG_FIVE_TRAITS)
    ax.set_title("Human Big Five x Strategy Frequency Correlations")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/casino_train.json"))
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("opponent_model/results/day10_human_bigfive_strategy_matrix"),
    )
    args = ap.parse_args()

    dialogues = _load_dialogues(args.data)
    participant_rows = _participant_rows(dialogues)
    matrix: list[dict[str, Any]] = []
    for trait in BIG_FIVE_TRAITS:
        x = [float(row[trait]) for row in participant_rows]
        for strategy in CASINO_STRATEGIES:
            y = [float(row[f"freq_{strategy}"]) for row in participant_rows]
            block = _corr(x, y)
            matrix.append({
                "trait": (trait, strategy),
                "trait_name": trait,
                "strategy": strategy,
                **block,
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "metadata": {
            "data": str(args.data),
            "unit": "dialogue participant",
            "strategy_frequency_denominator": "non-action utterances by that participant",
            "traits": BIG_FIVE_TRAITS,
            "strategies": CASINO_STRATEGIES,
        },
        "support": {
            "dialogues": len(dialogues),
            "participants": len(participant_rows),
            "utterances": int(sum(row["utterance_count"] for row in participant_rows)),
        },
        "trait_summary": {
            trait: _summary([float(row[trait]) for row in participant_rows])
            for trait in BIG_FIVE_TRAITS
        },
        "strategy_frequency_summary": {
            strategy: _summary([float(row[f"freq_{strategy}"]) for row in participant_rows])
            for strategy in CASINO_STRATEGIES
        },
        "correlations": matrix,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(participant_rows, args.output_dir / "participant_strategy_features.csv")
    _write_csv(matrix, args.output_dir / "bigfive_strategy_correlations_long.csv")
    _write_matrix_csv(matrix, args.output_dir / "bigfive_strategy_pearson_matrix.csv")
    _plot_heatmap(matrix, args.output_dir / "bigfive_strategy_pearson_heatmap.png")
    print(f"Wrote {args.output_dir / 'summary.json'}")
    print(f"Wrote {args.output_dir / 'bigfive_strategy_pearson_matrix.csv'}")
    print(f"Wrote {args.output_dir / 'bigfive_strategy_pearson_heatmap.png'}")
    print(json.dumps(summary["support"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
