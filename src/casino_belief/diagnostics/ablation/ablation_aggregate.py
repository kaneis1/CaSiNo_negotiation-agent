"""Aggregate NeurIPS-2026 ablation artifacts into tables and plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

DEFAULT_ROOT = Path("artifacts/results/ablation/main")


def _fmt_float(value: Any, digits: int = 4) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(v):
        return ""
    return f"{v:.{digits}f}"


def _summary_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("turn_summary.json")):
        try:
            obj = json.loads(path.read_text())
        except Exception:
            continue
        cfg = obj.get("config") or {}
        agent = obj.get("agent_summary") or {}
        rows.append({
            "run": str(path.parent.relative_to(root)),
            "path": str(path),
            "provider": cfg.get("provider"),
            "schema": cfg.get("schema"),
            "evidence_mode": cfg.get("evidence_mode"),
            "prefix_mode": cfg.get("prefix_mode"),
            "annotated_only": cfg.get("annotated_only"),
            "lambda": cfg.get("lambda_"),
            "posterior_temperature": cfg.get("posterior_temperature"),
            "likelihood_temperature": cfg.get("likelihood_temperature"),
            "likelihood_clip": cfg.get("likelihood_clip"),
            "n_dialogues": obj.get("n_dialogues"),
            "n_records": obj.get("n_records"),
            "brier": (obj.get("brier") or {}).get("mean"),
            "brier_support": (obj.get("brier") or {}).get("support"),
            "map_accuracy": (obj.get("posterior_diagnostics") or {}).get("map_accuracy"),
            "entropy_bits": (obj.get("posterior_diagnostics") or {}).get("entropy_bits_mean"),
            "accept_f1": (obj.get("accept") or {}).get("f1"),
            "accept_support": (obj.get("accept") or {}).get("support"),
            "bid_cosine": (obj.get("bid_cosine") or {}).get("mean"),
            "bid_support": (obj.get("bid_cosine") or {}).get("support"),
            "strategy_macro_f1": (obj.get("strategy_macro_f1") or {}).get("macro_f1"),
            "strategy_support": (obj.get("strategy_macro_f1") or {}).get("support"),
            "pareto_rate": (obj.get("pareto_efficiency") or {}).get("pareto_efficient_rate"),
            "pareto_support": (obj.get("pareto_efficiency") or {}).get("support"),
            "annotated_dialogues": (obj.get("annotation_support") or {}).get("annotated_dialogues"),
            "matched_annotation_rows": (obj.get("annotation_support") or {}).get("matched_annotation_rows"),
            "natural_rows": (obj.get("annotation_support") or {}).get("natural_rows"),
            "prefix_action_or_bid_changed_rate": agent.get("prefix_action_or_bid_changed_rate"),
            "mechanically_confounded": agent.get("mechanically_confounded"),
        })
    return rows


def _intervention_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("*intervention*.json")):
        if path.name == "turn_summary.json":
            continue
        try:
            obj = json.loads(path.read_text())
        except Exception:
            continue
        if "intervention_sensitivity_rate" not in obj:
            continue
        rows.append({
            "run": str(path.parent.relative_to(root)),
            "path": str(path),
            "mode": obj.get("mode"),
            "support": obj.get("support"),
            "belief_action_consistency_rate": obj.get("belief_action_consistency_rate"),
            "belief_action_consistency_support": obj.get("belief_action_consistency_support"),
            "intervention_sensitivity_rate": obj.get("intervention_sensitivity_rate"),
            "allocation_drift_l2_mean": obj.get("allocation_drift_l2_mean"),
            "student_to_intervention_drift_l2_mean": obj.get("student_to_intervention_drift_l2_mean"),
        })
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    headers = [
        "run", "Brier", "MAP", "Entropy", "Accept-F1", "Bid cos",
        "Strat-F1", "Pareto", "n_records",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        vals = [
            str(row.get("run", "")),
            _fmt_float(row.get("brier")),
            _fmt_float(row.get("map_accuracy")),
            _fmt_float(row.get("entropy_bits")),
            _fmt_float(row.get("accept_f1")),
            _fmt_float(row.get("bid_cosine")),
            _fmt_float(row.get("strategy_macro_f1")),
            _fmt_float(row.get("pareto_rate")),
            str(row.get("n_records") or ""),
        ]
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_brier_trajectories(root: Path, out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = 0
    for path in sorted(root.rglob("turn_summary.json")):
        try:
            obj = json.loads(path.read_text())
        except Exception:
            continue
        curve = obj.get("brier_by_turn_index") or []
        if not curve:
            continue
        xs = [int(r["turn_index"]) for r in curve]
        ys = [float(r["mean"]) for r in curve]
        ax.plot(xs, ys, marker="o", linewidth=1.2, markersize=2.5, label=str(path.parent.relative_to(root))[:60])
        plotted += 1
    if not plotted:
        return
    ax.set_xlabel("Turn index")
    ax.set_ylabel("Brier")
    ax.set_title("Brier Trajectories")
    ax.legend(fontsize=6, ncol=1, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_sweep_heatmap(rows: Sequence[Mapping[str, Any]], out_path: Path) -> None:
    sweep_rows = [
        r for r in rows
        if r.get("brier") is not None
        and (
            r.get("likelihood_temperature") is not None
            or r.get("posterior_temperature") is not None
        )
    ]
    if not sweep_rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return
    temps = sorted({
        str(r.get("likelihood_temperature") or r.get("posterior_temperature"))
        for r in sweep_rows
    }, key=lambda x: float(x) if x not in {"None", ""} else 0.0)
    clips = sorted({str(r.get("likelihood_clip") or "mc") for r in sweep_rows})
    grid = np.full((len(clips), len(temps)), np.nan, dtype=float)
    for r in sweep_rows:
        t = str(r.get("likelihood_temperature") or r.get("posterior_temperature"))
        c = str(r.get("likelihood_clip") or "mc")
        grid[clips.index(c), temps.index(t)] = float(r["brier"])
    fig, ax = plt.subplots(figsize=(max(6, len(temps) * 0.8), max(4, len(clips) * 0.5)))
    im = ax.imshow(grid, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(temps)), labels=temps, rotation=45, ha="right")
    ax.set_yticks(range(len(clips)), labels=clips)
    ax.set_xlabel("temperature")
    ax.set_ylabel("clip")
    ax.set_title("A8/A9 Brier Sweep")
    for y in range(len(clips)):
        for x in range(len(temps)):
            if not np.isnan(grid[y, x]):
                ax.text(x, y, f"{grid[y, x]:.3f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=ax, label="Brier")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_manifest(root: Path, out_path: Path) -> None:
    artifacts = []
    for suffix in ("*.json", "*.jsonl", "*.csv", "*.md", "*.png", "*.log", "*.out", "*.err"):
        artifacts.extend(sorted(root.rglob(suffix)))
    payload = {
        "root": str(root),
        "n_artifacts": len(artifacts),
        "artifacts": [str(p.relative_to(root)) for p in artifacts],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output-dir", type=Path, default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    root = args.root
    out = args.output_dir or root / "aggregate"
    out.mkdir(parents=True, exist_ok=True)

    rows = _summary_rows(root)
    int_rows = _intervention_rows(root)
    _write_csv(out / "ablation_results.csv", rows)
    _write_markdown(out / "ablation_results.md", rows)
    _write_csv(out / "ablation_interventions.csv", int_rows)
    _plot_brier_trajectories(root, out / "brier_trajectories.png")
    _plot_sweep_heatmap(rows, out / "a8_a9_brier_heatmap.png")
    _write_manifest(root, out / "artifact_manifest.json")
    print(json.dumps({
        "root": str(root),
        "output_dir": str(out),
        "n_summary_runs": len(rows),
        "n_intervention_runs": len(int_rows),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
