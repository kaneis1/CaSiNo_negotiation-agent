"""Aggregate per-fold summary.json files into one CV report.

Reads ``sft_8b/results/cv/fold_{k}/sft_eval/summary.json`` for k in 0..K-1,
prints a side-by-side table + mean ± std across folds, and writes
``sft_8b/results/cv/cv_summary.json`` with the same numbers.

Per-prediction OOF support: also concatenates the per-fold
``predictions.jsonl`` into ``cv_oof_predictions.jsonl`` so you have one
held-out prediction per dialogue across the full corpus (1030 rows-ish).

Usage
-----
    python -m sft_8b.cv_aggregate                     # default cv_root
    python -m sft_8b.cv_aggregate --cv-root sft_8b/results/cv
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_CV_ROOT = Path("sft_8b/results/cv")

# What we report. Each is a (top-level summary key, label) tuple.
_PREF_METRICS = [
    ("ema_kpenalty",  "EMA"),
    ("top1_kpenalty", "Top1"),
    ("ndcg_kpenalty", "NDCG"),
]
_SAT_METRICS = [
    ("sat_acc_kpenalty", "sat_acc"),
    ("sat_mae_kpenalty", "sat_mae"),
]


def _load_summary(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _flat_summary(s: Dict[str, Any]) -> Dict[str, float]:
    flat = dict(s["summary"])
    flat.update({f"sat_{k}": v for k, v in s["satisfaction"]["summary"].items()})
    return flat


def _ms(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0, "n": 1}
    return {
        "mean": statistics.mean(values),
        "std":  statistics.stdev(values),
        "n":    len(values),
    }


# ── Main ───────────────────────────────────────────────────────────────────


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cv-root",  type=Path, default=DEFAULT_CV_ROOT)
    p.add_argument("--n-splits", type=int,  default=None,
                   help="Override fold count (default = inferred from manifest).")
    p.add_argument("--no-oof",   action="store_true",
                   help="Skip writing the merged OOF predictions file.")
    args = p.parse_args(argv)

    manifest_path = args.cv_root / "cv_split_manifest.json"
    n_splits = args.n_splits
    if n_splits is None:
        if not manifest_path.exists():
            print(f"ERROR: --n-splits not given and no manifest at "
                  f"{manifest_path}", file=sys.stderr)
            return 2
        with manifest_path.open() as f:
            n_splits = json.load(f)["n_splits"]

    # ── Load each fold's summary.json ─────────────────────────────────────
    rows: List[Dict[str, Any]] = []
    missing: List[int] = []
    for k in range(n_splits):
        s = _load_summary(args.cv_root / f"fold_{k}" / "sft_eval" / "summary.json")
        if s is None:
            missing.append(k)
            continue
        flat = _flat_summary(s)
        rows.append({
            "fold":          k,
            "n_dialogues":   s["n_dialogues"],
            "n_predictions": s["n_predictions"],
            "elapsed_s":     s.get("elapsed_seconds"),
            **flat,
        })

    if missing:
        print(f"WARNING: missing summary.json for fold(s): {missing}",
              file=sys.stderr)
    if not rows:
        print("ERROR: no fold summaries found.", file=sys.stderr)
        return 2

    # ── Per-fold table ────────────────────────────────────────────────────
    cols = (["fold", "n_dialogues", "n_predictions"]
            + [k for k, _ in _PREF_METRICS]
            + [f"sat_{k}" for k, _ in _SAT_METRICS])

    print("\n=== Per-fold metrics ===")
    header = (f"{'fold':<5} {'n_dlg':>6} {'n_pred':>7} "
              f"{'EMA':>7} {'Top1':>7} {'NDCG':>7} "
              f"{'sat_acc':>8} {'sat_mae':>8}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['fold']:<5d} {r['n_dialogues']:>6d} {r['n_predictions']:>7d} "
              f"{r['ema_kpenalty']:>7.3f} {r['top1_kpenalty']:>7.3f} "
              f"{r['ndcg_kpenalty']:>7.3f} "
              f"{r['sat_sat_acc_kpenalty']:>8.3f} "
              f"{r['sat_sat_mae_kpenalty']:>8.3f}")

    # ── Mean ± std across folds ───────────────────────────────────────────
    print("\n=== CV mean ± std ===")
    agg: Dict[str, Dict[str, float]] = {}
    for key, label in _PREF_METRICS:
        agg[key] = _ms([r[key] for r in rows])
        print(f"  {label:8s}: {agg[key]['mean']:.3f} ± {agg[key]['std']:.3f}  "
              f"(n={agg[key]['n']})")
    for key, label in _SAT_METRICS:
        full_key = f"sat_{key}"
        agg[full_key] = _ms([r[full_key] for r in rows])
        print(f"  {label:8s}: {agg[full_key]['mean']:.3f} ± "
              f"{agg[full_key]['std']:.3f}  (n={agg[full_key]['n']})")

    # ── Pooled (concatenated OOF) predictions ─────────────────────────────
    pooled_n = 0
    pooled_path: Optional[Path] = None
    if not args.no_oof:
        pooled_path = args.cv_root / "cv_oof_predictions.jsonl"
        with pooled_path.open("w") as out:
            for k in range(n_splits):
                src = args.cv_root / f"fold_{k}" / "sft_eval" / "predictions.jsonl"
                if not src.exists():
                    continue
                with src.open() as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        rec["fold"] = k
                        out.write(json.dumps(rec) + "\n")
                        pooled_n += 1
        print(f"\nPooled OOF predictions: {pooled_n} rows -> {pooled_path}")

    # ── Persist machine-readable report ───────────────────────────────────
    out = {
        "cv_root":       str(args.cv_root),
        "n_splits":      n_splits,
        "n_folds_done":  len(rows),
        "missing_folds": missing,
        "per_fold":      rows,
        "agg":           agg,
        "pooled_oof_predictions": str(pooled_path) if pooled_path else None,
        "pooled_oof_n":  pooled_n,
    }
    cv_summary_path = args.cv_root / "cv_summary.json"
    with cv_summary_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {cv_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
