"""Aggregate DND transfer summaries into compact tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

DEFAULT_ROOT = Path("opponent_model/results/dnd_transfer")


def _num(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _fmt(value: Any, digits: int = 4) -> str:
    v = _num(value)
    return "" if v is None else f"{v:.{digits}f}"


def _rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("turn_summary.json")):
        rel_parent = path.parent.relative_to(root)
        if any(str(part).startswith("smoke") for part in rel_parent.parts):
            continue
        try:
            obj = json.loads(path.read_text())
        except Exception:
            continue
        cfg = obj.get("config") or {}
        metrics = obj.get("metrics") or {}
        overall = metrics.get("overall") or {}
        kp13 = metrics.get("kpenalty_1_3") or {}
        kp15 = metrics.get("kpenalty_1_5") or {}
        rows.append({
            "run": str(rel_parent),
            "provider": cfg.get("provider"),
            "name_mode": cfg.get("name_mode"),
            "value_mode": cfg.get("value_mode"),
            "adapter": cfg.get("adapter"),
            "n_records_eval": obj.get("n_records_eval"),
            "n_snapshots": obj.get("n_snapshots"),
            "brier": overall.get("brier"),
            "ema_at2": metrics.get("ema_at2"),
            "ema_kpenalty_1_3": kp13.get("ema"),
            "top1_kpenalty_1_3": kp13.get("top1"),
            "ndcg_kpenalty_1_3": kp13.get("ndcg"),
            "ema_kpenalty_1_5": kp15.get("ema"),
            "ndcg_kpenalty_1_5": kp15.get("ndcg"),
            "bid_cosine": overall.get("bid_cosine"),
            "utility_norm": overall.get("utility_norm"),
            "parse_errors": (obj.get("provider_summary") or {}).get("parse_errors"),
            "support_by_k": json.dumps(metrics.get("support_by_k") or {}, sort_keys=True),
        })
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    headers = [
        "run", "Brier", "EMA@2", "EMA k1-3", "NDCG k1-3",
        "Bid cos", "Util", "n",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        vals = [
            str(row.get("run", "")),
            _fmt(row.get("brier")),
            _fmt(row.get("ema_at2")),
            _fmt(row.get("ema_kpenalty_1_3")),
            _fmt(row.get("ndcg_kpenalty_1_3")),
            _fmt(row.get("bid_cosine")),
            _fmt(row.get("utility_norm")),
            str(row.get("n_snapshots") or ""),
        ]
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate(root: Path, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    out = output_dir or root / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    rows = _rows(root)
    _write_csv(out / "dnd_transfer_results.csv", rows)
    _write_md(out / "dnd_transfer_results.md", rows)
    manifest = {
        "root": str(root),
        "output_dir": str(out),
        "n_runs": len(rows),
        "runs": [r["run"] for r in rows],
    }
    (out / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output-dir", type=Path, default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    aggregate(args.root, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
