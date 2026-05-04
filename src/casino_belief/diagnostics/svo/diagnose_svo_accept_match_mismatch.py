#!/usr/bin/env python3
"""Diagnose accept-decision sensitivity for SVO match vs mismatch runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping


def _load_records(path: Path) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    rows: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (
                str(row.get("dialogue_id")),
                int(row.get("turn_index")),
                str(row.get("perspective")),
            )
            rows[key] = row
    return rows


def _accept_eligible(row: Mapping[str, Any]) -> bool:
    pred = row.get("pred") or {}
    true = row.get("true") or {}
    pending = row.get("pending_offer")
    return bool(
        true.get("accept") is not None
        and pred.get("accept") is not None
        and isinstance(pending, Mapping)
        and pending.get("to_perspective")
    )


def _summary(records: Mapping[tuple[str, int, str], Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records.values())
    eligible = [row for row in rows if _accept_eligible(row)]
    return {
        "records": len(rows),
        "accept_eligible": len(eligible),
        "accept_pred_true": sum(1 for row in eligible if bool((row.get("pred") or {}).get("accept"))),
        "accept_pred_false": sum(1 for row in eligible if not bool((row.get("pred") or {}).get("accept"))),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match-records", type=Path, required=True)
    ap.add_argument("--mismatch-records", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    match = _load_records(args.match_records)
    mismatch = _load_records(args.mismatch_records)
    joined_keys = sorted(set(match) & set(mismatch))

    rows: list[dict[str, Any]] = []
    for key in joined_keys:
        m = match[key]
        mm = mismatch[key]
        pred_m = m.get("pred") or {}
        pred_mm = mm.get("pred") or {}
        true_m = m.get("true") or {}
        eligible_m = _accept_eligible(m)
        eligible_mm = _accept_eligible(mm)
        row = {
            "dialogue_id": key[0],
            "turn_idx": key[1],
            "perspective": key[2],
            "accept_eligible_match": eligible_m,
            "accept_eligible_mismatch": eligible_mm,
            "accept_eligible": bool(eligible_m and eligible_mm),
            "lambda_match": pred_m.get("lambda"),
            "lambda_mismatch": pred_mm.get("lambda"),
            "accept_pred_match": pred_m.get("accept"),
            "accept_pred_mismatch": pred_mm.get("accept"),
            "accept_gold": true_m.get("accept"),
            "lambda_changed": pred_m.get("lambda") != pred_mm.get("lambda"),
            "accept_pred_changed": pred_m.get("accept") != pred_mm.get("accept"),
        }
        rows.append(row)

    eligible_rows = [row for row in rows if row["accept_eligible"]]
    lambda_changed = [row for row in eligible_rows if row["lambda_changed"]]
    pred_changed = [row for row in lambda_changed if row["accept_pred_changed"]]
    summary = {
        "match_records": str(args.match_records),
        "mismatch_records": str(args.mismatch_records),
        "match_support": _summary(match),
        "mismatch_support": _summary(mismatch),
        "joined_records": len(rows),
        "joined_accept_eligible": len(eligible_rows),
        "eligible_lambda_changed": len(lambda_changed),
        "eligible_lambda_changed_accept_pred_changed": len(pred_changed),
        "eligible_lambda_changed_accept_pred_unchanged": len(lambda_changed) - len(pred_changed),
        "accept_pred_changed_rows": pred_changed,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    csv_path = args.output_dir / "joined_accept_diagnostic.csv"
    jsonl_path = args.output_dir / "joined_accept_diagnostic.jsonl"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {summary_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {jsonl_path}")
    print(json.dumps({
        "joined_accept_eligible": summary["joined_accept_eligible"],
        "eligible_lambda_changed": summary["eligible_lambda_changed"],
        "eligible_lambda_changed_accept_pred_changed": summary[
            "eligible_lambda_changed_accept_pred_changed"
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
