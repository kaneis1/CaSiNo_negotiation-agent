#!/usr/bin/env python3
"""Verify ``--lambda-from-svo`` does not perturb non-menu plumbing.

The smoke filters to proself examples, where ``svo_to_lambda`` maps to the
same lambda as the constant ``--lambda 1`` run. With a dummy posterior model,
the two runs should produce identical posteriors, bids, accept decisions, and
true labels. Any mismatch means the flag leaked outside the intended menu
scoring path.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _dialogue_id(dialogue: Mapping[str, Any], fallback: int) -> Any:
    return dialogue.get("dialogue_id", fallback)


def _svo(dialogue: Mapping[str, Any], role: str) -> str:
    return str(
        ((dialogue.get("participant_info") or {})
         .get(role, {})
         .get("personality", {})
         .get("svo", ""))
    ).strip().lower()


def _write_proself_subset(
    *,
    data_path: Path,
    output_path: Path,
    role: str,
    n_dialogues: int,
) -> list[Any]:
    with data_path.open() as f:
        dialogues = json.load(f)
    selected = [
        dialogue
        for dialogue in dialogues
        if _svo(dialogue, role) in {"proself", "individualistic"}
    ][:n_dialogues]
    if len(selected) < n_dialogues:
        raise RuntimeError(
            f"needed {n_dialogues} proself dialogues for {role}, found {len(selected)}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    return [_dialogue_id(d, i) for i, d in enumerate(selected)]


def _run_eval(
    *,
    data_path: Path,
    output_dir: Path,
    annotations: Path,
    role: str,
    lambda_from_svo: bool,
    python_exe: str,
) -> None:
    cmd = [
        python_exe,
        "-m",
        "opponent_model.turn_eval_run",
        "--data",
        str(data_path),
        "--output-dir",
        str(output_dir),
        "--agent",
        "bayesian",
        "--dummy-llm",
        "--annotations",
        str(annotations),
        "--perspectives",
        role,
        "--lambda",
        "1.0",
        "--posterior-k",
        "2",
        "--posterior-temperature",
        "0.0",
        "--accept-margin",
        "5",
        "--accept-floor",
        "0.50",
        "--max-new-tokens",
        "32",
        "--temperature",
        "0.0",
    ]
    if lambda_from_svo:
        cmd.append("--lambda-from-svo")
    subprocess.run(cmd, check=True)


def _record_key(row: Mapping[str, Any]) -> tuple[Any, str, int]:
    return (row["dialogue_id"], row["perspective"], int(row["turn_index"]))


def _compare_records(
    constant_rows: Iterable[Mapping[str, Any]],
    svo_rows: Iterable[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    constant_by_key = {_record_key(row): row for row in constant_rows}
    svo_by_key = {_record_key(row): row for row in svo_rows}
    mismatches: list[Dict[str, Any]] = []
    all_keys = sorted(set(constant_by_key) | set(svo_by_key), key=lambda x: (str(x[0]), x[1], x[2]))
    for key in all_keys:
        left = constant_by_key.get(key)
        right = svo_by_key.get(key)
        if left is None or right is None:
            mismatches.append({"key": key, "field": "record_presence"})
            continue
        for field in ("posterior", "bid", "accept", "action", "lambda"):
            lval = (left.get("pred") or {}).get(field)
            rval = (right.get("pred") or {}).get(field)
            if lval != rval:
                mismatches.append({
                    "key": key,
                    "field": f"pred.{field}",
                    "constant": lval,
                    "lambda_from_svo": rval,
                })
        if left.get("true") != right.get("true"):
            mismatches.append({
                "key": key,
                "field": "true",
                "constant": left.get("true"),
                "lambda_from_svo": right.get("true"),
            })
    return mismatches


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/casino_test.json"))
    ap.add_argument("--annotations", type=Path, default=Path("CaSiNo/data/casino_ann.json"))
    ap.add_argument("--role", default="mturk_agent_1")
    ap.add_argument("--n-dialogues", type=int, default=3)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("opponent_model/results/day9_svo_integrity_smoke"),
    )
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subset_path = args.output_dir / "proself_subset.json"
    selected_ids = _write_proself_subset(
        data_path=args.data,
        output_path=subset_path,
        role=args.role,
        n_dialogues=args.n_dialogues,
    )

    constant_out = args.output_dir / "constant_lambda1"
    svo_out = args.output_dir / "lambda_from_svo"
    _run_eval(
        data_path=subset_path,
        output_dir=constant_out,
        annotations=args.annotations,
        role=args.role,
        lambda_from_svo=False,
        python_exe=args.python,
    )
    _run_eval(
        data_path=subset_path,
        output_dir=svo_out,
        annotations=args.annotations,
        role=args.role,
        lambda_from_svo=True,
        python_exe=args.python,
    )

    mismatches = _compare_records(
        _load_jsonl(constant_out / "turn_records.jsonl"),
        _load_jsonl(svo_out / "turn_records.jsonl"),
    )
    report = {
        "metadata": {
            "data_path": str(args.data),
            "subset_path": str(subset_path),
            "annotations": str(args.annotations),
            "role": args.role,
            "selected_dialogue_ids": selected_ids,
            "check": (
                "constant lambda=1 vs --lambda-from-svo on proself-only "
                "dialogues with dummy posterior"
            ),
        },
        "constant_summary": json.loads((constant_out / "turn_summary.json").read_text()),
        "lambda_from_svo_summary": json.loads((svo_out / "turn_summary.json").read_text()),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:50],
        "passed": len(mismatches) == 0,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    if mismatches:
        print(f"ERROR: found {len(mismatches)} mismatches")
        return 2
    print("SVO integrity smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
