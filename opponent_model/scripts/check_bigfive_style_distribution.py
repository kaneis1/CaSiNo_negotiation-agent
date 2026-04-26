#!/usr/bin/env python3
"""Report Big Five -> style bucket counts for the Day 9.2 gate.

The active rule is imported from ``sft_8b.bigfive_to_style``. The legacy rule
is kept here only to document the failed pre-retune distribution.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping

from sft_8b.bigfive_to_style import (
    COMPETITIVE_AGREEABLENESS_MAX,
    COMPETITIVE_EXTRAVERSION_MIN,
    COOPERATIVE_AGREEABLENESS_MIN,
    bigfive_to_style,
)


STYLE_ORDER = ("cooperative", "competitive", "balanced")


def _load_dialogues(path: Path) -> list[Mapping[str, Any]]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected list of dialogues in {path}")
    return data


def _traits(dialogue: Mapping[str, Any], role: str) -> Mapping[str, Any] | None:
    return (
        (dialogue.get("participant_info") or {})
        .get(role, {})
        .get("personality", {})
        .get("big-five")
    )


def legacy_bigfive_to_style(big_five: Mapping[str, float] | None) -> str:
    """The stale Day 9.1 rule that over-routed test examples to cooperative."""
    traits = {str(k).lower(): float(v) for k, v in (big_five or {}).items()}
    agree = traits.get("agreeableness", 4.0)
    extra = traits.get("extraversion", 4.0)
    if agree >= 5.0:
        return "cooperative"
    if agree <= 3.0 or extra >= 6.0:
        return "competitive"
    return "balanced"


def _counts(
    dialogues: list[Mapping[str, Any]],
    *,
    role: str,
    rule,
) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    missing = 0
    for dialogue in dialogues:
        big_five = _traits(dialogue, role)
        if big_five is None:
            missing += 1
            continue
        counts[rule(big_five)] += 1
    out = {style: int(counts.get(style, 0)) for style in STYLE_ORDER}
    if missing:
        out["missing_bigfive"] = missing
    return out


def _examples(
    dialogues: list[Mapping[str, Any]],
    *,
    role: str,
) -> Dict[str, Dict[str, Any]]:
    examples: Dict[str, Dict[str, Any]] = {}
    for dialogue in dialogues:
        big_five = _traits(dialogue, role)
        if big_five is None:
            continue
        style = bigfive_to_style(big_five)
        if style in examples:
            continue
        examples[style] = {
            "dialogue_id": dialogue.get("dialogue_id"),
            "role": role,
            "big_five": dict(big_five),
        }
    return examples


def _pass_gate(counts: Mapping[str, int], *, minimum: int) -> bool:
    return all(int(counts.get(style, 0)) >= minimum for style in STYLE_ORDER)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--train",
        type=Path,
        default=Path("data/casino_train_w0.2.json"),
        help="train metadata used to lock thresholds before looking at test",
    )
    ap.add_argument("--test", type=Path, default=Path("data/casino_test.json"))
    ap.add_argument("--role", default="mturk_agent_1")
    ap.add_argument("--min-test-bucket", type=int, default=25)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("opponent_model/results/day9_bigfive_distribution"),
    )
    args = ap.parse_args()

    train = _load_dialogues(args.train)
    test = _load_dialogues(args.test)

    report = {
        "metadata": {
            "role": args.role,
            "train_path": str(args.train),
            "test_path": str(args.test),
            "minimum_test_bucket_n": args.min_test_bucket,
            "active_rule": {
                "cooperative": (
                    f"agreeableness >= {COOPERATIVE_AGREEABLENESS_MIN:g}"
                ),
                "competitive": (
                    f"agreeableness <= {COMPETITIVE_AGREEABLENESS_MAX:g} "
                    f"or extraversion >= {COMPETITIVE_EXTRAVERSION_MIN:g}"
                ),
                "balanced": "otherwise",
                "thresholds_selected_on": str(args.train),
                "locked_before_test_report": True,
            },
            "legacy_rule": {
                "cooperative": "agreeableness >= 5",
                "competitive": "agreeableness <= 3 or extraversion >= 6",
                "balanced": "otherwise",
            },
        },
        "active": {
            "train_counts": _counts(train, role=args.role, rule=bigfive_to_style),
            "test_counts": _counts(test, role=args.role, rule=bigfive_to_style),
            "test_examples_by_style": _examples(test, role=args.role),
        },
        "legacy": {
            "train_counts": _counts(train, role=args.role, rule=legacy_bigfive_to_style),
            "test_counts": _counts(test, role=args.role, rule=legacy_bigfive_to_style),
        },
    }
    report["active"]["test_gate_passed"] = _pass_gate(
        report["active"]["test_counts"],
        minimum=args.min_test_bucket,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    md_path = args.output_dir / "report.md"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# Day 9.2 Big Five Distribution Gate",
                "",
                f"Role: `{args.role}`",
                f"Train file: `{args.train}`",
                f"Test file: `{args.test}`",
                "",
                "## Legacy Rule",
                f"Train counts: `{report['legacy']['train_counts']}`",
                f"Test counts: `{report['legacy']['test_counts']}`",
                "",
                "## Train-Locked Active Rule",
                f"Train counts: `{report['active']['train_counts']}`",
                f"Test counts: `{report['active']['test_counts']}`",
                f"Gate passed: `{report['active']['test_gate_passed']}`",
                "",
                "The active thresholds are selected on train metadata only and "
                "then frozen before reporting the test distribution.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote {summary_path}")
    print(f"Wrote {md_path}")
    print("legacy test counts:", report["legacy"]["test_counts"])
    print("active train counts:", report["active"]["train_counts"])
    print("active test counts:", report["active"]["test_counts"])
    if not report["active"]["test_gate_passed"]:
        print("ERROR: at least one active test bucket is below the minimum.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
