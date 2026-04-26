#!/usr/bin/env python3
"""Human SVO sanity check on CaSiNo outcome points.

For each participant row, group by the human's SVO label and report their
self-points plus the dialogue's joint-points. The primary Day 10 readout is
whether prosocial humans actually achieve higher joint-points than proself
humans on the same held-out split used for Protocol-3 eval.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, Mapping


ROLES = ("mturk_agent_1", "mturk_agent_2")
SVO_ORDER = ("proself", "prosocial", "unclassified")


def _load_dialogues(path: Path) -> list[Mapping[str, Any]]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected a list of dialogues in {path}")
    return data


def _points(dialogue: Mapping[str, Any], role: str) -> float:
    return float(
        ((dialogue.get("participant_info") or {})
         .get(role, {})
         .get("outcomes", {})
         .get("points_scored"))
    )


def _svo(dialogue: Mapping[str, Any], role: str) -> str:
    return str(
        ((dialogue.get("participant_info") or {})
         .get(role, {})
         .get("personality", {})
         .get("svo", "unclassified"))
    ).strip().lower() or "unclassified"


def _summ(vals: Iterable[float]) -> Dict[str, Any]:
    arr = list(vals)
    return {
        "n": len(arr),
        "mean": float(mean(arr)) if arr else float("nan"),
        "std": float(stdev(arr)) if len(arr) > 1 else float("nan"),
        "min": float(min(arr)) if arr else float("nan"),
        "max": float(max(arr)) if arr else float("nan"),
    }


def _welch(proself: list[float], prosocial: list[float]) -> Dict[str, Any]:
    out = {
        "mean_diff_prosocial_minus_proself": (
            float(mean(prosocial) - mean(proself))
            if proself and prosocial else float("nan")
        ),
        "scipy_available": False,
    }
    try:
        from scipy import stats  # type: ignore
    except Exception as exc:  # pragma: no cover - env-dependent
        out["scipy_error"] = repr(exc)
        return out
    out["scipy_available"] = True
    if len(proself) >= 2 and len(prosocial) >= 2:
        res = stats.ttest_ind(prosocial, proself, equal_var=False)
        out["welch_t"] = float(res.statistic)
        out["welch_p"] = float(res.pvalue)
    return out


def _analyze(
    dialogues: list[Mapping[str, Any]],
    *,
    roles: tuple[str, ...],
) -> Dict[str, Any]:
    groups: Dict[str, Dict[str, list[float]]] = defaultdict(
        lambda: {"self_points": [], "joint_points": []}
    )
    for dialogue in dialogues:
        joint = sum(_points(dialogue, role) for role in ROLES)
        for role in roles:
            svo = _svo(dialogue, role)
            groups[svo]["self_points"].append(_points(dialogue, role))
            groups[svo]["joint_points"].append(joint)

    summary = {
        svo: {
            "self_points": _summ(groups[svo]["self_points"]),
            "joint_points": _summ(groups[svo]["joint_points"]),
        }
        for svo in SVO_ORDER
    }
    summary["tests"] = {
        metric: _welch(groups["proself"][metric], groups["prosocial"][metric])
        for metric in ("self_points", "joint_points")
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/casino_test.json"))
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("opponent_model/results/day10_human_svo_baseline"),
    )
    args = ap.parse_args()

    dialogues = _load_dialogues(args.data)
    report = {
        "metadata": {
            "data": str(args.data),
            "n_dialogues": len(dialogues),
            "interpretation": (
                "If prosocial humans do not exceed proself humans on "
                "joint-points, then an SVO-conditioned agent that also lowers "
                "joint-points is consistent with behavioral fidelity rather "
                "than necessarily indicating planner failure."
            ),
        },
        "all_participants": _analyze(dialogues, roles=ROLES),
        "mturk_agent_1": _analyze(dialogues, roles=("mturk_agent_1",)),
    }
    primary = report["mturk_agent_1"]["tests"]["joint_points"]
    diff = primary["mean_diff_prosocial_minus_proself"]
    p_value = primary.get("welch_p")
    significant_positive = (
        isinstance(diff, float)
        and math.isfinite(diff)
        and diff > 0
        and isinstance(p_value, (float, int))
        and math.isfinite(float(p_value))
        and float(p_value) < 0.05
    )
    positive_inconclusive = (
        isinstance(diff, float)
        and math.isfinite(diff)
        and diff > 0
        and not significant_positive
    )
    report["decision"] = {
        "primary_comparison": "mturk_agent_1 joint-points, prosocial - proself",
        "mean_diff": diff,
        "welch_p": p_value,
        "human_prosocials_significantly_beat_proselfs_on_joint_points": significant_positive,
        "claim2_reframe": (
            "lambda_retuning_candidate"
            if significant_positive
            else "human_baseline_inconclusive"
            if positive_inconclusive
            else "behavioral_fidelity"
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    md_path = args.output_dir / "report.md"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    mt1 = report["mturk_agent_1"]
    allp = report["all_participants"]
    md_path.write_text(
        "\n".join(
            [
                "# Day 10 Human SVO Baseline",
                "",
                f"Data: `{args.data}`",
                "",
                "## mturk_agent_1",
                f"Proself joint mean: `{mt1['proself']['joint_points']['mean']:.3f}` (`n={mt1['proself']['joint_points']['n']}`)",
                f"Prosocial joint mean: `{mt1['prosocial']['joint_points']['mean']:.3f}` (`n={mt1['prosocial']['joint_points']['n']}`)",
                f"Prosocial - proself: `{mt1['tests']['joint_points']['mean_diff_prosocial_minus_proself']:.3f}`, Welch p=`{mt1['tests']['joint_points'].get('welch_p'):.3g}`",
                "",
                "## All Participants",
                f"Proself joint mean: `{allp['proself']['joint_points']['mean']:.3f}` (`n={allp['proself']['joint_points']['n']}`)",
                f"Prosocial joint mean: `{allp['prosocial']['joint_points']['mean']:.3f}` (`n={allp['prosocial']['joint_points']['n']}`)",
                f"Prosocial - proself: `{allp['tests']['joint_points']['mean_diff_prosocial_minus_proself']:.3f}`, Welch p=`{allp['tests']['joint_points'].get('welch_p'):.3g}`",
                "",
                f"Decision: `{report['decision']['claim2_reframe']}`.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote {summary_path}")
    print(f"Wrote {md_path}")
    print(json.dumps(report["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
