#!/usr/bin/env python3
"""Build quality-ranked subdatasets from CaSiNo training dialogues.

Implements the Q_w(d) quality function with Pareto normalization,
opponent satisfaction, and Pareto floor penalty.

Each dialogue is evaluated from both speaker perspectives, yielding
~2x trajectories. Top-30% are selected for each style weight w.

Outputs:
  - data/quality_audit.csv          (full scoring audit trail)
  - data/casino_train_w0.2.json     (cooperative style)
  - data/casino_train_w0.5.json     (balanced style)
  - data/casino_train_w0.8.json     (competitive style)
"""

from __future__ import annotations

import csv
import json
import statistics
from itertools import product
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "casino_train.json"
OUTPUT_DIR = SCRIPT_DIR

POINT_MAP = {"High": 5, "Medium": 4, "Low": 3}

SAT_TO_LIKERT = {
    "Extremely dissatisfied": 1,
    "Slightly dissatisfied": 2,
    "Undecided": 3,
    "Slightly satisfied": 4,
    "Extremely satisfied": 5,
}

W_VALUES = [0.2, 0.5, 0.8]
TAU = 0.3
LAM = 0.1
OPP_FLOOR = 15
TOP_FRACTION = 0.30


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def compute_points(deal: dict[str, int], value2issue: dict[str, str]) -> int:
    """Compute points given a deal {issue: count} and a priority map {priority: issue}."""
    issue2priority = {v: k for k, v in value2issue.items()}
    total = 0
    for issue, count in deal.items():
        priority = issue2priority[issue]
        total += count * POINT_MAP[priority]
    return total


def pareto_max_self(
    priorities_self: dict[str, str],
    priorities_opp: dict[str, str],
    opp_floor: int = OPP_FLOOR,
) -> int:
    """
    Enumerate all 64 possible splits (4^3 for 3 issues with 0-3 items each),
    return max self_pts where opp_pts >= opp_floor.
    """
    best = 0
    for food_self, water_self, firewood_self in product(range(4), repeat=3):
        self_deal = {"Food": food_self, "Water": water_self, "Firewood": firewood_self}
        opp_deal = {"Food": 3 - food_self, "Water": 3 - water_self, "Firewood": 3 - firewood_self}
        s_self = compute_points(self_deal, priorities_self)
        s_opp = compute_points(opp_deal, priorities_opp)
        if s_opp >= opp_floor and s_self > best:
            best = s_self
    return best if best > 0 else 1


def normalize_satisfaction(sat_likert: int) -> float:
    return (sat_likert - 1) / 4


def pareto_floor_penalty(
    s_opp: int,
    pareto_max_opp: int,
    tau: float = TAU,
    lam: float = LAM,
) -> float:
    s_opp_pareto = s_opp / pareto_max_opp
    shortfall = max(0.0, tau - s_opp_pareto)
    return lam * (shortfall ** 2)


def quality_score(
    s_self: int,
    s_opp: int,
    sat_opp_likert: int,
    priorities_self: dict[str, str],
    priorities_opp: dict[str, str],
    w: float,
    tau: float = TAU,
    lam: float = LAM,
    opp_floor: int = OPP_FLOOR,
) -> dict[str, Any]:
    """Compute Q_w and all intermediate values."""
    p_self = pareto_max_self(priorities_self, priorities_opp, opp_floor)
    p_opp = pareto_max_self(priorities_opp, priorities_self, opp_floor)

    s_self_norm = s_self / p_self
    s_opp_norm = s_opp / p_opp
    sat_norm = normalize_satisfaction(sat_opp_likert)
    penalty = pareto_floor_penalty(s_opp, p_opp, tau, lam)

    q = w * s_self_norm + (1 - w) * sat_norm - penalty

    return {
        "pareto_self": p_self,
        "pareto_opp": p_opp,
        "s_self_norm": s_self_norm,
        "s_opp_norm": s_opp_norm,
        "sat_norm": sat_norm,
        "penalty": penalty,
        "Q": q,
    }


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_deal(dialogue: dict) -> dict[str, dict[str, int]] | None:
    """Extract the accepted deal from chat_logs. Returns {proposer, acceptor} deals."""
    proposer_id = None
    proposer_deal = None
    acceptor_deal = None

    for log in dialogue["chat_logs"]:
        td = log.get("task_data", {})
        if not isinstance(td, dict):
            continue
        if "issue2youget" in td:
            proposer_id = log["id"]
            proposer_deal = {k: int(v) for k, v in td["issue2youget"].items()}
            acceptor_deal = {k: int(v) for k, v in td["issue2theyget"].items()}

    if proposer_deal is None:
        return None

    acceptor_id = "mturk_agent_2" if proposer_id == "mturk_agent_1" else "mturk_agent_1"
    return {
        proposer_id: proposer_deal,
        acceptor_id: acceptor_deal,
    }


def get_outcome(dialogue: dict) -> str:
    last = dialogue["chat_logs"][-1]
    td = last.get("task_data", {})
    if isinstance(td, dict):
        return td.get("data", "unknown")
    return "unknown"


def count_turns(dialogue: dict) -> int:
    """Count substantive conversation turns (exclude deal/accept/reject/walkaway actions)."""
    action_texts = {"Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away"}
    return sum(1 for log in dialogue["chat_logs"] if log.get("text") not in action_texts)


def compute_integrative_potential(
    priorities_self: dict[str, str],
    priorities_opp: dict[str, str],
) -> str:
    """Classify integrative potential based on priority overlap."""
    issues = ["Food", "Water", "Firewood"]
    issue2prio_self = {v: k for k, v in priorities_self.items()}
    issue2prio_opp = {v: k for k, v in priorities_opp.items()}

    same_high = 0
    for issue in issues:
        if issue2prio_self.get(issue) == issue2prio_opp.get(issue) == "High":
            same_high += 1

    if same_high >= 1:
        return "low"
    p_self = set(priorities_self.items())
    p_opp = set(priorities_opp.items())
    if p_self == p_opp:
        return "low"
    return "high"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_trajectories(dialogues: list[dict]) -> list[dict]:
    """Build dual-perspective trajectories from dialogues."""
    trajectories = []

    for dialogue in dialogues:
        outcome = get_outcome(dialogue)
        if outcome == "walk_away":
            continue

        deal = extract_deal(dialogue)
        if deal is None:
            continue

        n_turns = count_turns(dialogue)
        if n_turns < 6:
            continue

        for self_role in ["mturk_agent_1", "mturk_agent_2"]:
            opp_role = "mturk_agent_2" if self_role == "mturk_agent_1" else "mturk_agent_1"

            pinfo_self = dialogue["participant_info"][self_role]
            pinfo_opp = dialogue["participant_info"][opp_role]

            sat_opp_str = pinfo_opp["outcomes"]["satisfaction"]
            if sat_opp_str not in SAT_TO_LIKERT:
                continue

            s_self = pinfo_self["outcomes"]["points_scored"]
            s_opp = pinfo_opp["outcomes"]["points_scored"]
            sat_opp_likert = SAT_TO_LIKERT[sat_opp_str]

            priorities_self = pinfo_self["value2issue"]
            priorities_opp = pinfo_opp["value2issue"]

            integ = compute_integrative_potential(priorities_self, priorities_opp)

            scores = {}
            for w in W_VALUES:
                result = quality_score(
                    s_self, s_opp, sat_opp_likert,
                    priorities_self, priorities_opp, w,
                )
                scores[w] = result

            ref = scores[W_VALUES[0]]
            trajectories.append({
                "dialogue_id": dialogue["dialogue_id"],
                "speaker_role": self_role,
                "s_self": s_self,
                "s_opp": s_opp,
                "sat_opp_likert": sat_opp_likert,
                "pareto_self": ref["pareto_self"],
                "pareto_opp": ref["pareto_opp"],
                "s_self_norm": ref["s_self_norm"],
                "s_opp_norm": ref["s_opp_norm"],
                "sat_norm": ref["sat_norm"],
                "penalty": ref["penalty"],
                "integrative_potential": integ,
                "n_turns": n_turns,
                **{f"Q_{w}": scores[w]["Q"] for w in W_VALUES},
                "dialogue": dialogue,
            })

    return trajectories


def select_top_fraction(
    trajectories: list[dict],
    w: float,
    fraction: float = TOP_FRACTION,
) -> list[dict]:
    key = f"Q_{w}"
    scored = sorted(trajectories, key=lambda t: -t[key])
    n_keep = int(fraction * len(scored))
    return scored[:n_keep]


def write_csv(trajectories: list[dict], path: Path) -> None:
    fieldnames = [
        "dialogue_id", "speaker_role",
        "s_self", "s_opp", "sat_opp_likert",
        "pareto_self", "pareto_opp",
        "s_self_norm", "s_opp_norm", "sat_norm", "penalty",
        "integrative_potential", "n_turns",
    ]
    for w in W_VALUES:
        fieldnames.append(f"Q_{w}")
    for w in W_VALUES:
        fieldnames.append(f"selected_{w}")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trajectories:
            row = {k: t.get(k) for k in fieldnames}
            writer.writerow(row)


def write_subdataset(
    selected: list[dict],
    all_dialogues_map: dict[int, dict],
    path: Path,
) -> None:
    """Write selected trajectories as a JSON list of augmented dialogues."""
    output = []
    for t in selected:
        entry = dict(t["dialogue"])
        entry["_quality_meta"] = {
            "speaker_role": t["speaker_role"],
            "s_self_norm": round(t["s_self_norm"], 4),
            "s_opp_norm": round(t["s_opp_norm"], 4),
            "sat_norm": round(t["sat_norm"], 4),
            "penalty": round(t["penalty"], 6),
            "Q": round(t[f"Q_{path.stem.split('w')[-1]}".replace("casino_train_", "")], 4),
        }
        output.append(entry)

    with path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def main() -> None:
    print(f"Loading {INPUT_PATH} ...")
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        dialogues = json.load(f)
    print(f"  Total dialogues: {len(dialogues)}")

    trajectories = build_trajectories(dialogues)
    print(f"  Usable trajectories (dual-perspective): {len(trajectories)}")

    # Mark selection flags
    for w in W_VALUES:
        selected = select_top_fraction(trajectories, w)
        selected_keys = {(t["dialogue_id"], t["speaker_role"]) for t in selected}
        for t in trajectories:
            t[f"selected_{w}"] = (t["dialogue_id"], t["speaker_role"]) in selected_keys

    # Write CSV audit trail
    csv_path = OUTPUT_DIR / "quality_audit.csv"
    write_csv(trajectories, csv_path)
    print(f"\nWrote audit CSV: {csv_path}")
    print(f"  Rows: {len(trajectories)}")

    # Write subdatasets
    dialogue_map = {d["dialogue_id"]: d for d in dialogues}
    for w in W_VALUES:
        selected = select_top_fraction(trajectories, w)
        out_path = OUTPUT_DIR / f"casino_train_w{w}.json"

        output = []
        for t in selected:
            entry = dict(t["dialogue"])
            entry["_quality_meta"] = {
                "speaker_role": t["speaker_role"],
                "Q": round(t[f"Q_{w}"], 4),
                "s_self_norm": round(t["s_self_norm"], 4),
                "s_opp_norm": round(t["s_opp_norm"], 4),
                "sat_norm": round(t["sat_norm"], 4),
                "penalty": round(t["penalty"], 6),
            }
            output.append(entry)

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"  Wrote {out_path} ({len(selected)} trajectories)")

    # -----------------------------------------------------------------------
    # Validation summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    for w in W_VALUES:
        selected = [t for t in trajectories if t[f"selected_{w}"]]
        n = len(selected)
        mean_s_self_norm = statistics.mean(t["s_self_norm"] for t in selected)
        mean_s_opp_norm = statistics.mean(t["s_opp_norm"] for t in selected)
        mean_sat_opp = statistics.mean(t["sat_opp_likert"] for t in selected)
        mean_sat_norm = statistics.mean(t["sat_norm"] for t in selected)
        mean_penalty = statistics.mean(t["penalty"] for t in selected)
        mean_q = statistics.mean(t[f"Q_{w}"] for t in selected)

        high_integ = sum(1 for t in selected if t["integrative_potential"] == "high")
        low_integ = sum(1 for t in selected if t["integrative_potential"] == "low")

        unique_dialogues = len({t["dialogue_id"] for t in selected})

        print(f"\nw={w}: n={n} trajectories from {unique_dialogues} unique dialogues")
        print(f"  mean Q_{w}:           {mean_q:.4f}")
        print(f"  mean s_self_pareto:   {mean_s_self_norm:.4f}")
        print(f"  mean s_opp_pareto:    {mean_s_opp_norm:.4f}")
        print(f"  mean sat_opp(Likert): {mean_sat_opp:.3f}")
        print(f"  mean sat_norm:        {mean_sat_norm:.4f}")
        print(f"  mean penalty:         {mean_penalty:.6f}")
        print(f"  integrative high/low: {high_integ}/{low_integ}")

    # Global checks
    print(f"\n--- Monotonicity checks ---")
    means_self = []
    means_sat = []
    for w in W_VALUES:
        selected = [t for t in trajectories if t[f"selected_{w}"]]
        means_self.append(statistics.mean(t["s_self_norm"] for t in selected))
        means_sat.append(statistics.mean(t["sat_norm"] for t in selected))

    self_mono = means_self[0] <= means_self[1] <= means_self[2]
    sat_mono = means_sat[0] >= means_sat[1] >= means_sat[2]
    print(f"  s_self_norm increases with w?  {means_self}  -> {'PASS' if self_mono else 'FAIL'}")
    print(f"  sat_norm decreases with w?     {means_sat}  -> {'PASS' if sat_mono else 'FAIL'}")

    # Integrative potential distribution check
    print(f"\n--- Integrative potential distribution ---")
    all_high = sum(1 for t in trajectories if t["integrative_potential"] == "high")
    all_low = sum(1 for t in trajectories if t["integrative_potential"] == "low")
    print(f"  Overall: high={all_high}, low={all_low} ({all_high/(all_high+all_low)*100:.1f}% high)")
    for w in W_VALUES:
        selected = [t for t in trajectories if t[f"selected_{w}"]]
        h = sum(1 for t in selected if t["integrative_potential"] == "high")
        lo = sum(1 for t in selected if t["integrative_potential"] == "low")
        total = h + lo
        print(f"  w={w}: high={h}, low={lo} ({h/total*100:.1f}% high)")


if __name__ == "__main__":
    main()
