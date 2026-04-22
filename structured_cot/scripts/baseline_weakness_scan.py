"""Scan structured_cot/results/protocol1_70b_full for baseline failure modes.

Quantifies and samples four kinds of weakness in the Abdelnabi-style
Structured-CoT 70B baseline:

  (A) Misses opponent priorities   — <opponent_inference> claims the wrong
                                     top priority for the opponent.
  (B) Accepts dominated offers     — action=accept, but there exists an
                                     in-menu split with strictly higher
                                     U_self than the pending offer.
  (C) Proposes dominated offers    — action=reject with counter_offer π;
                                     some π' weakly dominates π on
                                     (U_self, U_opp_true) or is strictly
                                     better for the agent with the true
                                     opponent still happy.
  (D) No belief exposure           — the agent emits free-text hedges
                                     ("could be X or Y"), never a
                                     probability over the 6 orderings.
                                     This is a hard-coded fact about the
                                     prompt, so we just count how many
                                     <opponent_inference> blocks hedge.

Usage:
    python -m structured_cot.scripts.baseline_weakness_scan \\
        --turns structured_cot/results/protocol1_70b_full/turns.jsonl \\
        --out   structured_cot/results/protocol1_70b_full/baseline_weakness_scan.json
"""
from __future__ import annotations

import argparse
import json
import re
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

ITEMS = ("Food", "Water", "Firewood")
PRIORITY_POINTS = {"High": 5, "Medium": 4, "Low": 3}


# ── Utility helpers ────────────────────────────────────────────────────────
def priorities_to_points(priorities: Mapping[str, str]) -> Dict[str, int]:
    """{'High': 'Food', ...} → {'Food': 5, ...}."""
    return {priorities[rank]: PRIORITY_POINTS[rank] for rank in ("High", "Medium", "Low")}


def u(counts: Mapping[str, int], pts: Mapping[str, int]) -> int:
    return sum(int(counts.get(i, 0)) * int(pts.get(i, 0)) for i in ITEMS)


def all_splits() -> List[Dict[str, Dict[str, int]]]:
    out = []
    for f, w, fw in product(range(4), repeat=3):
        out.append({
            "self": {"Food": f, "Water": w, "Firewood": fw},
            "opp":  {"Food": 3 - f, "Water": 3 - w, "Firewood": 3 - fw},
        })
    return out


# ── Parsing the baseline's free-text priority claims ───────────────────────
ITEM_ALIASES = {
    "food": "Food", "foods": "Food", "fruit": "Food",
    "water": "Water", "waters": "Water", "drink": "Water", "drinks": "Water",
    "firewood": "Firewood", "wood": "Firewood", "fire": "Firewood",
}

# "Water > Food > Firewood" or "Water then Food then Firewood" or
# "prioritizes water ... then food ..."
RE_CHAIN = re.compile(
    r"\b(food|foods|water|waters|drinks?|firewood|wood|fruit)\b\s*"
    r"(?:>|>=|then|->|followed by|,\s*then|,\s*followed by|,|and then)\s*"
    r"\b(food|foods|water|waters|drinks?|firewood|wood|fruit)\b\s*"
    r"(?:>|>=|then|->|followed by|,\s*then|,\s*followed by|,|and then)\s*"
    r"\b(food|foods|water|waters|drinks?|firewood|wood|fruit)\b",
    re.IGNORECASE,
)

# "prioritizes Water most" / "top priority is Water" / "most likely ordering has water first"
RE_TOP = re.compile(
    r"(?:top priority|highest priority|most important|prioritizes?|"
    r"values most|most valued|most likely (?:top )?priority|"
    r"primarily values|mainly values|puts .* first)\s+(?:is|are|on|for)?\s*"
    r"\b(food|foods|water|waters|drinks?|firewood|wood|fruit)\b",
    re.IGNORECASE,
)

RE_HEDGE = re.compile(
    r"\b(uncertain|unsure|unclear|not sure|no clear evidence|no evidence|"
    r"could be|might be|may be|either .* or|two plausible|plausible orderings|"
    r"ambiguous|hard to tell|no strong signal)\b",
    re.IGNORECASE,
)


def claimed_top(opponent_inference: str) -> Tuple[Optional[str], Optional[List[str]], bool]:
    """Return (top_item, full_chain_or_None, hedged).

    Greedy: prefer a full 3-item chain when visible, else use a "prioritizes X"
    cue. Mark ``hedged=True`` when the block uses uncertainty language even if
    it still commits to a top item.
    """
    if not opponent_inference:
        return None, None, False
    hedged = bool(RE_HEDGE.search(opponent_inference))

    m = RE_CHAIN.search(opponent_inference)
    if m:
        chain = [ITEM_ALIASES[g.lower()] for g in m.groups()]
        if len(set(chain)) == 3:
            return chain[0], chain, hedged

    m = RE_TOP.search(opponent_inference)
    if m:
        return ITEM_ALIASES[m.group(1).lower()], None, hedged

    return None, None, hedged


# ── Pareto / dominance analysis on (U_self, U_opp_true) ────────────────────
def pareto_max_self_given_floor(pts_self: Mapping[str, int],
                                pts_opp:  Mapping[str, int],
                                opp_floor: int = 15) -> int:
    best = 0
    for split in all_splits():
        s = u(split["self"], pts_self)
        o = u(split["opp"],  pts_opp)
        if o >= opp_floor and s > best:
            best = s
    return best


def weakly_dominates(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    """a dominates b iff a ≥ b on both coords and strictly > on at least one."""
    return (a[0] >= b[0] and a[1] >= b[1]) and (a[0] > b[0] or a[1] > b[1])


def counter_offer_dominated(counter: Mapping[str, int],
                            pts_self: Mapping[str, int],
                            pts_opp:  Mapping[str, int]) -> Optional[Dict]:
    """Return a witness dominating split, or None if ``counter`` is Pareto-efficient."""
    c_s = u(counter, pts_self)
    c_o = u(counter, pts_opp)
    for split in all_splits():
        s = u(split["self"], pts_self)
        o = u(split["opp"],  pts_opp)
        if weakly_dominates((s, o), (c_s, c_o)):
            return {
                "self": split["self"], "opp": split["opp"],
                "u_self": s, "u_opp_true": o,
                "delta_self": s - c_s, "delta_opp": o - c_o,
            }
    return None


# ── Pending-offer reconstruction ───────────────────────────────────────────
def pending_self_points(pending_offer: Optional[Mapping],
                        pts_self: Mapping[str, int]) -> Optional[int]:
    """The pending offer in the baseline log is stored in AGENT's view
    (``issue2theyget`` in ``run_protocol1.py``): the counts are what the
    opponent is offering the agent. No inversion needed."""
    if not pending_offer:
        return None
    try:
        counts_self = {i: int(pending_offer.get(i, 0)) for i in ITEMS}
    except (TypeError, ValueError):
        return None
    if any(v < 0 or v > 3 for v in counts_self.values()):
        return None
    return u(counts_self, pts_self)


# ── Main pipeline ──────────────────────────────────────────────────────────
def scan(turns_path: Path) -> Dict:
    stats = {
        "n_turns": 0,
        "n_opp_inference_committed": 0,
        "n_opp_inference_hedged": 0,
        "A_top_wrong": 0,
        "A_top_wrong_examples": [],
        "B_accepts_total": 0,
        "B_accepts_dominated": 0,
        "B_accepts_dominated_examples": [],
        "C_counter_total": 0,
        "C_counter_dominated": 0,
        "C_counter_dominated_examples": [],
        "C_counter_strictly_worse_for_self": 0,
        "D_hedge_rate": 0.0,
    }

    with turns_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            stats["n_turns"] += 1

            opp_pri_true = row.get("opponent_priorities") or {}
            agent_pri    = row.get("agent_priorities") or {}
            if not opp_pri_true or not agent_pri:
                continue
            pts_opp_true  = priorities_to_points(opp_pri_true)
            pts_self_true = priorities_to_points(agent_pri)
            true_top = opp_pri_true.get("High")

            # (A) Top-opponent claim
            inf = row.get("parsed_opponent_inference") or ""
            top, chain, hedged = claimed_top(inf)
            if hedged:
                stats["n_opp_inference_hedged"] += 1
            if top is not None:
                stats["n_opp_inference_committed"] += 1
                if top != true_top:
                    stats["A_top_wrong"] += 1
                    if len(stats["A_top_wrong_examples"]) < 12:
                        stats["A_top_wrong_examples"].append({
                            "dialogue_id": row["dialogue_id"],
                            "turn_index": row["turn_index"],
                            "agent_claim_top": top,
                            "agent_claim_chain": chain,
                            "true_top": true_top,
                            "true_ordering": [opp_pri_true["High"],
                                              opp_pri_true["Medium"],
                                              opp_pri_true["Low"]],
                            "evidence_snippet": inf[:400],
                        })

            decision = row.get("parsed_decision") or {}
            action = decision.get("action")
            counter = decision.get("counter_offer")

            # (B) Accepts dominated offers
            if action == "accept":
                stats["B_accepts_total"] += 1
                pend_pts = pending_self_points(row.get("pending_offer"),
                                               pts_self_true)
                if pend_pts is not None:
                    # Best obtainable for self if they gave the opp a
                    # non-trivial floor (15 pts ≈ walkaway-equivalent).
                    pareto_self = pareto_max_self_given_floor(
                        pts_self_true, pts_opp_true, opp_floor=15,
                    )
                    if pareto_self > pend_pts + 3:  # ≥4-pt leftover on table
                        stats["B_accepts_dominated"] += 1
                        if len(stats["B_accepts_dominated_examples"]) < 12:
                            stats["B_accepts_dominated_examples"].append({
                                "dialogue_id": row["dialogue_id"],
                                "turn_index": row["turn_index"],
                                "pending_offer_opp_view": row["pending_offer"],
                                "pending_u_self": pend_pts,
                                "pareto_max_self_with_floor15": pareto_self,
                                "leftover_on_table": pareto_self - pend_pts,
                            })

            # (C) Proposes dominated offers
            if action in ("reject", "propose") and isinstance(counter, dict):
                try:
                    counts = {i: int(counter.get(i, 0)) for i in ITEMS}
                except (TypeError, ValueError):
                    counts = None
                if counts and all(0 <= v <= 3 for v in counts.values()):
                    stats["C_counter_total"] += 1
                    dom = counter_offer_dominated(counts, pts_self_true, pts_opp_true)
                    if dom is not None:
                        stats["C_counter_dominated"] += 1
                        if dom["delta_self"] > 0:
                            stats["C_counter_strictly_worse_for_self"] += 1
                        if len(stats["C_counter_dominated_examples"]) < 12:
                            stats["C_counter_dominated_examples"].append({
                                "dialogue_id": row["dialogue_id"],
                                "turn_index": row["turn_index"],
                                "agent_priorities": agent_pri,
                                "true_opp_priorities": opp_pri_true,
                                "counter": counts,
                                "counter_u_self": u(counts, pts_self_true),
                                "counter_u_opp_true": u({i: 3 - counts[i] for i in ITEMS}, pts_opp_true),
                                "dominated_by": dom,
                            })

    if stats["n_turns"]:
        stats["D_hedge_rate"] = stats["n_opp_inference_hedged"] / stats["n_turns"]
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=Path, required=True)
    ap.add_argument("--out",   type=Path, required=True)
    args = ap.parse_args()

    stats = scan(args.turns)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    # Short human summary to stdout.
    print(f"scan over {stats['n_turns']} agent-turns")
    print(f"  (A) wrong top-priority claim: "
          f"{stats['A_top_wrong']} / {stats['n_opp_inference_committed']} committed "
          f"({stats['A_top_wrong'] / max(1, stats['n_opp_inference_committed']):.1%}), "
          f"{stats['n_opp_inference_hedged']} hedged ({stats['D_hedge_rate']:.1%})")
    print(f"  (B) accepts with >=4pt leftover on table: "
          f"{stats['B_accepts_dominated']} / {stats['B_accepts_total']} accepts "
          f"({stats['B_accepts_dominated'] / max(1, stats['B_accepts_total']):.1%})")
    print(f"  (C) counter-offers dominated by another split: "
          f"{stats['C_counter_dominated']} / {stats['C_counter_total']} counters "
          f"({stats['C_counter_dominated'] / max(1, stats['C_counter_total']):.1%}); "
          f"{stats['C_counter_strictly_worse_for_self']} strictly worse for self")


if __name__ == "__main__":
    main()
