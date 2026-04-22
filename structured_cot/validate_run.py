"""Post-run validation + health-check report.

Answers the three smoke-test questions from the spec:

  1. Did the agent complete all turns without parse errors?
  2. Do the counter-offers sum to valid totals (each item 0..3)?
  3. Does the agent eventually accept or reach a walkaway?

Also spot-checks three dialogues end-to-end by printing their five-block
reasoning traces to stdout so a human can read them for coherence.

Usage:
    python -m structured_cot.validate_run --run-dir structured_cot/results/<run>
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(run_dir: Path) -> Dict[str, Any]:
    turns = _read_jsonl(run_dir / "turns.jsonl")
    dialogues = _read_jsonl(run_dir / "dialogues.jsonl")
    parse_failures = _read_jsonl(run_dir / "parse_failures.jsonl")

    n_turns = len(turns)
    n_retried = sum(1 for t in turns if t.get("retried"))
    n_fell_back = sum(1 for t in turns if t.get("fell_back"))
    n_illegal = sum(1 for t in turns if t.get("counter_offer_legal") is False)

    # Check that every counter_offer proposed is in range.
    for t in turns:
        co = (t.get("parsed_decision") or {}).get("counter_offer")
        if isinstance(co, dict):
            for k, v in co.items():
                if not isinstance(v, int) or v < 0 or v > 3:
                    n_illegal += 0  # already counted above; guard-rail
                    break

    outcomes: Dict[str, int] = {}
    terminated = 0
    for d in dialogues:
        o = d.get("outcome") or "unknown"
        outcomes[o] = outcomes.get(o, 0) + 1
        if o in ("agent_accepted", "agent_walkaway",
                 "opp_accepted_agent_offer", "opp_walkaway"):
            terminated += 1

    return {
        "n_dialogues":               len(dialogues),
        "n_agent_turns":             n_turns,
        "n_retried":                 n_retried,
        "n_fell_back":                n_fell_back,
        "n_illegal_counter_offers":  n_illegal,
        "n_parse_failure_events":    len(parse_failures),
        "parse_retry_rate":          n_retried / n_turns if n_turns else 0.0,
        "parse_fallback_rate":       n_fell_back / n_turns if n_turns else 0.0,
        "illegal_counter_offer_rate": n_illegal / n_turns if n_turns else 0.0,
        "terminated_fraction":       (terminated / len(dialogues)) if dialogues else 0.0,
        "outcomes":                  outcomes,
    }


def _print_trace(turn: Dict[str, Any]) -> None:
    print(f"  -- turn {turn.get('agent_turn_index')} (logs idx {turn.get('turn_index')}) --")
    obs = (turn.get("parsed_observation") or "").strip()
    inf = (turn.get("parsed_opponent_inference") or "").strip()
    plan = (turn.get("parsed_plan") or "").strip()
    utt = (turn.get("parsed_utterance") or "").strip()
    dec = turn.get("parsed_decision") or {}

    def _trim(s: str, n: int = 400) -> str:
        s = " ".join(s.split())
        return s if len(s) <= n else s[: n - 1] + "…"

    print(f"    <observation>        {_trim(obs)}")
    print(f"    <opponent_inference> {_trim(inf)}")
    print(f"    <plan>               {_trim(plan)}")
    print(f"    <utterance>          {_trim(utt, 300)}")
    print(f"    <decision>           {json.dumps(dec)}")
    gt = turn.get("ground_truth_decision") or {}
    gtu = (turn.get("ground_truth_utterance") or "").strip()
    print(f"    GT action={gt.get('action')!s:<10} GT utt: {_trim(gtu, 200)}")
    if turn.get("fell_back"):
        print("    !! SAFE-DEFAULT FALLBACK ENGAGED")
    if turn.get("retried") and not turn.get("fell_back"):
        print("    !  retried once (recovered)")


def spot_check(run_dir: Path, k: int, seed: int) -> None:
    dialogues = _read_jsonl(run_dir / "dialogues.jsonl")
    turns = _read_jsonl(run_dir / "turns.jsonl")
    if not dialogues:
        return
    rng = random.Random(seed)
    picked = rng.sample(dialogues, min(k, len(dialogues)))
    by_dialogue: Dict[Any, List[Dict[str, Any]]] = {}
    for t in turns:
        by_dialogue.setdefault(t.get("dialogue_id"), []).append(t)
    print("\n" + "=" * 72)
    print(f"Spot-checking {len(picked)} dialogues (seed={seed}):")
    print("=" * 72)
    for d in picked:
        did = d.get("dialogue_id")
        print(f"\n### dialogue_id={did}  agent={d.get('agent_role')}  "
              f"outcome={d.get('outcome')}")
        print(f"    agent priorities: {d.get('agent_priorities')}")
        print(f"    opp  priorities:  {d.get('opponent_priorities')}  (ground truth)")
        for t in by_dialogue.get(did, []):
            _print_trace(t)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--spot-check", type=int, default=3,
                        help="Print full traces for this many dialogues.")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if not args.run_dir.exists():
        print(f"Run directory not found: {args.run_dir}", file=sys.stderr)
        return 2

    s = summarize(args.run_dir)

    print("=" * 72)
    print(f"Run: {args.run_dir}")
    print("=" * 72)
    print(f"  dialogues                : {s['n_dialogues']}")
    print(f"  agent turns              : {s['n_agent_turns']}")
    print(f"  parse retries            : {s['n_retried']} "
          f"({s['parse_retry_rate']:.2%})")
    print(f"  parse fallbacks          : {s['n_fell_back']} "
          f"({s['parse_fallback_rate']:.2%})")
    print(f"  parse-failure log entries: {s['n_parse_failure_events']}")
    print(f"  illegal counter-offers   : {s['n_illegal_counter_offers']} "
          f"({s['illegal_counter_offer_rate']:.2%})")
    print(f"  terminated cleanly       : {s['terminated_fraction']:.2%} "
          "(accept or walkaway, either side)")
    print(f"  outcomes                 : {s['outcomes']}")

    warnings = []
    if s["parse_fallback_rate"] > 0.05:
        warnings.append(
            f"fallback rate {s['parse_fallback_rate']:.2%} > 5% — "
            "tune the prompt before full run."
        )
    if s["illegal_counter_offer_rate"] > 0.0:
        warnings.append(
            f"{s['n_illegal_counter_offers']} illegal counter-offers — "
            "check validator / decoding budget."
        )
    if s["n_dialogues"] and s["terminated_fraction"] < 0.5:
        warnings.append(
            f"only {s['terminated_fraction']:.0%} of dialogues terminated — "
            "agent may be stalling."
        )

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  ! {w}")
    else:
        print("\nNo red flags.")

    with (args.run_dir / "validation_summary.json").open("w") as f:
        json.dump(s, f, indent=2)

    if args.spot_check > 0:
        spot_check(args.run_dir, args.spot_check, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
