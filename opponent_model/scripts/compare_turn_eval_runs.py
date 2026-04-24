#!/usr/bin/env python3
"""Compare two ``opponent_model.turn_eval_run`` outputs (paper table + matched slice).

Usage (repo root):

    python -m opponent_model.scripts.compare_turn_eval_runs \\
        opponent_model/results/turn_eval_structured_cot_p3_live_70b_m1_150/turn_summary.json \\
        opponent_model/results/turn_eval_bayesian_lambda1.0_m5_f0.50_full150/turn_summary.json

With per-turn records, also reports **matched-support** Accept F1 on the
intersection of turns where *both* runs have a scored accept decision (same
harness rule as ``turn_level_eval``):

    python -m opponent_model.scripts.compare_turn_eval_runs \\
        path/to/baseline/turn_summary.json \\
        path/to/bayesian/turn_summary.json \\
        --records-a path/to/baseline/turn_records.jsonl \\
        --records-b path/to/bayesian/turn_records.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from opponent_model.turn_level_metrics import binary_accept_f1, cosine_similarity


def _load_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing summary: {path}")
    return json.loads(path.read_text())


def _eligible_accept_record(rec: Mapping[str, Any]) -> bool:
    gold = rec["true"].get("accept")
    pred = rec["pred"].get("accept")
    pend = rec.get("pending_offer")
    return (
        gold is not None
        and pred is not None
        and pend is not None
        and bool(pend.get("to_perspective", False))
    )


def _record_key(rec: Mapping[str, Any]) -> Tuple[Any, str, int]:
    return (
        rec["dialogue_id"],
        str(rec.get("perspective", "")),
        int(rec["turn_index"]),
    )


def _accept_map_from_jsonl(path: Path) -> Dict[Tuple[Any, str, int], Tuple[bool, bool]]:
    out: Dict[Tuple[Any, str, int], Tuple[bool, bool]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not _eligible_accept_record(rec):
                continue
            k = _record_key(rec)
            out[k] = (bool(rec["pred"]["accept"]), bool(rec["true"]["accept"]))
    return out


def _action_map_from_jsonl(path: Path) -> Dict[Tuple[Any, str, int], str]:
    out: Dict[Tuple[Any, str, int], str] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            action = rec.get("pred", {}).get("action")
            if action is None:
                continue
            out[_record_key(rec)] = str(action)
    return out


def _posterior_map_from_jsonl(path: Path) -> Dict[Tuple[Any, str, int], np.ndarray]:
    out: Dict[Tuple[Any, str, int], np.ndarray] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            posterior = rec.get("pred", {}).get("posterior")
            if posterior is None:
                continue
            arr = np.asarray(posterior, dtype=float).flatten()
            if arr.ndim != 1 or arr.size == 0:
                continue
            s = float(arr.sum())
            if s <= 0:
                continue
            out[_record_key(rec)] = arr / s
    return out


def _mean_kl(p_rows: Sequence[np.ndarray], q_rows: Sequence[np.ndarray], eps: float = 1e-12) -> float:
    vals = []
    for p, q in zip(p_rows, q_rows):
        p1 = np.clip(np.asarray(p, dtype=float), eps, 1.0)
        q1 = np.clip(np.asarray(q, dtype=float), eps, 1.0)
        p1 = p1 / p1.sum()
        q1 = q1 / q1.sum()
        vals.append(float(np.sum(p1 * np.log(p1 / q1))))
    return float(np.mean(vals)) if vals else float("nan")


def _fmt_summary(label: str, s: Dict[str, Any]) -> None:
    acc = s.get("accept") or {}
    bid = s.get("bid_cosine") or {}
    st = s.get("strategy_macro_f1") or {}
    br = s.get("brier") or {}
    print(f"{label}:")
    af, an = acc.get("f1"), acc.get("support")
    ap, ar = acc.get("precision"), acc.get("recall")
    if af is not None and an is not None:
        print(
            f"  accept F1     {float(af):.4f}  (n={an}, "
            f"P={float(ap):.3f} R={float(ar):.3f})"
        )
    else:
        print("  accept F1     n/a")
    bm, bn = bid.get("mean"), bid.get("support")
    if bm is not None and bn is not None:
        print(f"  bid cosine    {float(bm):.4f}  (n={bn})")
    else:
        print("  bid cosine    n/a")
    sf, sn = st.get("macro_f1"), st.get("support")
    if sf is not None and sn is not None:
        print(f"  strategy m-F1 {float(sf):.4f}  (n={sn})")
    else:
        print("  strategy m-F1 n/a")
    br_m, br_n = br.get("mean"), br.get("support")
    if br_m is not None and br_n is not None:
        print(f"  Brier         {br_m:.4f}  (n={br_n})")
    else:
        print("  Brier         n/a")
    print()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summary_a", type=Path, help="turn_summary.json (e.g. baseline)")
    ap.add_argument("summary_b", type=Path, help="turn_summary.json (e.g. bayesian)")
    ap.add_argument("--label-a", default="A (baseline)")
    ap.add_argument("--label-b", default="B (bayesian)")
    ap.add_argument("--records-a", type=Path, default=None,
                    help="turn_records.jsonl for matched-support slice")
    ap.add_argument("--records-b", type=Path, default=None,
                    help="turn_records.jsonl for matched-support slice")
    args = ap.parse_args(argv)

    sa = _load_summary(args.summary_a)
    sb = _load_summary(args.summary_b)

    print("=== Headline metrics (from turn_summary.json) ===\n")
    _fmt_summary(args.label_a, sa)
    _fmt_summary(args.label_b, sb)

    fa = (sa.get("accept") or {}).get("f1")
    fb = (sb.get("accept") or {}).get("f1")
    if isinstance(fa, (int, float)) and isinstance(fb, (int, float)):
        print(f"Accept-F1 delta ({args.label_b} − {args.label_a}): {fb - fa:+.4f}\n")

    if args.records_a and args.records_b:
        ma = _accept_map_from_jsonl(args.records_a)
        mb = _accept_map_from_jsonl(args.records_b)
        keys: Set[Tuple[Any, str, int]] = set(ma.keys()) & set(mb.keys())
        mismatched = [
            k for k in keys
            if ma[k][1] != mb[k][1]
        ]
        if mismatched:
            print(
                f"WARNING: {len(mismatched)} keys have differing gold accept labels "
                f"— records are not from the same harness run.\n"
            )

        pairs_a = [ma[k] for k in sorted(keys)]
        pairs_b = [mb[k] for k in sorted(keys)]
        ra = binary_accept_f1([(p, g) for p, g in pairs_a])
        rb = binary_accept_f1([(p, g) for p, g in pairs_b])
        print("=== Matched-support Accept F1 (same turns, both agents scored) ===\n")
        print(f"  Intersection size: {len(keys)}")
        print(
            f"  {args.label_a}: F1={ra['f1']:.4f}  "
            f"(P={ra['precision']:.3f} R={ra['recall']:.3f})"
        )
        print(
            f"  {args.label_b}: F1={rb['f1']:.4f}  "
            f"(P={rb['precision']:.3f} R={rb['recall']:.3f})"
        )
        print(f"  Δ F1: {rb['f1'] - ra['f1']:+.4f}\n")

        # Matched bid cosine (same turn indices in intersection of bid-scored turns)
        def bid_map(p: Path) -> Dict[Tuple[Any, str, int], Tuple[np.ndarray, np.ndarray]]:
            m: Dict[Tuple[Any, str, int], Tuple[np.ndarray, np.ndarray]] = {}
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    gb = rec["true"].get("bid")
                    pb = rec["pred"].get("bid")
                    if gb is None or pb is None:
                        continue
                    k = _record_key(rec)
                    m[k] = (
                        np.asarray(pb, dtype=float),
                        np.asarray(gb, dtype=float),
                    )
            return m

        bma = bid_map(args.records_a)
        bmb = bid_map(args.records_b)
        bkeys = set(bma.keys()) & set(bmb.keys())
        if bkeys:
            cos_a = [
                cosine_similarity(bma[k][0], bma[k][1]) for k in sorted(bkeys)
            ]
            cos_b = [
                cosine_similarity(bmb[k][0], bmb[k][1]) for k in sorted(bkeys)
            ]
            print("=== Matched-support bid cosine (intersection of Submit-Deal turns) ===\n")
            print(f"  Intersection size: {len(bkeys)}")
            print(f"  {args.label_a}: mean cosine = {float(np.mean(cos_a)):.4f}")
            print(f"  {args.label_b}: mean cosine = {float(np.mean(cos_b)):.4f}\n")

        ama = _action_map_from_jsonl(args.records_a)
        amb = _action_map_from_jsonl(args.records_b)
        akeys = set(ama.keys()) & set(amb.keys())
        if akeys:
            match = sum(1 for k in akeys if ama[k] == amb[k])
            print("=== Action agreement (matched turns with explicit action labels) ===\n")
            print(f"  Intersection size: {len(akeys)}")
            print(f"  Exact match rate: {match / len(akeys):.4f}\n")

        pma = _posterior_map_from_jsonl(args.records_a)
        pmb = _posterior_map_from_jsonl(args.records_b)
        pkeys = set(pma.keys()) & set(pmb.keys())
        if pkeys:
            pa = [pma[k] for k in sorted(pkeys)]
            pb = [pmb[k] for k in sorted(pkeys)]
            print("=== Posterior divergence (matched turns with both posteriors) ===\n")
            print(f"  Intersection size: {len(pkeys)}")
            print(
                f"  mean KL({args.label_a} || {args.label_b}) = "
                f"{_mean_kl(pa, pb):.6f}"
            )
            print(
                f"  mean KL({args.label_b} || {args.label_a}) = "
                f"{_mean_kl(pb, pa):.6f}\n"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
