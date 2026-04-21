"""Build per-fold SFT data for K-fold CV on the full CaSiNo corpus.

Splits the full ``CaSiNo/data/casino.json`` (1030 dialogues) deterministically
into K folds. For each fold:

    held_out_k  : 1/K of dialogues  (e.g. 206 for K=5) — held out for final eval
    pool_k      : the remaining K-1 chunks  (e.g. 824)
        train_k     : 90 % of pool   (e.g. ~742) — SFT training
        in_loop_k   : 10 % of pool   (e.g. ~82)  — TRL eval / early-stopping signal

Per dialogue ID assignment:
    1. Shuffle dialogue_ids with ``--seed`` (deterministic).
    2. Chunk into K equal blocks; block k is ``held_out_k``.
    3. Within ``pool_k`` (already shuffled), the LAST 10 % becomes ``in_loop_k``
       and the rest is ``train_k``. Same seed -> identical splits across reruns.

Outputs (under ``--cv-root sft_8b/results/cv/`` by default):

    fold_{K}/dialogues_heldout.json     full dialogues, ready for sft_8b.eval_run --data
    fold_{K}/sft_data/sft_train_rows.jsonl
    fold_{K}/sft_data/sft_test_rows.jsonl   (= in-loop early-stopping signal)
    fold_{K}/sft_data/data_build_summary.json
    cv_split_manifest.json                  list of dialogue_ids per fold (audit trail)

Usage
-----
    # build ONE fold (used by run_cv_fold.lsf):
    python -m sft_8b.cv_data --fold 0 --n-splits 5

    # build ALL folds at once (useful for sanity checking the splits):
    python -m sft_8b.cv_data --all --n-splits 5
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from sft_8b.data import build_dialogue_rows, DEFAULT_MAX_K

logger = logging.getLogger("sft_8b.cv_data")


DEFAULT_CORPUS    = Path("CaSiNo/data/casino.json")
DEFAULT_CV_ROOT   = Path("sft_8b/results/cv")
DEFAULT_N_SPLITS  = 5
DEFAULT_SEED      = 42
DEFAULT_VAL_FRAC  = 0.10  # fraction of (train pool) reserved as in-loop test


# ── Split planning ────────────────────────────────────────────────────────


def plan_folds(
    dialogue_ids: Sequence[int],
    *,
    n_splits: int,
    seed: int,
    val_frac: float,
) -> List[Dict[str, List[int]]]:
    """Return a list of K dicts {train, in_loop, held_out} of dialogue_ids.

    Deterministic given (dialogue_ids, n_splits, seed, val_frac).
    """
    rng = random.Random(seed)
    shuffled = list(dialogue_ids)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    fold_size = n_total // n_splits
    leftover = n_total - fold_size * n_splits  # spread into the first `leftover` folds

    folds: List[Dict[str, List[int]]] = []
    cursor = 0
    for k in range(n_splits):
        size_k = fold_size + (1 if k < leftover else 0)
        held = shuffled[cursor : cursor + size_k]
        cursor += size_k

        pool = [d for d in shuffled if d not in set(held)]
        # Already shuffled — the LAST val_frac is the in-loop test slice.
        n_in_loop = max(1, int(round(len(pool) * val_frac)))
        train     = pool[:-n_in_loop]
        in_loop   = pool[-n_in_loop:]

        folds.append({
            "fold":     k,
            "n_train":  len(train),
            "n_in_loop": len(in_loop),
            "n_held_out": len(held),
            "train":    sorted(train),
            "in_loop":  sorted(in_loop),
            "held_out": sorted(held),
        })
    return folds


# ── Per-fold builder ──────────────────────────────────────────────────────


def build_fold(
    *,
    dialogues_by_id: Mapping[int, Mapping[str, Any]],
    fold_plan: Mapping[str, Any],
    fold_dir: Path,
    max_k: int,
) -> Dict[str, Any]:
    """Materialize one fold: dialogues_heldout.json + sft_{train,test}_rows.jsonl."""
    fold_dir.mkdir(parents=True, exist_ok=True)
    sft_dir = fold_dir / "sft_data"
    sft_dir.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Any] = {"fold": fold_plan["fold"]}

    # 1) Dump held-out dialogues for sft_8b.eval_run --data ...
    held_out = [dialogues_by_id[d] for d in fold_plan["held_out"]]
    held_path = fold_dir / "dialogues_heldout.json"
    with held_path.open("w") as f:
        json.dump(held_out, f)
    stats["held_out_path"] = str(held_path)
    stats["n_held_out_dialogues"] = len(held_out)

    # 2) Build SFT rows for train + in-loop test (= "test" file in the existing
    #    train.py CLI; same role, different data).
    for split_name, ids in [
        ("train", fold_plan["train"]),
        ("test",  fold_plan["in_loop"]),  # repurpose the existing CLI flag
    ]:
        out_path = sft_dir / f"sft_{split_name}_rows.jsonl"
        n_rows = 0
        n_dlg_with_rows = 0
        with out_path.open("w") as f:
            for did in ids:
                rows, _skips = build_dialogue_rows(
                    dialogues_by_id[did], max_k=max_k,
                )
                if rows:
                    n_dlg_with_rows += 1
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n_rows += 1
        stats[f"{split_name}_rows"] = n_rows
        stats[f"{split_name}_dialogues_with_rows"] = n_dlg_with_rows
        logger.info(
            "  fold %d %s: %d dialogues -> %d SFT rows -> %s",
            fold_plan["fold"], split_name, len(ids), n_rows, out_path,
        )

    # 3) Persist a per-fold build summary alongside the data.
    with (sft_dir / "data_build_summary.json").open("w") as f:
        json.dump(stats, f, indent=2)
    return stats


# ── CLI ────────────────────────────────────────────────────────────────────


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus",   type=Path, default=DEFAULT_CORPUS,
                   help="Full CaSiNo JSON (1030 dialogues).")
    p.add_argument("--cv-root",  type=Path, default=DEFAULT_CV_ROOT,
                   help="Root dir for fold_{K}/ subdirs.")
    p.add_argument("--n-splits", type=int,  default=DEFAULT_N_SPLITS)
    p.add_argument("--seed",     type=int,  default=DEFAULT_SEED)
    p.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC,
                   help="Fraction of train pool kept as in-loop early-stop set.")
    p.add_argument("--max-k",    type=int,  default=DEFAULT_MAX_K)

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--fold", type=int, default=None,
                   help="Build a single fold (0..N-1).")
    g.add_argument("--all",  action="store_true",
                   help="Build every fold (sanity check / one-shot prep).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_argparser().parse_args(argv)

    if not args.corpus.exists():
        print(f"ERROR: corpus not found: {args.corpus}", file=sys.stderr)
        return 2

    with args.corpus.open() as f:
        dialogues = json.load(f)
    dialogues_by_id = {d["dialogue_id"]: d for d in dialogues}
    logger.info("Loaded %d dialogues from %s", len(dialogues), args.corpus)

    folds = plan_folds(
        dialogue_ids=list(dialogues_by_id),
        n_splits=args.n_splits,
        seed=args.seed,
        val_frac=args.val_frac,
    )

    args.cv_root.mkdir(parents=True, exist_ok=True)

    # Always (re)write the manifest so a later --fold call is consistent
    # with whatever the previous --all call wrote.
    manifest = {
        "corpus":   str(args.corpus),
        "n_total":  len(dialogues),
        "n_splits": args.n_splits,
        "seed":     args.seed,
        "val_frac": args.val_frac,
        "max_k":    args.max_k,
        "folds":    folds,
    }
    with (args.cv_root / "cv_split_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote split manifest -> %s/cv_split_manifest.json", args.cv_root)

    target_folds = range(args.n_splits) if args.all else [args.fold]
    for k in target_folds:
        if not 0 <= k < args.n_splits:
            print(f"ERROR: --fold {k} out of range 0..{args.n_splits-1}",
                  file=sys.stderr)
            return 2
        fold_dir = args.cv_root / f"fold_{k}"
        logger.info("=== building fold %d -> %s ===", k, fold_dir)
        build_fold(
            dialogues_by_id=dialogues_by_id,
            fold_plan=folds[k],
            fold_dir=fold_dir,
            max_k=args.max_k,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
