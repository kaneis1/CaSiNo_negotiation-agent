"""Build the 50-dialogue DND few-shot prefs-only SFT add-on."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from casino_belief.transfer.dnd.dnd_data import (
    DNDRecord,
    download_raw_split,
    group_records_by_pair,
    parse_split_file,
)
from casino_belief.transfer.dnd.dnd_prompts import build_prefs_target_json, build_prefs_user_prompt, prefs_system_prompt

DEFAULT_ROOT = Path("artifacts/results/dnd_transfer/main")
DEFAULT_CASINO_TRAIN = Path("artifacts/training_metadata/sft_data/sft_train_rows.jsonl")
DEFAULT_CASINO_EVAL = Path("artifacts/training_metadata/sft_data/sft_test_rows.jsonl")


def stable_eval_split(key: str, *, seed: int, eval_fraction: float) -> str:
    digest = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "eval" if bucket < eval_fraction else "train"


def _history_snapshots(record: DNDRecord, *, max_k: int) -> Iterable[tuple[int, list]]:
    history = []
    opp_seen = 0
    for turn in record.dialogue:
        if turn.is_selection:
            break
        history.append(turn)
        if turn.speaker != "THEM":
            continue
        opp_seen += 1
        if opp_seen > max_k:
            break
        yield opp_seen, list(history)


def _row(record: DNDRecord, *, name_mode: str, k: int, history: Sequence[Any], split: str) -> Dict[str, Any]:
    assert record.partner_ordering is not None
    return {
        "messages": [
            {"role": "system", "content": prefs_system_prompt(name_mode=name_mode)},
            {
                "role": "user",
                "content": build_prefs_user_prompt(
                    record=record,
                    history=history,
                    name_mode=name_mode,
                ),
            },
            {
                "role": "assistant",
                "content": build_prefs_target_json(
                    ordering=record.partner_ordering,
                    name_mode=name_mode,
                ),
            },
        ],
        "split": split,
        "source": "dnd_fewshot",
        "name_mode": name_mode,
        "dialogue_id": record.dialogue_id,
        "pair_key": record.pair_key,
        "perspective": "YOU",
        "opp_role": "THEM",
        "k": int(k),
        "true_prefs": list(record.partner_ordering),
    }


def sample_fewshot_records(
    records: Sequence[DNDRecord],
    *,
    n_dialogues: int,
    seed: int,
) -> List[DNDRecord]:
    groups = group_records_by_pair([r for r in records if r.strict_both])
    eligible = [
        sorted(group, key=lambda r: r.line_index)[:2]
        for _key, group in sorted(groups.items())
        if len(group) >= 2
    ]
    if len(eligible) < n_dialogues:
        raise RuntimeError(f"need {n_dialogues} strict paired DND dialogues, found {len(eligible)}")
    rng = random.Random(seed)
    chosen = rng.sample(eligible, n_dialogues)
    out: List[DNDRecord] = []
    for pair in chosen:
        out.extend(pair)
    return sorted(out, key=lambda r: (r.pair_key, r.line_index))


def copy_jsonl(src: Path, dst: Path) -> int:
    n = 0
    with src.open(encoding="utf-8") as fsrc, dst.open("w", encoding="utf-8") as fdst:
        for line in fsrc:
            if line.strip():
                fdst.write(line)
                n += 1
    return n


def append_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir)
    raw_train = raw_dir / "train.txt"
    if not raw_train.exists() or args.download_raw:
        raw_train = download_raw_split("train", raw_dir, overwrite=args.download_raw)
    records = parse_split_file(raw_train, split="train")
    fewshot_records = sample_fewshot_records(
        records,
        n_dialogues=args.n_dialogues,
        seed=args.seed,
    )

    dnd_train: List[Dict[str, Any]] = []
    dnd_eval: List[Dict[str, Any]] = []
    for rec in fewshot_records:
        split = stable_eval_split(rec.pair_key, seed=args.seed, eval_fraction=args.eval_fraction)
        for k, history in _history_snapshots(rec, max_k=args.max_k):
            for name_mode in ("native", "renamed"):
                row = _row(rec, name_mode=name_mode, k=k, history=history, split=split)
                if split == "eval":
                    dnd_eval.append(row)
                else:
                    dnd_train.append(row)

    train_path = out / "dnd_fewshot_train_rows.jsonl"
    eval_path = out / "dnd_fewshot_eval_rows.jsonl"
    combined_train = out / "combined_train_rows.jsonl"
    combined_eval = out / "combined_eval_rows.jsonl"
    selected_path = out / "dnd_fewshot_50.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for row in dnd_train:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with eval_path.open("w", encoding="utf-8") as f:
        for row in dnd_eval:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with selected_path.open("w", encoding="utf-8") as f:
        for rec in fewshot_records:
            f.write(json.dumps(rec.to_json(), ensure_ascii=False) + "\n")

    n_casino_train = copy_jsonl(Path(args.casino_train), combined_train)
    append_rows(combined_train, dnd_train)
    n_casino_eval = copy_jsonl(Path(args.casino_eval), combined_eval)
    append_rows(combined_eval, dnd_eval)

    summary = {
        "output_dir": str(out),
        "raw_train": str(raw_train),
        "casino_train": str(args.casino_train),
        "casino_eval": str(args.casino_eval),
        "combined_train": str(combined_train),
        "combined_eval": str(combined_eval),
        "dnd_train": str(train_path),
        "dnd_eval": str(eval_path),
        "selected_records": str(selected_path),
        "n_dialogues": args.n_dialogues,
        "n_perspective_records": len(fewshot_records),
        "n_casino_train": n_casino_train,
        "n_casino_eval": n_casino_eval,
        "n_dnd_train": len(dnd_train),
        "n_dnd_eval": len(dnd_eval),
        "n_combined_train": n_casino_train + len(dnd_train),
        "n_combined_eval": n_casino_eval + len(dnd_eval),
        "rows_by_name_mode": dict(Counter(r["name_mode"] for r in dnd_train + dnd_eval)),
        "rows_by_split": {"train": len(dnd_train), "eval": len(dnd_eval)},
        "seed": args.seed,
        "max_k": args.max_k,
        "eval_fraction": args.eval_fraction,
    }
    (out / "dnd_fewshot_data_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return summary


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "sft_data" / "fewshot_opponent")
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_ROOT / "data" / "raw")
    p.add_argument("--download-raw", action="store_true")
    p.add_argument("--casino-train", type=Path, default=DEFAULT_CASINO_TRAIN)
    p.add_argument("--casino-eval", type=Path, default=DEFAULT_CASINO_EVAL)
    p.add_argument("--n-dialogues", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-fraction", type=float, default=0.10)
    p.add_argument("--max-k", type=int, default=5)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    build_dataset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
