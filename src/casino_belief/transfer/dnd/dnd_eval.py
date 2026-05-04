"""Evaluate DND transfer rows with prefs-only posterior providers."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from casino_belief.transfer.dnd.dnd_data import (
    DNDRecord,
    DND_ITEMS,
    compute_stats,
    download_raw_split,
    parse_split_file,
)
from casino_belief.transfer.dnd.dnd_menu import (
    allocation_vector,
    build_dnd_menu,
    build_value_map_543,
    empirical_value_map,
    utility,
)
from casino_belief.transfer.dnd.dnd_metrics import (
    brier_reference,
    cosine_similarity,
    ema,
    ndcg_at_3,
    normalized_brier_sum,
    summarize_snapshot_metrics,
    top1,
)
from casino_belief.transfer.dnd.dnd_posterior import (
    ORDERING_INDEX,
    ORDERINGS,
    DNDChatModel,
    DirectPosteriorDNDProvider,
    MCPrefsDNDProvider,
    RuleDNDProvider,
    UniformDNDProvider,
)

logger = logging.getLogger("casino_belief.transfer.dnd.dnd_eval")

DEFAULT_BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_ADAPTER = "artifacts/training_metadata/teacher_lora_run/lora_best"
DEFAULT_ROOT = Path("artifacts/results/dnd_transfer/main")


def _load_records(args: argparse.Namespace) -> List[DNDRecord]:
    raw_dir = Path(args.raw_dir)
    path = raw_dir / f"{args.split}.txt"
    if not path.exists() or args.download_raw:
        path = download_raw_split(args.split, raw_dir, overwrite=args.download_raw)
    records = parse_split_file(path, split=args.split)
    if args.max_examples is not None:
        records = records[: args.max_examples]
    return records


def _filter_records(records: Sequence[DNDRecord], strict_filter: str) -> List[DNDRecord]:
    if strict_filter == "none":
        return list(records)
    if strict_filter == "partner":
        return [r for r in records if r.partner_ordering is not None]
    if strict_filter == "both":
        return [r for r in records if r.strict_both]
    raise ValueError(f"unknown strict_filter {strict_filter!r}")


def _provider(args: argparse.Namespace):
    if args.provider == "uniform":
        return UniformDNDProvider()
    if args.provider == "rule":
        return RuleDNDProvider()
    model = DNDChatModel(
        base_model=args.base_model,
        adapter_path=args.adapter,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    if args.provider == "mc_k16":
        return MCPrefsDNDProvider(model, K=args.posterior_k, temperature=args.posterior_temperature)
    if args.provider == "direct_zero_shot":
        return DirectPosteriorDNDProvider(model)
    raise ValueError(f"unknown provider {args.provider!r}")


def _value_map(args: argparse.Namespace):
    if args.value_mode == "543":
        return build_value_map_543(ORDERINGS)
    raw_train = Path(args.raw_dir) / "train.txt"
    if not raw_train.exists():
        raw_train = download_raw_split("train", Path(args.raw_dir))
    train_records = parse_split_file(raw_train, split="train")
    return empirical_value_map(train_records, ORDERINGS)


def _selection_depth(record: DNDRecord) -> int:
    k = 0
    for turn in record.dialogue:
        if turn.is_selection:
            return k
        if turn.speaker == "THEM":
            k += 1
    return k


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "dnd_eval.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.FileHandler(log_path, mode="a"), logging.StreamHandler()],
    )
    logger.info("Args: %s", vars(args))
    all_records = _load_records(args)
    records = _filter_records(all_records, args.strict_filter)
    provider = _provider(args)
    opp_value_map = _value_map(args)

    rows_path = output_dir / "turn_records.jsonl"
    if rows_path.exists():
        rows_path.unlink()
    snapshot_rows: List[Dict[str, Any]] = []
    selection_depths: List[int] = []
    t0 = time.time()

    for rec_i, rec in enumerate(records):
        selection_depths.append(_selection_depth(rec))
        history = []
        opp_seen = 0
        for turn_i, turn in enumerate(rec.dialogue):
            if turn.is_selection:
                break
            history.append(turn)
            if turn.speaker != "THEM":
                continue
            opp_seen += 1
            if opp_seen > args.max_k:
                break

            posterior = provider.posterior(record=rec, history=tuple(history), name_mode=args.name_mode)
            pred_idx = int(np.argmax(posterior))
            pred_ordering = ORDERINGS[pred_idx]
            true_ordering = rec.partner_ordering
            true_idx = ORDERING_INDEX[true_ordering] if true_ordering is not None else None
            brier = normalized_brier_sum(posterior, true_idx) if true_idx is not None else None
            menu = build_dnd_menu(
                posterior=posterior,
                orderings=ORDERINGS,
                counts=rec.counts,
                self_values=rec.self_values,
                opp_value_map=opp_value_map,
                lambda_=args.lambda_,
                top_k=5,
            )
            top = menu[0]
            pred_vec = allocation_vector(top.self_tuple, rec.counts)
            gold_vec = tuple(rec.output_self) + tuple(rec.output_partner)
            bid_cos = cosine_similarity(pred_vec, gold_vec) if rec.output_valid else None
            final_self_utility = utility(rec.output_self, rec.self_values) if rec.output_valid else None
            row = {
                "dialogue_id": rec.dialogue_id,
                "pair_key": rec.pair_key,
                "split": rec.split,
                "line_index": rec.line_index,
                "k": opp_seen,
                "turn_index": turn_i,
                "name_mode": args.name_mode,
                "provider": args.provider,
                "value_mode": args.value_mode,
                "counts": list(rec.counts),
                "self_values": list(rec.self_values),
                "partner_values": list(rec.partner_values),
                "self_ordering": list(rec.self_ordering) if rec.self_ordering else None,
                "partner_ordering": list(true_ordering) if true_ordering else None,
                "output_valid": rec.output_valid,
                "pred_ordering": list(pred_ordering),
                "posterior": [float(x) for x in posterior],
                "brier": brier,
                "ema": ema(pred_ordering, true_ordering) if true_ordering else None,
                "top1": top1(pred_ordering, true_ordering) if true_ordering else None,
                "ndcg": ndcg_at_3(pred_ordering, true_ordering) if true_ordering else None,
                "menu_top": top.to_dict(),
                "bid_cosine": bid_cos,
                "utility_self": top.u_self,
                "utility_norm": top.u_self / 10.0,
                "final_self_utility": final_self_utility,
                "selection_depth_opp_utterances": _selection_depth(rec),
                "selection_speaker": rec.selection_speaker,
            }
            snapshot_rows.append(row)
            with rows_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    metrics = summarize_snapshot_metrics(snapshot_rows, max_k=args.max_k)
    all_stats = compute_stats(all_records)
    eval_stats = compute_stats(records)
    summary = {
        "config": vars(args),
        "elapsed_seconds": elapsed,
        "n_records_input": len(all_records),
        "n_records_eval": len(records),
        "n_snapshots": len(snapshot_rows),
        "brier_reference_uniform_six_way": brier_reference(6),
        "data_stats_all": all_stats,
        "data_stats_eval": eval_stats,
        "metrics": metrics,
        "selection_diagnostic": {
            "support": len(selection_depths),
            "mean_opp_utterances_before_selection": (
                float(np.mean(selection_depths)) if selection_depths else float("nan")
            ),
            "hist_opp_utterances_before_selection": {
                str(k): int(sum(1 for x in selection_depths if x == k))
                for k in sorted(set(selection_depths))
            },
            "note": "DND <selection> is a terminal commitment, not CaSiNo accept/reject.",
        },
        "provider_summary": provider.summary(),
    }
    with (output_dir / "turn_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    logger.info("Wrote %s", output_dir / "turn_summary.json")
    logger.info("Wrote %s", rows_path)
    return summary


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_ROOT / "data" / "raw")
    p.add_argument("--download-raw", action="store_true")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--provider", choices=("uniform", "rule", "direct_zero_shot", "mc_k16"), required=True)
    p.add_argument("--name-mode", choices=("native", "renamed"), default="native")
    p.add_argument("--value-mode", choices=("543", "empirical"), default="543")
    p.add_argument("--strict-filter", choices=("none", "partner", "both"), default="partner")
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--max-k", type=int, default=5)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--adapter", default=DEFAULT_ADAPTER)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--posterior-temperature", type=float, default=0.7)
    p.add_argument("--posterior-k", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
