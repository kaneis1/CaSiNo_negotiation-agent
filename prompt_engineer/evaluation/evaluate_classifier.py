#!/usr/bin/env python3
"""Evaluate the LLM-based strategy classifier against CaSiNo-Ann gold labels.

Compares per-strategy P/R/F1 to the CaSiNo paper baselines (Table 3,
NAACL 2021) and evaluates opponent preference inference accuracy.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from prompt_engineer.core.classify_strategy import (
    CASINO_ITEMS,
    VALID_LABELS,
    classify_with_preferences,
    format_context,
)

# ── CaSiNo paper baselines (Table 3, best BERT model) ─────────────────────

PAPER_BASELINES_F1 = {
    "small-talk":           82.6,
    "self-need":            75.2,
    "other-need":           78.8,
    "elicit-pref":          81.8,
    "no-need":              46.2,
    "promote-coordination": 70.3,
    "vouch-fair":           66.1,
    "uv-part":              47.3,
    "showing-empathy":      65.0,
    "non-strategic":        72.4,
}

DEAL_ACTIONS = {"Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away"}


# ── Core evaluator ─────────────────────────────────────────────────────────


def evaluate_classifier(
    annotated_dialogues: List[Dict[str, Any]],
    llm_client: Any,
    context_window: int = 5,
    max_dialogues: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compare classifier output to ground-truth annotations.

    Evaluates both:
      • Strategy detection  — per-label P/R/F1, macro/micro averages
      • Preference inference — item-level accuracy and exact-match rate
        (using the last utterance's preference prediction per dialogue)

    Args:
        annotated_dialogues: CaSiNo dialogues with non-empty annotations.
        llm_client: Object with .generate(prompt: str) -> str.
        context_window: Number of preceding turns to include as context.
        max_dialogues: Cap on number of dialogues (None = all).
        verbose: Print progress every 10 dialogues.

    Returns:
        {
            "num_dialogues": int,
            "num_utterances": int,
            "per_label": {label: {"precision", "recall", "f1", "support"}},
            "macro": {"precision", "recall", "f1"},
            "micro": {"precision", "recall", "f1"},
            "preference": {
                "item_accuracy":        float,   # % items ranked correctly
                "exact_match_rate":     float,   # % dialogues all-3 correct
                "dialogues_evaluated":  int,
            },
        }
    """
    dialogues = [d for d in annotated_dialogues if d.get("annotations")]
    if max_dialogues:
        dialogues = dialogues[:max_dialogues]

    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    total_correct   = 0
    total_predicted = 0
    total_gold      = 0
    n_utterances    = 0

    pref_correct_items   = 0
    pref_total_items     = 0
    pref_exact_dialogues = 0
    pref_evaluated       = 0

    for d_idx, dialogue in enumerate(dialogues):
        chat_logs        = dialogue["chat_logs"]
        gold_annotations = dialogue["annotations"]
        participant_info = dialogue.get("participant_info", {})

        # Identify opponent (second speaker to appear)
        speakers_seen: List[str] = []
        for turn in chat_logs:
            if turn["id"] not in speakers_seen:
                speakers_seen.append(turn["id"])
            if len(speakers_seen) == 2:
                break
        opponent_id = speakers_seen[1] if len(speakers_seen) > 1 else None

        utt_idx         = 0
        last_pred_prefs: Optional[Dict[str, str]] = None

        for i, turn in enumerate(chat_logs):
            if turn["text"] in DEAL_ACTIONS:
                continue
            if utt_idx >= len(gold_annotations):
                break

            gold_text, gold_str = gold_annotations[utt_idx]
            utt_idx += 1

            gt_labels = {s.strip() for s in gold_str.split(",") if s.strip()}
            context   = format_context(chat_logs, i, window=context_window)

            out = classify_with_preferences(turn["text"], context, llm_client)
            pred_labels     = set(out["strategies"])
            last_pred_prefs = out["opponent_preferences"]
            print(f"")
            for label in VALID_LABELS:
                in_gt   = label in gt_labels
                in_pred = label in pred_labels
                if in_gt and in_pred:
                    counts[label]["tp"] += 1
                elif in_pred and not in_gt:
                    counts[label]["fp"] += 1
                elif in_gt and not in_pred:
                    counts[label]["fn"] += 1

            total_correct   += len(pred_labels & gt_labels)
            total_predicted += len(pred_labels)
            total_gold      += len(gt_labels)
            n_utterances    += 1

        # ── Preference evaluation for this dialogue ────────────────────
        if opponent_id and last_pred_prefs:
            gold_v2i = (
                participant_info.get(opponent_id, {})
                .get("value2issue", {})          # {"High": "Food", ...}
            )
            if gold_v2i:
                gold_prefs = {item: level for level, item in gold_v2i.items()}
                pref_evaluated += 1
                correct = sum(
                    last_pred_prefs.get(item) == gold_prefs.get(item)
                    for item in CASINO_ITEMS if item in gold_prefs
                )
                total = sum(1 for item in CASINO_ITEMS if item in gold_prefs)
                pref_correct_items += correct
                pref_total_items   += total
                if correct == total:
                    pref_exact_dialogues += 1

        if verbose and (d_idx + 1) % 10 == 0:
            print(
                f"  Evaluated {d_idx + 1}/{len(dialogues)} dialogues "
                f"({n_utterances} utterances so far)"
            )

    # ── Per-label strategy metrics ─────────────────────────────────────

    per_label: Dict[str, Dict[str, float]] = {}
    for label in sorted(VALID_LABELS):
        c  = counts[label]
        p  = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
        r  = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_label[label] = {
            "precision": p, "recall": r, "f1": f1,
            "support": c["tp"] + c["fn"],
        }

    macro_p  = sum(v["precision"] for v in per_label.values()) / len(per_label)
    macro_r  = sum(v["recall"]    for v in per_label.values()) / len(per_label)
    macro_f1 = sum(v["f1"]        for v in per_label.values()) / len(per_label)

    micro_p  = total_correct / total_predicted if total_predicted else 0.0
    micro_r  = total_correct / total_gold      if total_gold      else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r)
        if (micro_p + micro_r) else 0.0
    )

    return {
        "num_dialogues":  len(dialogues),
        "num_utterances": n_utterances,
        "per_label":      per_label,
        "macro":          {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "micro":          {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "preference": {
            "item_accuracy":       pref_correct_items / pref_total_items if pref_total_items else None,
            "exact_match_rate":    pref_exact_dialogues / pref_evaluated  if pref_evaluated  else None,
            "dialogues_evaluated": pref_evaluated,
        },
    }


# ── Pretty-print ───────────────────────────────────────────────────────────


def print_results(results: Dict[str, Any], experiment_name: str = "") -> None:
    """Print evaluation results table with paper baseline comparison
    and preference inference summary."""

    title = f"Results — {experiment_name}" if experiment_name else "Results"
    print(f"\n{'=' * 70}")
    print(title)
    print(f"Evaluated {results['num_dialogues']} dialogues, "
          f"{results['num_utterances']} utterances\n")

    # ── Strategy table ─────────────────────────────────────────────────
    header = (
        f"{'Strategy':<25s}  {'P':>6s}  {'R':>6s}  {'F1':>6s}  "
        f"{'Paper':>6s}  {'Δ F1':>7s}  {'Support':>7s}"
    )
    print(header)
    print("─" * len(header))

    for label in sorted(results["per_label"].keys()):
        m         = results["per_label"][label]
        paper_f1  = PAPER_BASELINES_F1.get(label)
        delta     = (m["f1"] * 100 - paper_f1) if paper_f1 is not None else None
        delta_str = f"{delta:+.1f}" if delta is not None else "   n/a"
        paper_str = f"{paper_f1:.1f}"  if paper_f1 is not None else "   n/a"

        print(
            f"{label:<25s}  {m['precision']:6.3f}  {m['recall']:6.3f}  "
            f"{m['f1'] * 100:6.1f}  {paper_str:>6s}  {delta_str:>7s}  "
            f"{m['support']:>7d}"
        )

    print("─" * len(header))

    macro = results["macro"]
    micro = results["micro"]
    print(
        f"{'Macro avg':<25s}  {macro['precision']:6.3f}  {macro['recall']:6.3f}  "
        f"{macro['f1'] * 100:6.1f}"
    )
    print(
        f"{'Micro avg':<25s}  {micro['precision']:6.3f}  {micro['recall']:6.3f}  "
        f"{micro['f1'] * 100:6.1f}"
    )

    # ── Preference summary ─────────────────────────────────────────────
    pref = results.get("preference", {})
    print(f"\n{'─' * len(header)}")
    print("Opponent Preference Inference")
    print(f"{'─' * len(header)}")

    item_acc   = pref.get("item_accuracy")
    exact_rate = pref.get("exact_match_rate")
    n_eval     = pref.get("dialogues_evaluated", 0)

    print(f"  Item-level accuracy  : "
          f"{item_acc * 100:.1f}%"   if item_acc   is not None else "  Item-level accuracy  : N/A")
    print(f"  Exact match rate     : "
          f"{exact_rate * 100:.1f}%" if exact_rate is not None else "  Exact match rate     : N/A")
    print(f"  Dialogues evaluated  : {n_eval}")
    print(f"{'=' * 70}\n")


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate strategy classifier against CaSiNo-Ann."
    )
    parser.add_argument(
        "--input", type=Path,
        default=Path(__file__).resolve().parents[2] / "CaSiNo" / "data" / "casino_ann.json",
    )
    parser.add_argument("--max-dialogues", type=int, default=None)
    parser.add_argument("--context-window", type=int, default=5)
    args = parser.parse_args()

    print("CaSiNo Strategy Classifier Evaluation")
    print("=" * 40)
    print(f"Data:           {args.input}")
    print(f"Max dialogues:  {args.max_dialogues or 'all'}")
    print(f"Context window: {args.context_window}")
    print()
    print("Paper baselines (BERT, Table 3):")
    for label, f1 in sorted(PAPER_BASELINES_F1.items()):
        print(f"  {label:.<30} F1 = {f1:.1f}")
