#!/usr/bin/env python3
"""Opponent modeling module for CaSiNo negotiation.

Infers the opponent's priority mapping ({"High": item, "Medium": item,
"Low": item}) from conversational signals, detected strategies, and
item mentions.  Also tracks negotiation style (prosocial vs proself).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Item mention extraction ────────────────────────────────────────────────

ITEM_KEYWORDS: Dict[str, List[str]] = {
    "Food": ["food", "meal", "eat", "hungry", "snack", "diet", "feed", "cook",
             "nutrition", "starve", "protein", "calories", "vegetarian", "vegan"],
    "Water": ["water", "drink", "hydrat", "thirst", "dehydrat", "fluid",
              "beverage", "canteen", "bottle"],
    "Firewood": ["firewood", "fire", "wood", "warm", "cold", "heat", "freeze",
                 "camp fire", "campfire", "bonfire", "flame", "chill", "lantern",
                 "light", "s'more", "smore"],
}


def extract_mentioned_items(text: str) -> List[str]:
    """Keyword extraction for Food/Water/Firewood mentions."""
    text_lower = text.lower()
    items: List[str] = []
    for item, keywords in ITEM_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            items.append(item)
    return items


# ── Signal weights ─────────────────────────────────────────────────────────

STRONG_SIGNAL = 2.0
WEAK_SIGNAL = 1.0
NEGATIVE_SIGNAL = -1.5

# Strategies that indicate HIGH priority for mentioned items
NEED_STRATEGIES = {"self-need", "other-need"}

# Strategies that indicate LOW priority for mentioned items
NO_NEED_STRATEGIES = {"no-need"}

# Strategies hinting at proself orientation
PROSELF_STRATEGIES = {"vouch-fair", "uv-part"}

# Strategies hinting at prosocial orientation
PROSOCIAL_STRATEGIES = {"small-talk", "showing-empathy", "promote-coordination"}


# ── Opponent model ─────────────────────────────────────────────────────────


class OpponentModel:
    """Track and infer an opponent's priorities and negotiation style.

    Priority scores accumulate over the conversation. Higher scores for an
    item indicate the opponent likely values it more (High priority).
    Negative scores suggest Low priority.
    """

    def __init__(self) -> None:
        self.priority_scores: Dict[str, float] = {
            "Food": 0.0, "Water": 0.0, "Firewood": 0.0,
        }
        self.confidence: str = "low"       # low / medium / high
        self.style: str = "unknown"        # prosocial / proself / unknown
        self.prosocial_count: int = 0
        self.proself_count: int = 0
        self.signals: List[str] = []

    # ── Core update ────────────────────────────────────────────────────

    def update(
        self,
        utterance_text: str,
        detected_strategies: List[str],
        mentioned_items: Optional[List[str]] = None,
    ) -> None:
        """Update model based on an observed opponent utterance.

        Args:
            utterance_text: The raw utterance text.
            detected_strategies: Strategy labels from the classifier.
            mentioned_items: Items mentioned in the utterance. If None,
                extracted automatically via keyword matching.
        """
        if mentioned_items is None:
            mentioned_items = extract_mentioned_items(utterance_text)

        for strategy in detected_strategies:
            if strategy in NEED_STRATEGIES:
                for item in mentioned_items:
                    self.priority_scores[item] += STRONG_SIGNAL
                    self.signals.append(f"{strategy} → {item}")

            elif strategy in NO_NEED_STRATEGIES:
                for item in mentioned_items:
                    self.priority_scores[item] += NEGATIVE_SIGNAL
                    self.signals.append(f"no-need → {item}")

            elif strategy in PROSELF_STRATEGIES:
                self.proself_count += 1

            elif strategy in PROSOCIAL_STRATEGIES:
                self.prosocial_count += 1

        # Also apply weak signal from deal proposals mentioning quantities
        self._update_from_deal_hints(utterance_text)

        self._refresh_confidence()
        self._refresh_style()

    def _update_from_deal_hints(self, text: str) -> None:
        """Extract weak signals from explicit quantity mentions like '3 food'."""
        for item, canonical in [("food", "Food"), ("water", "Water"),
                                ("firewood", "Firewood")]:
            match = re.search(rf"(\d)\s*{item}", text.lower())
            if match:
                qty = int(match.group(1))
                if qty == 3:
                    self.priority_scores[canonical] += WEAK_SIGNAL
                    self.signals.append(f"wants {qty} {canonical}")
                elif qty == 0:
                    self.priority_scores[canonical] += NEGATIVE_SIGNAL * 0.5
                    self.signals.append(f"offers 0 {canonical}")

    def _refresh_confidence(self) -> None:
        total = sum(abs(v) for v in self.priority_scores.values())
        if total >= 6:
            self.confidence = "high"
        elif total >= 3:
            self.confidence = "medium"
        else:
            self.confidence = "low"

    def _refresh_style(self) -> None:
        total = self.prosocial_count + self.proself_count
        if total >= 3:
            self.style = (
                "prosocial" if self.prosocial_count > self.proself_count
                else "proself"
            )

    # ── Predictions ────────────────────────────────────────────────────

    def get_predicted_priorities(self) -> Dict[str, str]:
        """Return predicted priority mapping: {"High": item, ...}."""
        sorted_items = sorted(
            self.priority_scores.items(), key=lambda x: x[1], reverse=True,
        )
        return {
            "High": sorted_items[0][0],
            "Medium": sorted_items[1][0],
            "Low": sorted_items[2][0],
        }

    def get_accuracy(self, ground_truth_priorities: Dict[str, str]) -> float:
        """Fraction of priority levels correctly predicted (0, 0.33, 0.67, 1.0)."""
        predicted = self.get_predicted_priorities()
        correct = sum(
            1 for level in ("High", "Medium", "Low")
            if predicted[level] == ground_truth_priorities[level]
        )
        return correct / 3.0

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-friendly snapshot of the current model state."""
        return {
            "predicted_priorities": self.get_predicted_priorities(),
            "priority_scores": dict(self.priority_scores),
            "confidence": self.confidence,
            "style": self.style,
            "signals": list(self.signals),
        }

    def reset(self) -> None:
        """Clear all accumulated state."""
        self.__init__()  # type: ignore[misc]


# ── Batch evaluation against CaSiNo-Ann ────────────────────────────────────

DEAL_ACTIONS = {"Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away"}


def evaluate_opponent_model(
    dialogues: List[Dict[str, Any]],
    strategy_source: str = "gold",
    llm_client: Any = None,
    max_dialogues: Optional[int] = None,
) -> Dict[str, Any]:
    """Run opponent modeling on dialogues and measure accuracy.

    For each dialogue, simulate being mturk_agent_1 observing mturk_agent_2
    (and vice-versa), then compare predicted vs actual priorities.

    Args:
        dialogues: CaSiNo dialogue list.
        strategy_source: "gold" uses ground-truth annotations (only for
            annotated dialogues). "llm" uses classify_strategies + llm_client.
        llm_client: Required when strategy_source="llm".
        max_dialogues: Cap the number of dialogues.

    Returns:
        Dict with per-role accuracy, overall accuracy, confidence distribution,
        style accuracy.
    """
    if strategy_source == "llm" and llm_client is None:
        raise ValueError("llm_client required when strategy_source='llm'")

    if strategy_source == "gold":
        dialogues = [d for d in dialogues if d.get("annotations")]

    if max_dialogues:
        dialogues = dialogues[:max_dialogues]

    results: List[Dict[str, Any]] = []

    for dialogue in dialogues:
        chat_logs = dialogue["chat_logs"]
        pinfo = dialogue["participant_info"]
        annotations = dialogue.get("annotations", [])

        for observer_role, opponent_role in [
            ("mturk_agent_1", "mturk_agent_2"),
            ("mturk_agent_2", "mturk_agent_1"),
        ]:
            model = OpponentModel()
            ann_idx = 0

            for i, turn in enumerate(chat_logs):
                if turn["text"] in DEAL_ACTIONS:
                    continue

                is_opponent = turn["id"] == opponent_role

                if strategy_source == "gold" and ann_idx < len(annotations):
                    _, strat_str = annotations[ann_idx]
                    strategies = [s.strip() for s in strat_str.split(",") if s.strip()]
                    ann_idx += 1
                elif strategy_source == "llm":
                    from prompt_engineer.core.classify_strategy import (
                        classify_strategies,
                        format_context,
                    )
                    ctx = format_context(chat_logs, i)
                    strategies = classify_strategies(turn["text"], ctx, llm_client)
                else:
                    strategies = []

                if is_opponent:
                    model.update(turn["text"], strategies)

            gt_priorities = pinfo[opponent_role]["value2issue"]
            gt_svo = pinfo[opponent_role].get("personality", {}).get("svo")
            accuracy = model.get_accuracy(gt_priorities)
            style_correct = (
                model.style == gt_svo if model.style != "unknown" and gt_svo else None
            )

            results.append({
                "dialogue_id": dialogue["dialogue_id"],
                "observer": observer_role,
                "opponent": opponent_role,
                "accuracy": accuracy,
                "confidence": model.confidence,
                "predicted": model.get_predicted_priorities(),
                "actual": gt_priorities,
                "style_predicted": model.style,
                "style_actual": gt_svo,
                "style_correct": style_correct,
                "num_signals": len(model.signals),
            })

    accuracies = [r["accuracy"] for r in results]
    perfect = sum(1 for a in accuracies if a == 1.0)
    partial = sum(1 for a in accuracies if 0 < a < 1.0)
    wrong = sum(1 for a in accuracies if a == 0.0)

    conf_dist = defaultdict(int)
    for r in results:
        conf_dist[r["confidence"]] += 1

    style_results = [r for r in results if r["style_correct"] is not None]
    style_acc = (
        sum(1 for r in style_results if r["style_correct"]) / len(style_results)
        if style_results else None
    )

    high_conf = [r for r in results if r["confidence"] == "high"]
    high_conf_acc = (
        float(sum(r["accuracy"] for r in high_conf) / len(high_conf))
        if high_conf else None
    )

    return {
        "num_dialogues": len(dialogues),
        "num_evaluations": len(results),
        "avg_accuracy": float(sum(accuracies) / len(accuracies)) if accuracies else 0.0,
        "perfect_predictions": perfect,
        "partial_predictions": partial,
        "wrong_predictions": wrong,
        "confidence_distribution": dict(conf_dist),
        "high_confidence_accuracy": high_conf_acc,
        "style_accuracy": style_acc,
        "per_evaluation": results,
    }


def validate_opponent_model(
    dialogues: List[Dict[str, Any]],
    max_dialogues: Optional[int] = None,
) -> Dict[str, Any]:
    """Run opponent model on all dialogues using keyword-only signals.

    Unlike evaluate_opponent_model (which requires gold annotations or an LLM),
    this uses only extract_mentioned_items + heuristic strategy defaults to
    work on the full 1030-dialogue dataset.

    For each dialogue, observe each agent's own utterances to infer that
    agent's priorities, then compare to ground truth.

    Targets: >= 0.60 overall accuracy, >= 0.75 for high-confidence predictions.
    """
    if max_dialogues:
        dialogues = dialogues[:max_dialogues]

    results: List[Dict[str, Any]] = []

    for dialogue in dialogues:
        chat_logs = dialogue["chat_logs"]
        pinfo = dialogue["participant_info"]
        agents = list(pinfo.keys())

        for target_agent, observing_agent in [
            (agents[0], agents[1]),
            (agents[1], agents[0]),
        ]:
            model = OpponentModel()

            for turn in chat_logs:
                if turn["text"] in DEAL_ACTIONS:
                    continue
                if turn["id"] != target_agent:
                    continue

                items = extract_mentioned_items(turn["text"])
                text_lower = turn["text"].lower()

                strategies: List[str] = []
                if any(kw in text_lower for kw in
                       ["i need", "i want", "i'd like", "i would like",
                        "i require", "important to me", "my priority"]):
                    strategies.append("self-need")
                if any(kw in text_lower for kw in
                       ["my kid", "my child", "my dog", "my family",
                        "my grandm", "my wife", "my husband", "my partner",
                        "our group", "senior", "elderly", "baby", "toddler",
                        "my pet", "my cat", "my mom", "my dad", "my parent"]):
                    strategies.append("other-need")
                if any(kw in text_lower for kw in
                       ["don't need", "do not need", "don't want",
                        "can do without", "make do without", "no use for",
                        "not important", "least important"]):
                    strategies.append("no-need")
                if any(kw in text_lower for kw in
                       ["what do you", "what would you", "what are you",
                        "do you need", "do you want", "your priority",
                        "most interested", "least interested"]):
                    strategies.append("elicit-pref")
                if any(kw in text_lower for kw in
                       ["fair", "equal", "balanced", "even split",
                        "both of us", "leave me with nothing"]):
                    strategies.append("vouch-fair")
                if any(kw in text_lower for kw in
                       ["trade", "deal", "offer", "exchange", "how about",
                        "what if", "willing to give", "in return",
                        "work together", "let's try"]):
                    strategies.append("promote-coordination")
                if any(kw in text_lower for kw in
                       ["sorry to hear", "understand", "that's tough",
                        "hope you", "sounds difficult", "oh no"]):
                    strategies.append("showing-empathy")
                if any(kw in text_lower for kw in
                       ["you don't really", "you could buy",
                        "there might be a store", "do you have help",
                        "you probably don't"]):
                    strategies.append("uv-part")
                if any(kw in text_lower for kw in
                       ["hello", "hi!", "hey!", "how are you",
                        "nice to meet", "enjoy your trip", "have fun",
                        "good luck", "take care", "great chatting"]):
                    strategies.append("small-talk")

                if not strategies:
                    strategies = ["non-strategic"]

                model.update(turn["text"], strategies, items)

            gt = pinfo[target_agent]["value2issue"]
            gt_svo = pinfo[target_agent].get("personality", {}).get("svo")
            accuracy = model.get_accuracy(gt)
            style_correct = (
                model.style == gt_svo
                if model.style != "unknown" and gt_svo else None
            )

            results.append({
                "dialogue_id": dialogue["dialogue_id"],
                "target_agent": target_agent,
                "accuracy": accuracy,
                "confidence": model.confidence,
                "predicted": model.get_predicted_priorities(),
                "actual": gt,
                "style_predicted": model.style,
                "style_actual": gt_svo,
                "style_correct": style_correct,
                "num_signals": len(model.signals),
            })

    accuracies = [r["accuracy"] for r in results]
    conf_dist = defaultdict(int)
    for r in results:
        conf_dist[r["confidence"]] += 1

    high_conf = [r for r in results if r["confidence"] == "high"]
    high_conf_acc = (
        float(sum(r["accuracy"] for r in high_conf) / len(high_conf))
        if high_conf else None
    )

    style_results = [r for r in results if r["style_correct"] is not None]
    style_acc = (
        sum(1 for r in style_results if r["style_correct"]) / len(style_results)
        if style_results else None
    )

    avg_acc = float(sum(accuracies) / len(accuracies)) if accuracies else 0.0

    return {
        "num_dialogues": len(dialogues),
        "num_evaluations": len(results),
        "avg_accuracy": avg_acc,
        "meets_target_overall": avg_acc >= 0.60,
        "perfect_predictions": sum(1 for a in accuracies if a == 1.0),
        "partial_predictions": sum(1 for a in accuracies if 0 < a < 1.0),
        "wrong_predictions": sum(1 for a in accuracies if a == 0.0),
        "confidence_distribution": dict(conf_dist),
        "high_confidence_accuracy": high_conf_acc,
        "meets_target_high_conf": (
            high_conf_acc >= 0.75 if high_conf_acc is not None else False
        ),
        "style_accuracy": style_acc,
        "per_evaluation": results,
    }


def print_validation_results(results: Dict[str, Any]) -> None:
    """Pretty-print validation results with target comparison."""
    print(f"\nOpponent Model Validation ({results['num_dialogues']} dialogues, "
          f"{results['num_evaluations']} agent evaluations)\n")

    target_mark = "PASS" if results["meets_target_overall"] else "FAIL"
    print(f"{'Mean accuracy':.<40} {results['avg_accuracy']:.3f}  "
          f"(target >= 0.60: {target_mark})")
    print(f"{'  Perfect (3/3 correct)':.<40} {results['perfect_predictions']}")
    print(f"{'  Partial (1-2/3 correct)':.<40} {results['partial_predictions']}")
    print(f"{'  Wrong (0/3 correct)':.<40} {results['wrong_predictions']}")
    print()

    if results["high_confidence_accuracy"] is not None:
        hc_mark = "PASS" if results["meets_target_high_conf"] else "FAIL"
        print(f"{'High-confidence accuracy':.<40} "
              f"{results['high_confidence_accuracy']:.3f}  "
              f"(target >= 0.75: {hc_mark})")

    print()
    print("Confidence distribution:")
    for level in ("low", "medium", "high"):
        count = results["confidence_distribution"].get(level, 0)
        pct = count / results["num_evaluations"] * 100
        print(f"  {level:.<20} {count:>5d}  ({pct:.1f}%)")

    if results["style_accuracy"] is not None:
        print(f"\n{'Style prediction accuracy':.<40} {results['style_accuracy']:.3f}")


def print_opponent_model_results(results: Dict[str, Any]) -> None:
    """Pretty-print opponent model evaluation results."""
    print(f"\nOpponent Model Evaluation ({results['num_dialogues']} dialogues, "
          f"{results['num_evaluations']} observer-opponent pairs)\n")

    print(f"{'Priority prediction accuracy':.<40} {results['avg_accuracy']:.3f}")
    print(f"{'  Perfect (3/3 correct)':.<40} {results['perfect_predictions']}")
    print(f"{'  Partial (1-2/3 correct)':.<40} {results['partial_predictions']}")
    print(f"{'  Wrong (0/3 correct)':.<40} {results['wrong_predictions']}")
    print()

    print("Confidence distribution:")
    for level in ("low", "medium", "high"):
        count = results["confidence_distribution"].get(level, 0)
        print(f"  {level:.<20} {count}")

    if results["high_confidence_accuracy"] is not None:
        print(f"\n{'High-confidence accuracy':.<40} "
              f"{results['high_confidence_accuracy']:.3f}")

    if results["style_accuracy"] is not None:
        print(f"{'Style prediction accuracy':.<40} {results['style_accuracy']:.3f}")


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate / validate opponent model on CaSiNo data."
    )
    parser.add_argument(
        "--mode", choices=["evaluate", "validate"], default="validate",
        help="'evaluate' uses gold annotations (CaSiNo-Ann only). "
             "'validate' uses keyword heuristics (all dialogues).",
    )
    parser.add_argument(
        "--input", type=Path,
        default=None,
        help="Path to CaSiNo JSON (defaults based on mode).",
    )
    parser.add_argument("--max-dialogues", type=int, default=None)
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[2] / "CaSiNo" / "data"
    if args.input is None:
        args.input = (
            base / "casino_ann.json" if args.mode == "evaluate"
            else base / "casino.json"
        )

    with args.input.open() as f:
        data = json.load(f)

    if args.mode == "evaluate":
        results = evaluate_opponent_model(
            data, strategy_source="gold", max_dialogues=args.max_dialogues,
        )
        print_opponent_model_results(results)
    else:
        results = validate_opponent_model(data, max_dialogues=args.max_dialogues)
        print_validation_results(results)
