#!/usr/bin/env python3
"""Generate SFT training data from CaSiNo train dialogues.

Runs the 70B model over every utterance in casino_train.json and saves
each (system, user, assistant) triple as one JSONL line — ready to plug
into TRL / LLaMA-Factory / any standard SFT trainer.

Features
--------
- Checkpointing: completed dialogue IDs are written to a checkpoint file
  after every dialogue.  Re-running the script skips already-done dialogues,
  so a crash loses at most one dialogue worth of work.
- Progress bar: shows dialogue- and utterance-level progress.
- Two modes (set USE_PERSONALITY = True/False):
    False → build_system_prompt()           (strategy definitions only)
    True  → build_system_prompt_with_personality()  (+ Big-Five primer,
             speaker/opponent scores injected per utterance)

Output format (one JSON object per line)
-----------------------------------------
{
  "dialogue_id":  int,
  "utt_index":    int,          # position among non-deal utterances
  "speaker_id":   str,
  "messages": [
    {"role": "system",    "content": "<system prompt>"},
    {"role": "user",      "content": "<context + utterance>"},
    {"role": "assistant", "content": "STRATEGIES: ...\nOPPONENT_PREFERENCES: ..."}
  ]
}

Usage
-----
    # quick smoke test (5 dialogues)
    python -m prompt_engineer.scripts.generate_sft_data --max-dialogues 5

    # full run
    python -m prompt_engineer.scripts.generate_sft_data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Config (edit here or override via CLI args) ────────────────────────────

MODEL_ID = (
    "/sc/arion/scratch/cuiz02/hf_cache/transformers/"
    "models--meta-llama--Llama-3.3-70B-Instruct/snapshots/"
    "6f6073b423013f6a7d4d9f39144961bfbfbc386b"
)
TRAIN_DATA_PATH  = "CaSiNo/data/split/casino_train.json"
OUTPUT_PATH      = "prompt_engineer/results/sft_train_data.jsonl"
CHECKPOINT_PATH  = "prompt_engineer/results/sft_checkpoint.json"
USE_PERSONALITY  = False   # set True to inject Big-Five scores
MAX_NEW_TOKENS   = 64      # two short lines: STRATEGIES + PREFERENCES
MAX_DIALOGUES    = None    # None = all 900

DEAL_ACTIONS = {"Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away"}

# ── Imports (deferred so argparse works without torch available) ───────────

from prompt_engineer.core.classify_strategy import (
    build_system_prompt,
    build_classification_prompt,
    format_context,
)
from prompt_engineer.core.classify_strategy_personality import (
    build_system_prompt_with_personality,
    build_classification_prompt_with_personality,
    _format_big_five,
)

# ── Checkpoint helpers ─────────────────────────────────────────────────────


def load_checkpoint(path: str) -> set:
    """Return set of already-completed dialogue IDs."""
    if os.path.exists(path):
        with open(path) as f:
            return set(json.load(f))
    return set()


def save_checkpoint(path: str, completed: set) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(sorted(completed), f)


# ── Personality helper ─────────────────────────────────────────────────────


def _get_personality(participant_info: Dict, agent_id: Optional[str]):
    if not agent_id or agent_id not in participant_info:
        return None, None
    p = participant_info[agent_id].get("personality", {})
    return p.get("big-five"), p.get("svo")


# ── Per-dialogue processor ─────────────────────────────────────────────────


def process_dialogue(
    dialogue: Dict[str, Any],
    llm_client: LlamaClient,
    system_prompt: str,
    use_personality: bool,
) -> List[Dict[str, Any]]:
    """Run the model over every non-deal utterance in one dialogue.

    Returns a list of SFT example dicts (one per utterance).
    """
    chat_logs        = dialogue["chat_logs"]
    participant_info = dialogue.get("participant_info", {})
    dialogue_id      = dialogue.get("dialogue_id")

    # Identify player (first speaker) and opponent (second speaker)
    speakers_seen: List[str] = []
    for turn in chat_logs:
        if turn["id"] not in speakers_seen:
            speakers_seen.append(turn["id"])
        if len(speakers_seen) == 2:
            break
    player_id   = speakers_seen[0] if speakers_seen else None
    opponent_id = speakers_seen[1] if len(speakers_seen) > 1 else None

    examples: List[Dict[str, Any]] = []
    utt_index = 0

    for i, turn in enumerate(chat_logs):
        if turn["text"] in DEAL_ACTIONS:
            continue

        context = format_context(chat_logs, i, window=5)

        # Build user prompt
        if use_personality:
            speaker_id = turn["id"]
            if speaker_id == player_id:
                spk_bf, spk_svo = _get_personality(participant_info, player_id)
                opp_bf, opp_svo = _get_personality(participant_info, opponent_id)
            else:
                spk_bf, spk_svo = _get_personality(participant_info, opponent_id)
                opp_bf, opp_svo = _get_personality(participant_info, player_id)
            user_prompt = build_classification_prompt_with_personality(
                turn["text"], context,
                spk_bf, spk_svo, opp_bf, opp_svo,
            )
        else:
            user_prompt = build_classification_prompt(turn["text"], context)

        # Call model
        assistant_response = llm_client.generate(user_prompt)

        examples.append({
            "dialogue_id": dialogue_id,
            "utt_index":   utt_index,
            "speaker_id":  turn["id"],
            "messages": [
                {"role": "system",    "content": system_prompt},
                {"role": "user",      "content": user_prompt},
                {"role": "assistant", "content": assistant_response.strip()},
            ],
        })
        utt_index += 1

    return examples


# ── Main ───────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    # ── Load data ──────────────────────────────────────────────────────
    print(f"Loading training data from {args.data} …")
    with open(args.data) as f:
        dialogues = json.load(f)
    if args.max_dialogues:
        dialogues = dialogues[: args.max_dialogues]
    print(f"  {len(dialogues)} dialogues to process")

    # ── Restore checkpoint ─────────────────────────────────────────────
    completed = load_checkpoint(args.checkpoint)
    remaining = [d for d in dialogues if d.get("dialogue_id") not in completed]
    if completed:
        print(f"  Checkpoint found — {len(completed)} done, "
              f"{len(remaining)} remaining")

    if not remaining:
        print("Nothing left to process. Exiting.")
        return

    # ── Build system prompt ────────────────────────────────────────────
    system_prompt = (
        build_system_prompt_with_personality()
        if args.personality
        else build_system_prompt()
    )
    mode_label = "with_personality" if args.personality else "baseline"
    print(f"  Mode: {mode_label}")

    # ── Load model ─────────────────────────────────────────────────────
    from prompt_engineer.llm.client import LlamaClient
    print(f"\nLoading model: {args.model_id}")
    llm = LlamaClient(
        model_id=args.model_id,
        temperature=0.0,
        max_new_tokens=args.max_new_tokens,
        system_prompt=system_prompt,
    )
    print("Model ready.\n")

    # ── Output file (append mode — safe to resume) ─────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_file = open(args.output, "a")

    total_examples = 0

    try:
        for d_idx, dialogue in enumerate(remaining):
            dialogue_id = dialogue.get("dialogue_id", d_idx)
            n_utts = sum(
                1 for t in dialogue["chat_logs"] if t["text"] not in DEAL_ACTIONS
            )
            print(f"[{d_idx + 1}/{len(remaining)}] dialogue {dialogue_id} "
                  f"({n_utts} utterances) …", end=" ", flush=True)

            examples = process_dialogue(
                dialogue, llm, system_prompt, args.personality
            )

            for ex in examples:
                out_file.write(json.dumps(ex, ensure_ascii=False) + "\n")
            out_file.flush()

            completed.add(dialogue_id)
            save_checkpoint(args.checkpoint, completed)
            total_examples += len(examples)

            print(f"done  [{total_examples} examples so far]")

    except KeyboardInterrupt:
        print("\n\nInterrupted. Progress saved to checkpoint.")
    finally:
        out_file.close()

    print(f"\nFinished.")
    print(f"  Total examples written : {total_examples}")
    print(f"  Output file            : {args.output}")
    print(f"  Checkpoint             : {args.checkpoint}")


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SFT training data from CaSiNo train dialogues."
    )
    parser.add_argument("--data",         default=TRAIN_DATA_PATH)
    parser.add_argument("--model-id",     default=MODEL_ID)
    parser.add_argument("--output",       default=OUTPUT_PATH)
    parser.add_argument("--checkpoint",   default=CHECKPOINT_PATH)
    parser.add_argument("--personality",  action="store_true", default=USE_PERSONALITY,
                        help="Inject Big-Five scores into prompts")
    parser.add_argument("--max-dialogues", type=int, default=MAX_DIALOGUES,
                        help="Process only first N dialogues (default: all)")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()
    main(args)
