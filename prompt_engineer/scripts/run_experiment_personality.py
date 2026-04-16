"""Experiment B — personality-aware (strategy definitions + Big-Five primer).

Same evaluation as run_experiment_baseline.py but the system prompt includes
a Big-Five negotiation primer and each utterance is classified with the
speaker's and opponent's personality scores injected into the prompt.

Usage:
    python -m prompt_engineer.scripts.run_experiment_personality
    python -m prompt_engineer.scripts.run_experiment_personality --max-dialogues 10
"""

import json
from prompt_engineer.llm.client import LlamaClient
from prompt_engineer.core.classify_strategy_personality import (
    build_system_prompt_with_personality,
    benchmark,
)
from prompt_engineer.evaluation.evaluate_classifier import print_results

MODEL_ID = (
    "/sc/arion/scratch/cuiz02/hf_cache/transformers/"
    "models--meta-llama--Llama-3.3-70B-Instruct/snapshots/"
    "6f6073b423013f6a7d4d9f39144961bfbfbc386b"
)
DATA_PATH    = "CaSiNo/data/casino_ann.json"
OUTPUT_PATH  = "prompt_engineer/results/personality_results.json"
MAX_DIALOGUES = None   # set to an int (e.g. 10) to test on a subset

# ── Load data ──────────────────────────────────────────────────────────────

with open(DATA_PATH) as f:
    dialogues = json.load(f)

# ── Load model ─────────────────────────────────────────────────────────────

client = LlamaClient(
    model_id=MODEL_ID,
    temperature=0.0,
    max_new_tokens=64,          # two short lines: STRATEGIES + PREFERENCES
    system_prompt=build_system_prompt_with_personality(),
)

# ── Run benchmark ──────────────────────────────────────────────────────────

results = benchmark(dialogues, client, max_dialogues=MAX_DIALOGUES)

# ── Print & save ───────────────────────────────────────────────────────────

print_results(results, experiment_name="With Personality (Big-Five + SVO)")

import pathlib
pathlib.Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved → {OUTPUT_PATH}")
