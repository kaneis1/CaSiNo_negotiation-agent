"""Hybrid Bayesian opponent model — full CaSiNo evaluation.

Mirrors prompt_engineer/scripts/compare_gold_standard.py in style:
constants up top, linear script body, no argparse. Edit the constants
below to change behavior; for finer-grained CLI control use
``python -m opponent_model.eval_run --help`` instead.

Run (either form works from the repo root):
    python -m opponent_model.scripts.run_hybrid_eval
    python opponent_model/scripts/run_hybrid_eval.py
"""

import json
import sys
from pathlib import Path

# Make ``import opponent_model`` work whether the script is launched as
# ``python -m`` (CWD is on sys.path) or ``python path/to/script.py`` (only
# the script's own dir is). Walks up to the repo root that owns the package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from opponent_model.cache import CachedLLM
from opponent_model.eval_run import make_hybrid_model_fn, run_eval
from prompt_engineer.llm.client import LlamaClient

# ── Config ────────────────────────────────────────────────────────────────

MODEL_ID = (
    "/sc/arion/scratch/cuiz02/hf_cache/transformers/"
    "models--meta-llama--Llama-3.3-70B-Instruct/snapshots/"
    "6f6073b423013f6a7d4d9f39144961bfbfbc386b"
)
DATA_PATH      = "CaSiNo/data/casino.json"
OUTPUT_DIR     = Path("opponent_model/results/hybrid_eval")
MAX_DIALOGUES  = None      # set to e.g. 20 for a smoke run; None = full 1030

# Bayesian update knobs (sweep these during calibration).
LIKELIHOOD_TEMPERATURE = 25.0
LIKELIHOOD_CLIP        = (-3.0, 3.0)

# Checkpointing / progress reporting.
CHECKPOINT_EVERY = 10
SUMMARY_EVERY    = 25
RESUME           = True    # pick up from OUTPUT_DIR/checkpoint.json if present

# ── Load data ─────────────────────────────────────────────────────────────

with open(DATA_PATH) as f:
    dialogues = json.load(f)
if MAX_DIALOGUES is not None:
    dialogues = dialogues[:MAX_DIALOGUES]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load model (from local snapshot) ─────────────────────────────────────-

raw_client = LlamaClient(
    model_id=MODEL_ID,
    temperature=0.0,        # greedy for cache-sound eval
    max_new_tokens=256,
)

# Disk-cached wrapper: re-runs with the same prompt are free.
client = CachedLLM(raw_client, cache_path=OUTPUT_DIR / "llm_cache.sqlite")

model_fn = make_hybrid_model_fn(
    client,
    likelihood_temperature=LIKELIHOOD_TEMPERATURE,
    likelihood_clip=LIKELIHOOD_CLIP,
    strict_likelihood=False,    # warn-and-fill on missing scores during prod runs
)

# ── Run evaluation (writes log, predictions.jsonl, checkpoint.json, summary.json) ──

config = {
    "model_id": MODEL_ID,
    "data": DATA_PATH,
    "n_dialogues": len(dialogues),
    "likelihood_temperature": LIKELIHOOD_TEMPERATURE,
    "likelihood_clip": list(LIKELIHOOD_CLIP),
}

run_eval(
    dialogues=dialogues,
    model_fn=model_fn,
    output_dir=OUTPUT_DIR,
    checkpoint_every=CHECKPOINT_EVERY,
    summary_every=SUMMARY_EVERY,
    max_k=5,
    cached_client=client,
    resume=RESUME,
    config=config,
)
