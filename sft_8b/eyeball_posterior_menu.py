"""Eyeball diagnostic: posterior + menu on held-out dialogues.

For each held-out dialogue and k=1..max_k opponent utterances, prints:
  * top-3 entries of the posterior over the 6 orderings
  * posterior entropy (should decrease as k grows)
  * top-5 menu at each λ ∈ {0.0, 1.0, 2.0}  (see menu.py for rationale)
  * the ground-truth opponent ordering (for sanity-checking belief convergence)

Runs the main (non-CV) SFT adapter by default, whose held-out set is
``CaSiNo/data/split/casino_test.json``.

Usage
-----
    # 5 dialogues, agent_1 perspective, k=1..5, K=16 samples per k:
    python -m sft_8b.eyeball_posterior_menu

    # pick a specific subset by dialogue_id:
    python -m sft_8b.eyeball_posterior_menu --dialogue-ids 100,101,102

    # quick CPU-free check (uses uniform posterior):
    python -m sft_8b.eyeball_posterior_menu --no-model
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from sft_8b.menu import ITEMS_COUNT, build_menu, format_menu
from sft_8b.posterior import (
    N_ORDERINGS,
    ORDERINGS,
    entropy,
    get_posterior,
)
from sft_8b.prompts import DEAL_ACTIONS, ITEMS

logger = logging.getLogger("sft_8b.eyeball_posterior_menu")


DEFAULT_BASE_MODEL = (
    "/sc/arion/scratch/cuiz02/hf_cache/transformers/Meta-Llama-3.1-8B-Instruct"
)
DEFAULT_ADAPTER = "sft_8b/results/lora_run/lora_best"
DEFAULT_DATA = "CaSiNo/data/split/casino_test.json"
DEFAULT_LAMBDAS = (0.0, 1.0, 2.0)


# ── Dialogue prefix helpers ────────────────────────────────────────────────


def _snapshot_at_k(
    chat_logs: Sequence[Mapping[str, Any]], *, me_role: str, k: int,
) -> List[Mapping[str, Any]]:
    """Return the prefix that lets the model see the k-th opponent utterance.

    Mirrors ``sft_8b.data.build_dialogue_rows``: filter out empty / deal-action
    turns, keep a running list, stop right after opp's k-th utterance.
    """
    opp_role = "mturk_agent_2" if me_role == "mturk_agent_1" else "mturk_agent_1"
    partial: List[Mapping[str, Any]] = []
    opp_count = 0
    for turn in chat_logs:
        text = turn.get("text", "")
        if not text or text in DEAL_ACTIONS:
            continue
        partial.append(turn)
        if turn.get("id") == opp_role:
            opp_count += 1
            if opp_count >= k:
                break
    return partial


def _truth_ordering(pinfo_role: Mapping[str, Any]) -> Optional[Tuple[str, str, str]]:
    v2i = pinfo_role.get("value2issue") or {}
    if set(v2i.keys()) >= {"High", "Medium", "Low"}:
        return (v2i["High"], v2i["Medium"], v2i["Low"])
    return None


def _describe_posterior(p: np.ndarray, *, top: int = 3) -> str:
    order = np.argsort(-p)
    parts = []
    for i in order[:top]:
        if p[i] <= 1e-9:
            continue
        ordering = ORDERINGS[i]
        parts.append(
            f"[{ordering[0][:3]}>{ordering[1][:3]}>{ordering[2][:3]}] {p[i]:.2f}"
        )
    return "  ".join(parts) if parts else "(all zero)"


# ── Driver ─────────────────────────────────────────────────────────────────


def run_eyeball(
    *,
    dialogues: Sequence[Mapping[str, Any]],
    model_fn: Any,              # SftModelFn or None (then uniform)
    me_role: str,
    max_k: int,
    K: int,
    temperature: float,
    lambdas: Sequence[float],
    top_k_menu: int,
) -> None:
    overall_t0 = time.time()

    for dnum, dialogue in enumerate(dialogues, 1):
        did = dialogue.get("dialogue_id")
        pinfo = dialogue.get("participant_info") or {}
        me_info = pinfo.get(me_role) or {}
        opp_role = "mturk_agent_2" if me_role == "mturk_agent_1" else "mturk_agent_1"
        opp_info = pinfo.get(opp_role) or {}

        my_v2i = me_info.get("value2issue") or {}
        my_reasons = me_info.get("value2reason") or {}
        truth = _truth_ordering(opp_info)

        if not (set(my_v2i.keys()) >= {"High", "Medium", "Low"}):
            logger.warning("[%d/%d] dlg %s: missing speaker priorities, skipping",
                           dnum, len(dialogues), did)
            continue

        my_self_priorities = {k: my_v2i[k] for k in ("High", "Medium", "Low")}

        print("\n" + "=" * 78)
        print(f"Dialogue {did}   (me_role={me_role}, perspective #{dnum}/{len(dialogues)})")
        print(f"  my priorities: High={my_v2i['High']}  "
              f"Medium={my_v2i['Medium']}  Low={my_v2i['Low']}")
        if truth is not None:
            print(f"  GROUND-TRUTH opponent: High={truth[0]}  "
                  f"Medium={truth[1]}  Low={truth[2]}")
        else:
            print("  GROUND-TRUTH opponent: (missing)")
        print("=" * 78)

        for k in range(1, max_k + 1):
            partial = _snapshot_at_k(
                dialogue.get("chat_logs") or [], me_role=me_role, k=k,
            )
            if partial and sum(1 for t in partial if t.get("id") == opp_role) < k:
                print(f"\n[k={k}] (dialogue ended before {k} opponent utterances; stopping)")
                break

            t0 = time.time()
            if model_fn is None:
                p = np.full(N_ORDERINGS, 1.0 / N_ORDERINGS)
                elapsed = 0.0
            else:
                p = get_posterior(
                    dialogue_prefix=partial,
                    speaker_priorities=my_self_priorities,
                    model_fn=model_fn,
                    speaker_reasons=my_reasons,
                    me_role=me_role,
                    K=K,
                    temperature=temperature,
                )
                elapsed = time.time() - t0

            # Posterior summary line
            h = entropy(p)
            truth_mass = float(p[ORDERINGS.index(truth)]) if truth in ORDERINGS else float("nan")
            print(
                f"\n[k={k}]   entropy={h:.3f} bits   "
                f"truth_mass={truth_mass:.2f}   "
                f"K={K}   elapsed={elapsed:.1f}s"
            )
            print(f"  top-3 posterior: {_describe_posterior(p)}")

            # Menu at each λ
            for lam in lambdas:
                menu = build_menu(p, my_self_priorities, lambda_=lam, top_k=top_k_menu)
                print(f"  λ={lam}  top-{top_k_menu} menu:")
                print(format_menu(menu, indent="    "))

    print("\n" + "─" * 78)
    print(f"Total elapsed: {time.time() - overall_t0:.1f}s")


# ── CLI ────────────────────────────────────────────────────────────────────


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(DEFAULT_DATA))
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER,
                        help="Path to LoRA adapter (or empty string for zero-shot).")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--n-dialogues", type=int, default=5)
    parser.add_argument("--dialogue-ids",
                        help="Comma-separated dialogue_ids to use instead of first N.")
    parser.add_argument("--me-role", default="mturk_agent_1",
                        choices=("mturk_agent_1", "mturk_agent_2"))
    parser.add_argument("--max-k", type=int, default=5)
    parser.add_argument("--K", type=int, default=16,
                        help="Samples per posterior (default 16).")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k-menu", type=int, default=5)
    parser.add_argument(
        "--lambdas", default=",".join(str(x) for x in DEFAULT_LAMBDAS),
        help=f"Comma-separated λ values (default {DEFAULT_LAMBDAS}). "
             f"See sft_8b/menu.py for the re-tuning rationale.",
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--no-model", action="store_true",
                        help="Use a uniform posterior (no model load). "
                             "Useful for validating the menu side of the pipeline "
                             "on CPU.")
    parser.add_argument("--log-level", default="WARNING")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    with args.data.open() as f:
        dialogues = json.load(f)

    if args.dialogue_ids:
        wanted = {int(x) if x.lstrip("-").isdigit() else x
                  for x in args.dialogue_ids.split(",")}
        dialogues = [d for d in dialogues if d.get("dialogue_id") in wanted]
        logger.info("Filtered to %d dialogue(s) by id.", len(dialogues))
    else:
        if args.shuffle:
            random.Random(args.seed).shuffle(dialogues)
        dialogues = dialogues[: args.n_dialogues]

    lambdas = tuple(float(x) for x in args.lambdas.split(","))

    if args.no_model:
        print("*** --no-model: posterior will be uniform ***")
        model_fn = None
    else:
        from sft_8b.predict import SftModelFn
        t0 = time.time()
        model_fn = SftModelFn(
            base_model=args.base_model,
            adapter_path=args.adapter if args.adapter else None,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(f"Model loaded in {time.time() - t0:.1f}s "
              f"(base={args.base_model}, adapter={args.adapter or '<none>'})")

    run_eyeball(
        dialogues=dialogues,
        model_fn=model_fn,
        me_role=args.me_role,
        max_k=args.max_k,
        K=args.K,
        temperature=args.temperature,
        lambdas=lambdas,
        top_k_menu=args.top_k_menu,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
