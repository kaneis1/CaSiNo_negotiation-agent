"""Build Day 7 distillation rows from quality-selected CaSiNo trajectories.

The Bayesian teacher supplies belief state (posterior over opponent priority
orderings) and two menus. The human selected speaker supplies the target
intent/content/utterance at speaker-burst granularity.

The epistemic menu uses an opponent-utility variance proxy:

    Var_theta[U_opp(pi | theta)]

High variance in opponent utility across priority hypotheses identifies offers
whose reception would best discriminate among the six possible opponent
orderings, without requiring an extra response-model call.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from opponent_model.turn_level_metrics import DEAL_ACTIONS
from sft_8b.menu import (
    ITEMS_COUNT,
    ScoredSplit,
    build_menu,
    ordering_to_priorities,
    points,
)
from sft_8b.posterior import N_ORDERINGS, ORDERINGS, entropy, get_posterior
from sft_8b.prompts import ITEMS

logger = logging.getLogger("sft_8b.build_distill_data")


STYLE_LAMBDA: Dict[str, float] = {
    "cooperative": 2.0,
    "balanced": 1.0,
    "competitive": 0.0,
}
STYLE_W: Dict[str, float] = {
    "cooperative": 0.2,
    "balanced": 0.5,
    "competitive": 0.8,
}
W_STYLE: Dict[str, str] = {str(v): k for k, v in STYLE_W.items()}

UTILITY_SCALE: Dict[str, int] = {
    "High": 5,
    "Medium": 4,
    "Low": 3,
    "items_per_issue": ITEMS_COUNT,
}

POSTERIOR_MODEL_NAME = "sft_8b_lora_posterior"
EPISTEMIC_MENU_METHOD = "opp_utility_variance_proxy"
EPISTEMIC_JUSTIFICATION = (
    "High variance in opponent utility across priority hypotheses identifies "
    "offers whose reception would best discriminate among the six possible "
    "opponent orderings, without requiring an extra response-model call."
)

DEFAULT_BASE_MODEL = (
    "/sc/arion/scratch/cuiz02/hf_cache/transformers/Meta-Llama-3.1-8B-Instruct"
)
DEFAULT_ADAPTER = "sft_8b/results/lora_run/lora_best"
DEFAULT_OUTPUT_DIR = Path("sft_8b/results/distill/day7")
DEFAULT_SEED = 2024

DISTILL_SYSTEM_PROMPT = """\
You imitate a CaSiNo negotiator conditioned on their own priorities, dialogue
history, inferred opponent-priority posterior, candidate offer menus, and style.
Reply with JSON only:
{"selected_intent": "...", "selected_content": null or {...}, "utterance": "..."}"""


@dataclass(frozen=True)
class Burst:
    start: int
    end: int  # exclusive
    turns: Tuple[Mapping[str, Any], ...]


def git_sha(short: bool = True) -> str:
    cmd = ["git", "rev-parse", "--short" if short else "HEAD"]
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--short"], text=True)
        return bool(out.strip())
    except Exception:
        return True


def parse_styles(raw: str) -> List[str]:
    styles: List[str] = []
    for tok in (x.strip() for x in raw.split(",") if x.strip()):
        if tok in STYLE_LAMBDA:
            styles.append(tok)
            continue
        if tok in W_STYLE:
            styles.append(W_STYLE[tok])
            continue
        raise ValueError(
            f"unknown style {tok!r}; use one of {sorted(STYLE_LAMBDA)} "
            f"or weights {sorted(W_STYLE)}"
        )
    return styles


def quality_path_for_style(style: str) -> Path:
    return Path(f"data/casino_train_w{STYLE_W[style]}.json")


def is_counter(history: Sequence[Mapping[str, Any]], speaker_role: str) -> bool:
    """True iff the most recent prior action is opponent Submit-Deal."""
    for turn in reversed(history):
        if turn.get("text") in DEAL_ACTIONS:
            return turn.get("text") == "Submit-Deal" and turn.get("id") != speaker_role
    return False


def iter_speaker_bursts(
    chat_logs: Sequence[Mapping[str, Any]],
    speaker_role: str,
) -> Iterable[Burst]:
    i = 0
    n = len(chat_logs)
    while i < n:
        if chat_logs[i].get("id") != speaker_role:
            i += 1
            continue
        j = i + 1
        while j < n and chat_logs[j].get("id") == speaker_role:
            j += 1
        yield Burst(start=i, end=j, turns=tuple(chat_logs[i:j]))
        i = j


def terminal_action(burst: Burst) -> Optional[Mapping[str, Any]]:
    action_turns = [t for t in burst.turns if t.get("text") in DEAL_ACTIONS]
    return action_turns[-1] if action_turns else None


def utterance_from_burst(burst: Burst) -> str:
    texts = [
        str(t.get("text", "")).strip()
        for t in burst.turns
        if t.get("text") and t.get("text") not in DEAL_ACTIONS
    ]
    return "\n".join(t for t in texts if t)


def coerce_counts(raw: Mapping[str, Any]) -> Dict[str, int]:
    return {item: int(raw.get(item, 0)) for item in ITEMS}


def submit_content(
    turn: Mapping[str, Any],
    *,
    speaker_role: str,
) -> Optional[Dict[str, Any]]:
    td = turn.get("task_data") or {}
    if "issue2youget" not in td or "issue2theyget" not in td:
        return None
    try:
        if turn.get("id") == speaker_role:
            self_counts = coerce_counts(td["issue2youget"])
            opp_counts = coerce_counts(td["issue2theyget"])
        else:
            self_counts = coerce_counts(td["issue2theyget"])
            opp_counts = coerce_counts(td["issue2youget"])
    except (TypeError, ValueError):
        return None
    return {
        "self_counts": self_counts,
        "opp_counts": opp_counts,
        "self_tuple": [self_counts[item] for item in ITEMS],
        "opp_tuple": [opp_counts[item] for item in ITEMS],
    }


def intent_and_content(
    burst: Burst,
    *,
    history: Sequence[Mapping[str, Any]],
    speaker_role: str,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    action = terminal_action(burst)
    if action is None:
        return "utter", None

    text = action.get("text")
    if text == "Submit-Deal":
        content = submit_content(action, speaker_role=speaker_role)
        intent = "counter" if is_counter(history, speaker_role) else "offer"
        return intent, content
    if text == "Accept-Deal":
        return "accept", None
    if text == "Reject-Deal":
        return "reject", None
    if text == "Walk-Away":
        return "walkaway", None
    return "utter", None


def n_opp_utterances_seen(
    history: Sequence[Mapping[str, Any]],
    *,
    opp_role: str,
) -> int:
    return sum(
        1
        for turn in history
        if turn.get("id") == opp_role
        and turn.get("text")
        and turn.get("text") not in DEAL_ACTIONS
    )


def format_priorities(value2issue: Mapping[str, str]) -> str:
    return "\n".join(
        f"{level}: {value2issue.get(level, '?')}"
        for level in ("High", "Medium", "Low")
    )


def format_reasons(value2reason: Mapping[str, str]) -> str:
    lines = [
        f"{level}: {str(value2reason.get(level, '')).strip()}"
        for level in ("High", "Medium", "Low")
        if value2reason.get(level)
    ]
    return "\n".join(lines) if lines else "(no reasons provided)"


def format_submit_for_history(
    turn: Mapping[str, Any],
    *,
    perspective: str,
) -> str:
    td = turn.get("task_data") or {}
    if "issue2youget" not in td or "issue2theyget" not in td:
        return "Submit-Deal"
    if turn.get("id") == perspective:
        self_counts = coerce_counts(td["issue2youget"])
        opp_counts = coerce_counts(td["issue2theyget"])
    else:
        self_counts = coerce_counts(td["issue2theyget"])
        opp_counts = coerce_counts(td["issue2youget"])
    return (
        "Submit-Deal "
        f"self=({', '.join(f'{it}:{self_counts[it]}' for it in ITEMS)}) "
        f"opp=({', '.join(f'{it}:{opp_counts[it]}' for it in ITEMS)})"
    )


def format_history(
    history: Sequence[Mapping[str, Any]],
    *,
    perspective: str,
) -> str:
    lines: List[str] = []
    for turn in history:
        text = str(turn.get("text", "")).strip()
        if not text:
            continue
        speaker = "Me" if turn.get("id") == perspective else "Opponent"
        if text == "Submit-Deal":
            text = format_submit_for_history(turn, perspective=perspective)
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines) if lines else "(conversation not yet started)"


def format_posterior(posterior: Sequence[float]) -> str:
    return "\n".join(
        f"p({' > '.join(ordering)})={float(prob):.4f}"
        for ordering, prob in zip(ORDERINGS, posterior)
    )


def split_key(counts: Mapping[str, int]) -> Tuple[int, int, int]:
    return tuple(int(counts[item]) for item in ITEMS)


def menu_entry(split: ScoredSplit, *, rank: int) -> Dict[str, Any]:
    return {
        "rank": rank,
        "self_counts": dict(split.self_counts),
        "opp_counts": dict(split.opp_counts),
        "self_tuple": [int(split.self_counts[item]) for item in ITEMS],
        "opp_tuple": [int(split.opp_counts[item]) for item in ITEMS],
        "u_self": int(split.u_self),
        "exp_u_opp": float(split.exp_u_opp),
        "score": float(split.score),
    }


def u_opp_values_for_split(
    opp_counts: Mapping[str, int],
) -> np.ndarray:
    return np.asarray(
        [
            points(opp_counts, ordering_to_priorities(ordering))
            for ordering in ORDERINGS
        ],
        dtype=np.float64,
    )


def build_epistemic_menu(
    posterior: Sequence[float],
    self_priorities: Mapping[str, str],
    *,
    exclude_self_tuples: Sequence[Tuple[int, int, int]],
    top_k: int,
) -> List[Dict[str, Any]]:
    posterior_arr = np.asarray(posterior, dtype=np.float64)
    posterior_arr = posterior_arr / posterior_arr.sum()
    excluded = set(exclude_self_tuples)

    entries: List[Dict[str, Any]] = []
    for f, w, fw in product(range(ITEMS_COUNT + 1), repeat=3):
        self_counts = {"Food": f, "Water": w, "Firewood": fw}
        key = split_key(self_counts)
        if key in excluded:
            continue
        opp_counts = {item: ITEMS_COUNT - self_counts[item] for item in ITEMS}
        u_self = points(self_counts, self_priorities)
        u_opp = u_opp_values_for_split(opp_counts)
        exp_u_opp = float(posterior_arr @ u_opp)
        var_u_opp = float(posterior_arr @ ((u_opp - exp_u_opp) ** 2))
        entries.append({
            "self_counts": self_counts,
            "opp_counts": opp_counts,
            "self_tuple": [int(self_counts[item]) for item in ITEMS],
            "opp_tuple": [int(opp_counts[item]) for item in ITEMS],
            "u_self": int(u_self),
            "exp_u_opp": exp_u_opp,
            "epistemic_score": var_u_opp,
            "variance_u_opp": var_u_opp,
            "u_opp_by_hypothesis": [float(x) for x in u_opp.tolist()],
        })

    entries.sort(
        key=lambda e: (
            -float(e["epistemic_score"]),
            -float(e["exp_u_opp"]),
            -int(e["u_self"]),
            tuple(e["self_tuple"]),
        )
    )
    top = entries[: int(top_k)]
    for i, entry in enumerate(top, 1):
        entry["rank"] = i
    return top


def format_menu_for_prompt(
    *,
    utility_menu: Sequence[Dict[str, Any]],
    epistemic_menu: Sequence[Dict[str, Any]],
    epistemic_method: str,
) -> str:
    util_lines = [
        (
            f"#{m['rank']} self={m['self_tuple']} opp={m['opp_tuple']} "
            f"U_self={m['u_self']} E_U_opp={m['exp_u_opp']:.2f} "
            f"score={m['score']:.2f}"
        )
        for m in utility_menu
    ]
    epi_lines = [
        (
            f"#{m['rank']} self={m['self_tuple']} opp={m['opp_tuple']} "
            f"U_self={m['u_self']} E_U_opp={m['exp_u_opp']:.2f} "
            f"Var_U_opp={m['epistemic_score']:.2f}"
        )
        for m in epistemic_menu
    ]
    return (
        '<utility_menu top_k="5">\n'
        + "\n".join(util_lines)
        + "\n</utility_menu>\n"
        + f'<epistemic_menu method="{epistemic_method}" top_k="3">\n'
        + "\n".join(epi_lines)
        + "\n</epistemic_menu>"
    )


def build_user_content(
    *,
    my_priorities: Mapping[str, str],
    my_reasons: Mapping[str, str],
    history: Sequence[Mapping[str, Any]],
    perspective: str,
    posterior: Sequence[float],
    utility_menu: Sequence[Dict[str, Any]],
    epistemic_menu: Sequence[Dict[str, Any]],
    style: str,
) -> str:
    return (
        "<self_priorities>\n"
        f"{format_priorities(my_priorities)}\n"
        "</self_priorities>\n\n"
        "<self_reasons>\n"
        f"{format_reasons(my_reasons)}\n"
        "</self_reasons>\n\n"
        "<history>\n"
        f"{format_history(history, perspective=perspective)}\n"
        "</history>\n\n"
        "<posterior>\n"
        f"{format_posterior(posterior)}\n"
        "</posterior>\n\n"
        "<menu>\n"
        f"{format_menu_for_prompt(utility_menu=utility_menu, epistemic_menu=epistemic_menu, epistemic_method=EPISTEMIC_MENU_METHOD)}\n"
        "</menu>\n\n"
        "<style>\n"
        f"{style}\n"
        "</style>"
    )


def validate_posterior(posterior: Sequence[float]) -> np.ndarray:
    arr = np.asarray(posterior, dtype=np.float64).flatten()
    if arr.shape != (N_ORDERINGS,):
        raise ValueError(f"posterior shape {arr.shape}, expected {(N_ORDERINGS,)}")
    if np.any(arr < 0):
        raise ValueError("posterior has negative mass")
    s = float(arr.sum())
    if not math.isfinite(s) or s <= 0:
        raise ValueError("posterior has non-positive mass")
    arr = arr / s
    if abs(float(arr.sum()) - 1.0) > 1e-6:
        raise ValueError("posterior did not normalize")
    return arr


def validate_row(row: Mapping[str, Any]) -> None:
    posterior = validate_posterior(row["posterior"])
    if posterior.shape != (N_ORDERINGS,):
        raise ValueError("posterior validation failed")

    user_content = row["messages"][1]["content"]
    if "<self_priorities>" not in user_content or "<self_reasons>" not in user_content:
        raise ValueError("prompt missing self priorities or reasons")
    if "<history>" not in user_content or "<posterior>" not in user_content:
        raise ValueError("prompt missing required sections")
    if "<menu>" not in user_content or "<style>" not in user_content:
        raise ValueError("prompt missing menu or style")

    if len(row["utility_menu"]) != 5:
        raise ValueError("utility menu length is not 5")
    if len(row["epistemic_menu"]) != 3:
        raise ValueError("epistemic menu length is not 3")
    all_menu_keys = [
        tuple(m["self_tuple"])
        for m in row["utility_menu"]
    ] + [
        tuple(m["self_tuple"])
        for m in row["epistemic_menu"]
    ]
    if len(all_menu_keys) != len(set(all_menu_keys)):
        raise ValueError("utility/epistemic menu entries are not distinct")

    intent = row["target"]["selected_intent"]
    content = row["target"]["selected_content"]
    if intent in {"accept", "reject", "walkaway", "utter"} and content is not None:
        raise ValueError(f"{intent} row has non-null selected_content")
    if intent in {"offer", "counter"} and content is None:
        raise ValueError(f"{intent} row has null selected_content")


def cache_key(dialogue_id: Any, speaker_role: str, burst_start_index: int) -> str:
    return json.dumps(
        [dialogue_id, speaker_role, int(burst_start_index)],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def read_cache(path: Path, metadata: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return cache

    with path.open("r", encoding="utf-8") as f:
        first = f.readline()
        if not first:
            return cache
        header = json.loads(first)
        if header.get("type") != "metadata":
            raise ValueError(f"{path} has no metadata header")
        expected = dict(metadata)
        found = dict(header.get("metadata") or {})
        for key, val in expected.items():
            if found.get(key) != val:
                raise ValueError(
                    f"posterior cache metadata mismatch for {key}: "
                    f"found {found.get(key)!r}, expected {val!r}"
                )
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("type") != "posterior":
                continue
            cache[str(obj["cache_key"])] = obj
    return cache


def write_cache_header(path: Path, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "metadata", "metadata": dict(metadata)}) + "\n")


def append_cache_entry(path: Path, entry: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()


def latency_stats(vals: Sequence[float]) -> Dict[str, Any]:
    if not vals:
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    ordered = sorted(float(v) for v in vals)
    idx95 = min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1)
    return {
        "n": len(ordered),
        "mean": statistics.mean(ordered),
        "p50": statistics.median(ordered),
        "p95": ordered[idx95],
    }


def make_bar_plot(
    counter: Counter,
    path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str = "count",
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("matplotlib unavailable; skipping %s (%s)", path, exc)
        return

    labels = list(counter.keys())
    values = [counter[k] for k in labels]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.7), 4))
    ax.bar([str(x) for x in labels], values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_entropy_plot(
    entropy_points: Mapping[str, Mapping[int, List[float]]],
    path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("matplotlib unavailable; skipping %s (%s)", path, exc)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for style in STYLE_LAMBDA:
        by_turn = entropy_points.get(style, {})
        if not by_turn:
            continue
        xs = sorted(by_turn)
        ys = [statistics.mean(by_turn[x]) for x in xs]
        ax.plot(xs, ys, marker="o", label=style)
    ax.set_title("Posterior entropy by opponent utterances seen")
    ax.set_xlabel("n_opp_utterances_seen")
    ax.set_ylabel("posterior entropy (bits)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_config(args: argparse.Namespace, *, model_commit: str) -> Dict[str, Any]:
    styles = parse_styles(args.styles)
    return {
        "styles": styles,
        "style_w": STYLE_W,
        "style_lambda": STYLE_LAMBDA,
        "utility_scale": UTILITY_SCALE,
        "posterior_model": POSTERIOR_MODEL_NAME,
        "model_commit": model_commit,
        "git_dirty": git_dirty(),
        "base_model": args.base_model,
        "adapter": args.adapter,
        "posterior_k": args.posterior_k,
        "posterior_temperature": args.posterior_temperature,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "utility_top_k": args.utility_top_k,
        "epistemic_top_k": args.epistemic_top_k,
        "epistemic_menu_method": EPISTEMIC_MENU_METHOD,
        "epistemic_menu_justification": EPISTEMIC_JUSTIFICATION,
        "burst_policy": (
            "Collapse consecutive same-speaker chat_logs entries; the last "
            "action in the burst determines intent and non-action texts are "
            "concatenated as utterance."
        ),
        "counter_offer_predicate": (
            "counter iff the most recent prior action tag from either speaker "
            "is the opponent's Submit-Deal"
        ),
    }


def get_or_compute_posterior(
    *,
    key: str,
    cache: Dict[str, Dict[str, Any]],
    cache_path: Path,
    cache_metadata: Mapping[str, Any],
    model_fn: Any,
    history: Sequence[Mapping[str, Any]],
    my_priorities: Mapping[str, str],
    my_reasons: Mapping[str, str],
    me_role: str,
    posterior_k: int,
    posterior_temperature: float,
) -> Tuple[np.ndarray, bool, Optional[float]]:
    if key in cache:
        posterior = validate_posterior(cache[key]["posterior"])
        return posterior, True, None

    t0 = time.time()
    posterior = get_posterior(
        dialogue_prefix=history,
        speaker_priorities=my_priorities,
        model_fn=model_fn,
        speaker_reasons=my_reasons,
        me_role=me_role,
        K=posterior_k,
        temperature=posterior_temperature,
    )
    elapsed = time.time() - t0
    posterior = validate_posterior(posterior)

    entry = {
        "type": "posterior",
        "cache_key": key,
        "posterior_model": cache_metadata["posterior_model"],
        "model_commit": cache_metadata["model_commit"],
        "posterior": [float(x) for x in posterior.tolist()],
        "latency_seconds": elapsed,
    }
    cache[key] = entry
    append_cache_entry(cache_path, entry)
    return posterior, False, elapsed


def process(
    args: argparse.Namespace,
) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    model_commit = git_sha(short=True)
    config = build_config(args, model_commit=model_commit)
    config_path = output_dir / "day7_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    cache_metadata = {
        "posterior_model": POSTERIOR_MODEL_NAME,
        "model_commit": model_commit,
        "base_model": args.base_model,
        "adapter": args.adapter,
        "posterior_k": args.posterior_k,
        "posterior_temperature": args.posterior_temperature,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    cache_path = output_dir / f"day7_posterior_cache.{model_commit}.jsonl"
    write_cache_header(cache_path, cache_metadata)
    cache = read_cache(cache_path, cache_metadata)
    cache_hits = 0
    cache_misses = 0
    latencies: List[float] = []

    from sft_8b.predict import SftModelFn

    model_fn = SftModelFn(
        base_model=args.base_model,
        adapter_path=args.adapter,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
    )

    distill_path = output_dir / "day7_distill.jsonl"
    failures_path = output_dir / "day7_failures.jsonl"
    summary_path = output_dir / "day7_summary.json"

    counts_by_style: Counter = Counter()
    counts_by_intent: Counter = Counter()
    counts_by_perspective: Counter = Counter()
    counts_by_turn_depth: Counter = Counter()
    counts_by_burst_size: Counter = Counter()
    failures: Counter = Counter()
    entropy_points: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    attempts = 0
    rows_written = 0

    with distill_path.open("w", encoding="utf-8") as out, failures_path.open(
        "w", encoding="utf-8"
    ) as ferr:
        for style in parse_styles(args.styles):
            path = quality_path_for_style(style)
            with path.open("r", encoding="utf-8") as f:
                trajectories = json.load(f)
            if args.max_trajectories is not None:
                trajectories = trajectories[: args.max_trajectories]

            for traj_i, dialogue in enumerate(trajectories):
                dialogue_id = dialogue.get("dialogue_id")
                meta = dialogue.get("_quality_meta") or {}
                speaker_role = meta.get("speaker_role")
                if speaker_role not in ("mturk_agent_1", "mturk_agent_2"):
                    failures["bad_speaker_role"] += 1
                    ferr.write(json.dumps({
                        "style": style,
                        "dialogue_id": dialogue_id,
                        "trajectory_index": traj_i,
                        "reason": "bad_speaker_role",
                        "speaker_role": speaker_role,
                    }) + "\n")
                    continue

                opp_role = (
                    "mturk_agent_2"
                    if speaker_role == "mturk_agent_1"
                    else "mturk_agent_1"
                )
                pinfo = dialogue.get("participant_info") or {}
                try:
                    my_info = pinfo[speaker_role]
                    my_priorities = dict(my_info["value2issue"])
                    my_reasons = dict(my_info.get("value2reason") or {})
                except Exception as exc:
                    failures["bad_participant_info"] += 1
                    ferr.write(json.dumps({
                        "style": style,
                        "dialogue_id": dialogue_id,
                        "perspective": speaker_role,
                        "reason": "bad_participant_info",
                        "error": repr(exc),
                    }) + "\n")
                    continue

                chat_logs = list(dialogue.get("chat_logs") or [])
                for burst in iter_speaker_bursts(chat_logs, speaker_role):
                    attempts += 1
                    history = chat_logs[: burst.start]
                    try:
                        intent, selected_content = intent_and_content(
                            burst,
                            history=history,
                            speaker_role=speaker_role,
                        )
                        if intent in {"offer", "counter"} and selected_content is None:
                            raise ValueError("Submit-Deal burst has malformed task_data")

                        key = cache_key(dialogue_id, speaker_role, burst.start)
                        posterior, hit, latency = get_or_compute_posterior(
                            key=key,
                            cache=cache,
                            cache_path=cache_path,
                            cache_metadata=cache_metadata,
                            model_fn=model_fn,
                            history=history,
                            my_priorities=my_priorities,
                            my_reasons=my_reasons,
                            me_role=speaker_role,
                            posterior_k=args.posterior_k,
                            posterior_temperature=args.posterior_temperature,
                        )
                        if hit:
                            cache_hits += 1
                        else:
                            cache_misses += 1
                            if latency is not None:
                                latencies.append(latency)

                        utility = build_menu(
                            posterior,
                            my_priorities,
                            lambda_=STYLE_LAMBDA[style],
                            top_k=args.utility_top_k,
                        )
                        utility_menu = [
                            menu_entry(split, rank=i)
                            for i, split in enumerate(utility, 1)
                        ]
                        epistemic_menu = build_epistemic_menu(
                            posterior,
                            my_priorities,
                            exclude_self_tuples=[
                                tuple(m["self_tuple"]) for m in utility_menu
                            ],
                            top_k=args.epistemic_top_k,
                        )
                        opp_seen = n_opp_utterances_seen(
                            history,
                            opp_role=opp_role,
                        )
                        target = {
                            "selected_intent": intent,
                            "selected_content": selected_content,
                            "utterance": utterance_from_burst(burst),
                        }
                        user_content = build_user_content(
                            my_priorities=my_priorities,
                            my_reasons=my_reasons,
                            history=history,
                            perspective=speaker_role,
                            posterior=posterior,
                            utility_menu=utility_menu,
                            epistemic_menu=epistemic_menu,
                            style=style,
                        )
                        assistant_content = json.dumps(target, ensure_ascii=False)
                        row = {
                            "messages": [
                                {"role": "system", "content": DISTILL_SYSTEM_PROMPT},
                                {"role": "user", "content": user_content},
                                {"role": "assistant", "content": assistant_content},
                            ],
                            "dialogue_id": dialogue_id,
                            "trajectory_index": traj_i,
                            "perspective": speaker_role,
                            "opp_role": opp_role,
                            "turn_index_start": burst.start,
                            "turn_index_end": burst.end - 1,
                            "burst_size": burst.end - burst.start,
                            "n_opp_utterances_seen": opp_seen,
                            "w": STYLE_W[style],
                            "style": style,
                            "lambda": STYLE_LAMBDA[style],
                            "posterior_model": POSTERIOR_MODEL_NAME,
                            "model_commit": model_commit,
                            "posterior_cache_key": key,
                            "posterior_cache_hit": hit,
                            "posterior": [float(x) for x in posterior.tolist()],
                            "posterior_entropy_bits": entropy(posterior),
                            "utility_scale": UTILITY_SCALE,
                            "utility_menu": utility_menu,
                            "epistemic_menu_method": EPISTEMIC_MENU_METHOD,
                            "epistemic_menu": epistemic_menu,
                            "target": target,
                            "quality_meta": meta,
                        }
                        validate_row(row)

                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        rows_written += 1
                        counts_by_style[style] += 1
                        counts_by_intent[intent] += 1
                        counts_by_perspective[speaker_role] += 1
                        counts_by_turn_depth[opp_seen] += 1
                        counts_by_burst_size[burst.end - burst.start] += 1
                        entropy_points[style][opp_seen].append(entropy(posterior))
                    except Exception as exc:
                        failures["row_failure"] += 1
                        ferr.write(json.dumps({
                            "style": style,
                            "dialogue_id": dialogue_id,
                            "perspective": speaker_role,
                            "turn_index_start": burst.start,
                            "turn_index_end": burst.end - 1,
                            "reason": "row_failure",
                            "error": repr(exc),
                        }, ensure_ascii=False) + "\n")

    failure_rate = (sum(failures.values()) / attempts) if attempts else 0.0
    if rows_written == 0:
        raise RuntimeError("no distillation rows were written")
    if failure_rate > args.max_failure_rate:
        raise RuntimeError(
            f"failure rate {failure_rate:.3f} exceeds {args.max_failure_rate:.3f}"
        )

    summary = {
        "output_dir": str(output_dir),
        "distill_jsonl": str(distill_path),
        "config_json": str(config_path),
        "failures_jsonl": str(failures_path),
        "posterior_cache_jsonl": str(cache_path),
        "n_attempted_bursts": attempts,
        "n_rows": rows_written,
        "failure_rate": failure_rate,
        "failures": dict(failures),
        "rows_by_style": dict(counts_by_style),
        "rows_by_intent": dict(counts_by_intent),
        "rows_by_perspective": dict(counts_by_perspective),
        "rows_by_n_opp_utterances_seen": dict(counts_by_turn_depth),
        "rows_by_burst_size": dict(counts_by_burst_size),
        "posterior_cache": {
            "path": str(cache_path),
            "hits": cache_hits,
            "misses": cache_misses,
            "hit_rate": cache_hits / (cache_hits + cache_misses)
            if (cache_hits + cache_misses)
            else None,
            "n_entries_loaded_or_written": len(cache),
        },
        "posterior_latency_seconds": latency_stats(latencies),
        "config": config,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plots_dir = output_dir / "plots"
    make_bar_plot(
        counts_by_intent,
        plots_dir / "intent_hist.png",
        title="Day 7 intent histogram",
        xlabel="intent",
    )
    make_bar_plot(
        counts_by_perspective,
        plots_dir / "perspective_split.png",
        title="Day 7 perspective split",
        xlabel="perspective",
    )
    make_bar_plot(
        counts_by_turn_depth,
        plots_dir / "turn_depth_hist.png",
        title="Day 7 turn depth histogram",
        xlabel="n_opp_utterances_seen",
    )
    make_entropy_plot(
        entropy_points,
        plots_dir / "posterior_entropy_by_turn.png",
    )

    return summary


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--styles",
        default="0.2,0.5,0.8",
        help="Comma-separated styles or weights. Example: 0.2,0.5,0.8",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-trajectories", type=int, default=None)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--adapter", default=DEFAULT_ADAPTER)
    p.add_argument("--posterior-k", type=int, default=16)
    p.add_argument("--posterior-temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--utility-top-k", type=int, default=5)
    p.add_argument("--epistemic-top-k", type=int, default=3)
    p.add_argument("--max-failure-rate", type=float, default=0.02)
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    np.random.seed(args.seed)
    summary = process(args)
    logger.info(
        "Wrote %d rows -> %s",
        summary["n_rows"],
        summary["distill_jsonl"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

