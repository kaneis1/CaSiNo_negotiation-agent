"""Build LoRA SFT data for ablation variants.

The output is regular chat JSONL consumed by ``sft_8b.train``.  Variants:

* ``direct_posterior`` / ``direct_posterior_groundtruth``:
  context -> smoothed one-hot posterior.
* ``direct_posterior_teacher``: context -> Day-7 teacher MC posterior.
* ``action_only``: no posterior tag in the student target.
* ``map_only``: MAP ordering instead of a six-way posterior.
* ``reversed``: utterance/action tags before posterior.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from casino_belief.diagnostics.ablation.ablation import DIRECT_POSTERIOR_SYSTEM, direct_posterior_prompt
from casino_belief.training.build_day8_sft_data import stable_eval_split
from casino_belief.training.build_day8_sft_data import compute_repeat_map, expanded_rows
from casino_belief.belief.posterior import ORDERINGS
from casino_belief.student.student_prompts import (
    STUDENT_SYSTEM_PROMPT,
    build_student_target,
    build_student_user_prompt,
    extract_tagged_section,
    format_posterior,
)

logger = logging.getLogger("casino_belief.training.build_ablation_sft_data")

DEFAULT_SOURCE_JSONL = Path("artifacts/training_metadata/distill/day7/day7_distill.jsonl")
DEFAULT_DIALOGUES = Path("data/casino/casino_train.json")
DEFAULT_OUTPUT_ROOT = Path("artifacts/training_metadata/ablation_sft_data")

VARIANTS = {
    "direct_posterior",
    "direct_posterior_groundtruth",
    "direct_posterior_teacher",
    "action_only",
    "map_only",
    "reversed",
}

ACTION_ONLY_SYSTEM_PROMPT = """\
You are a CaSiNo negotiation policy model.

You will be given one speaker's perspective: a style token, that speaker's
own priorities and reasons, and the dialogue history so far.

Predict the speaker's next move. Reply with the three tagged fields below, in
exactly this order and with no extra prose:

<selected_intent>
submit|accept|reject|walkaway|utter
</selected_intent>
<selected_content>
null or a JSON object
</selected_content>
<utterance>
...
</utterance>
"""

MAP_ONLY_SYSTEM_PROMPT = """\
You are a CaSiNo negotiation policy model.

Reply with the four tagged fields below, in exactly this order and with no
extra prose. The <posterior> field must contain only the MAP opponent ordering
as "MAP: Food > Water > Firewood".

<posterior>
MAP: ...
</posterior>
<selected_intent>
submit|accept|reject|walkaway|utter
</selected_intent>
<selected_content>
null or a JSON object
</selected_content>
<utterance>
...
</utterance>
"""

REVERSED_SYSTEM_PROMPT = """\
You are a CaSiNo negotiation policy model.

You will be given one speaker's perspective: a style token, that speaker's
own priorities and reasons, and the dialogue history so far.

Predict the speaker's next move and then their current belief state. Reply
with the four tagged fields below, in exactly this order and with no extra
prose:

<utterance>
...
</utterance>
<selected_intent>
submit|accept|reject|walkaway|utter
</selected_intent>
<selected_content>
null or a JSON object
</selected_content>
<posterior>
...
</posterior>

The posterior must contain exactly six lines, one per ordering, formatted
as p(Food > Water > Firewood)=0.1234. Use JSON null for selected_content
unless the intent is submit.
"""


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"no rows found in {path}")
    return rows


def _dialogue_lookup(path: Path) -> Dict[str, Mapping[str, Any]]:
    data = json.load(path.open())
    return {str(d.get("dialogue_id")): d for d in data}


def _true_posterior(row: Mapping[str, Any], dialogues: Mapping[str, Mapping[str, Any]]) -> List[float]:
    d = dialogues[str(row["dialogue_id"])]
    opp = row["opp_role"]
    pri = d["participant_info"][opp]["value2issue"]
    true = (pri["High"], pri["Medium"], pri["Low"])
    out = [0.02] * len(ORDERINGS)
    out[ORDERINGS.index(true)] = 0.90
    return out


def _teacher_posterior(row: Mapping[str, Any]) -> List[float]:
    vals = [float(x) for x in row["posterior"]]
    s = sum(max(0.0, x) for x in vals)
    if s <= 0:
        return [1.0 / len(vals)] * len(vals)
    return [max(0.0, x) / s for x in vals]


def _source_user_sections(row: Mapping[str, Any]) -> Dict[str, str]:
    user = str(row["messages"][1]["content"])
    return {
        "self_priorities": extract_tagged_section(user, "self_priorities"),
        "self_reasons": extract_tagged_section(user, "self_reasons"),
        "history": extract_tagged_section(user, "history"),
    }


def _student_user(row: Mapping[str, Any]) -> str:
    sec = _source_user_sections(row)
    return build_student_user_prompt(
        self_priorities=sec["self_priorities"],
        self_reasons=sec["self_reasons"],
        history=sec["history"],
        style=str(row["style"]),
    )


def _direct_user(row: Mapping[str, Any]) -> str:
    sec = _source_user_sections(row)
    return direct_posterior_prompt(
        evidence=sec["history"],
        my_priorities=_parse_priorities(sec["self_priorities"]),
        my_reasons=_parse_reasons(sec["self_reasons"]),
    )


def _parse_priorities(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            if k in {"High", "Medium", "Low"}:
                out[k] = v.strip()
    return out


def _parse_reasons(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            if k in {"High", "Medium", "Low"}:
                out[k] = v.strip()
    return out


def _content_text(content: Optional[Mapping[str, Any]]) -> str:
    if content is None:
        return "null"
    return json.dumps(content, ensure_ascii=False, separators=(",", ":"))


def _map_target(row: Mapping[str, Any], posterior: Sequence[float]) -> str:
    idx = max(range(len(posterior)), key=lambda i: posterior[i])
    return "MAP: " + " > ".join(ORDERINGS[idx])


def _build_messages(
    row: Mapping[str, Any],
    *,
    variant: str,
    dialogues: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    target = row["target"]
    if variant.startswith("direct_posterior"):
        posterior = (
            _true_posterior(row, dialogues)
            if variant in {"direct_posterior", "direct_posterior_groundtruth"}
            else _teacher_posterior(row)
        )
        assistant = "<posterior>\n" + format_posterior(posterior, ORDERINGS) + "\n</posterior>"
        messages = [
            {"role": "system", "content": DIRECT_POSTERIOR_SYSTEM},
            {"role": "user", "content": _direct_user(row)},
            {"role": "assistant", "content": assistant},
        ]
        selected_intent = "posterior"
    else:
        user = _student_user(row)
        intent = str(target["selected_intent"])
        content = target.get("selected_content")
        utterance = str(target.get("utterance", ""))
        posterior = _teacher_posterior(row)
        if variant == "action_only":
            system = ACTION_ONLY_SYSTEM_PROMPT
            assistant = (
                "<selected_intent>\n"
                f"{intent}\n"
                "</selected_intent>\n"
                "<selected_content>\n"
                f"{_content_text(content)}\n"
                "</selected_content>\n"
                "<utterance>\n"
                f"{utterance}\n"
                "</utterance>"
            )
        elif variant == "map_only":
            system = MAP_ONLY_SYSTEM_PROMPT
            assistant = (
                "<posterior>\n"
                f"{_map_target(row, posterior)}\n"
                "</posterior>\n"
                "<selected_intent>\n"
                f"{intent}\n"
                "</selected_intent>\n"
                "<selected_content>\n"
                f"{_content_text(content)}\n"
                "</selected_content>\n"
                "<utterance>\n"
                f"{utterance}\n"
                "</utterance>"
            )
        elif variant == "reversed":
            system = REVERSED_SYSTEM_PROMPT
            assistant = (
                "<utterance>\n"
                f"{utterance}\n"
                "</utterance>\n"
                "<selected_intent>\n"
                f"{intent}\n"
                "</selected_intent>\n"
                "<selected_content>\n"
                f"{_content_text(content)}\n"
                "</selected_content>\n"
                "<posterior>\n"
                f"{format_posterior(posterior, ORDERINGS)}\n"
                "</posterior>"
            )
        else:
            assistant = build_student_target(
                posterior=posterior,
                orderings=ORDERINGS,
                selected_intent=intent,
                selected_content=content,
                utterance=utterance,
            )
            system = STUDENT_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        selected_intent = str(target["selected_intent"])

    return {
        "messages": messages,
        "dialogue_id": row["dialogue_id"],
        "perspective": row["perspective"],
        "style": row.get("style"),
        "selected_intent": selected_intent,
        "source_variant": variant,
        "source_index": row.get("source_index"),
    }


def build_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = _load_jsonl(Path(args.input_jsonl))
    dialogues = _dialogue_lookup(Path(args.dialogues))

    train_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    counts = {"train": Counter(), "eval": Counter()}
    for idx, row in enumerate(source_rows):
        row = dict(row)
        row["source_index"] = idx
        split = stable_eval_split(
            row["dialogue_id"],
            seed=args.seed,
            eval_fraction=args.eval_fraction,
        )
        out = _build_messages(row, variant=args.variant, dialogues=dialogues)
        out["split"] = split
        if split == "eval":
            eval_rows.append(out)
        else:
            train_rows.append(out)
        counts[split][out["selected_intent"]] += 1

    repeat_map = (
        compute_repeat_map(
            counts["train"],
            mode=args.intent_balance_mode,
            max_repeat=args.max_intent_repeat,
        )
        if not args.variant.startswith("direct_posterior")
        else {"posterior": 1}
    )
    train_rows_expanded = list(expanded_rows(train_rows, repeat_map))

    train_path = output_dir / f"{args.variant}_train_rows.jsonl"
    eval_path = output_dir / f"{args.variant}_eval_rows.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for row in train_rows_expanded:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with eval_path.open("w", encoding="utf-8") as f:
        for row in eval_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "variant": args.variant,
        "input_jsonl": str(args.input_jsonl),
        "dialogues": str(args.dialogues),
        "train_jsonl": str(train_path),
        "eval_jsonl": str(eval_path),
        "n_source_rows": len(source_rows),
        "n_train_rows_pre_oversample": len(train_rows),
        "n_train_rows": len(train_rows_expanded),
        "n_eval_rows": len(eval_rows),
        "counts": {k: dict(v) for k, v in counts.items()},
        "intent_balance_mode": (
            args.intent_balance_mode
            if not args.variant.startswith("direct_posterior") else "none"
        ),
        "intent_repeat_factors": repeat_map,
        "direct_groundtruth_target": (
            {"true": 0.90, "false_each": 0.02}
            if args.variant in {"direct_posterior", "direct_posterior_groundtruth"} else None
        ),
    }
    summary_path = output_dir / f"{args.variant}_data_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    p.add_argument("--input-jsonl", type=Path, default=DEFAULT_SOURCE_JSONL)
    p.add_argument("--dialogues", type=Path, default=DEFAULT_DIALOGUES)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--eval-fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--intent-balance-mode",
        choices=("none", "oversample_to_anchor"),
        default="oversample_to_anchor",
    )
    p.add_argument("--max-intent-repeat", type=int, default=32)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = build_dataset(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
