"""DealOrNoDeal raw-data parsing utilities for the transfer experiments.

The archived Facebook dataset stores one *perspective-line* per example.
Each dialogue is usually represented twice, once from each agent's view:

    <input> c0 v0 c1 v1 c2 v2 </input>
    <dialogue> YOU: ... <eos> THEM: ... </dialogue>
    <output> item0=... item1=... item2=... item0=... item1=... item2=... </output>
    <partner_input> c0 v0 c1 v1 c2 v2 </partner_input>

Canonical item order in this module is DND-native: books, hats, balls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

DND_ITEMS: Tuple[str, str, str] = ("books", "hats", "balls")
DND_TOTAL_POINTS = 10
CASINO_ITEMS: Tuple[str, str, str] = ("Food", "Water", "Firewood")
NATIVE_TO_CASINO: Dict[str, str] = {
    "books": "Food",
    "hats": "Water",
    "balls": "Firewood",
}
CASINO_TO_NATIVE: Dict[str, str] = {v: k for k, v in NATIVE_TO_CASINO.items()}

RAW_BASE_URL = (
    "https://raw.githubusercontent.com/facebookresearch/"
    "end-to-end-negotiator/master/src/data/negotiate"
)

_TAG_RE = re.compile(r"<(?P<tag>\w+)>\s*(?P<body>.*?)\s*</(?P=tag)>", re.S)
_OUTPUT_RE = re.compile(r"item(?P<idx>[0-2])=(?P<count>-?\d+)")


@dataclass(frozen=True)
class DNDTurn:
    speaker: str  # YOU or THEM
    text: str
    is_selection: bool = False


@dataclass(frozen=True)
class DNDRecord:
    split: str
    line_index: int
    dialogue_id: str
    pair_key: str
    raw_line: str
    counts: Tuple[int, int, int]
    self_values: Tuple[int, int, int]
    partner_values: Tuple[int, int, int]
    self_ordering: Optional[Tuple[str, str, str]]
    partner_ordering: Optional[Tuple[str, str, str]]
    self_ordering_tiebreak: Tuple[str, str, str]
    partner_ordering_tiebreak: Tuple[str, str, str]
    self_tie: bool
    partner_tie: bool
    output_self: Tuple[int, int, int]
    output_partner: Tuple[int, int, int]
    output_valid: bool
    dialogue: Tuple[DNDTurn, ...]
    selection_speaker: Optional[str]

    @property
    def strict_both(self) -> bool:
        return self.self_ordering is not None and self.partner_ordering is not None

    def to_json(self) -> Dict[str, Any]:
        obj = asdict(self)
        obj["dialogue"] = [asdict(t) for t in self.dialogue]
        return obj


def item_label(item: str, *, name_mode: str) -> str:
    if name_mode == "native":
        return item
    if name_mode == "renamed":
        return NATIVE_TO_CASINO[item]
    raise ValueError(f"unknown name_mode {name_mode!r}")


def labels_for_mode(name_mode: str) -> Tuple[str, str, str]:
    return tuple(item_label(it, name_mode=name_mode) for it in DND_ITEMS)


def canonical_item(raw: Any) -> Optional[str]:
    """Map native, singular, and CaSiNo item labels to DND-native labels."""
    text = str(raw).strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    aliases = {
        "book": "books",
        "books": "books",
        "food": "books",
        "hat": "hats",
        "hats": "hats",
        "water": "hats",
        "ball": "balls",
        "balls": "balls",
        "firewood": "balls",
        "fire wood": "balls",
        "wood": "balls",
    }
    return aliases.get(text)


def map_text_names(text: str, *, name_mode: str) -> str:
    """Apply the Chawla-style DND→CaSiNo lexical mapping when requested."""
    if name_mode == "native":
        return text
    if name_mode != "renamed":
        raise ValueError(f"unknown name_mode {name_mode!r}")
    replacements = [
        (r"\bbooks\b", "food"),
        (r"\bbook\b", "food"),
        (r"\bhats\b", "water"),
        (r"\bhat\b", "water"),
        (r"\bballs\b", "firewood"),
        (r"\bball\b", "firewood"),
    ]
    out = text
    for pattern, repl in replacements:
        out = re.sub(pattern, repl, out, flags=re.I)
    return out


def values_to_ordering(
    values: Sequence[int],
    *,
    break_ties: bool = False,
) -> Optional[Tuple[str, str, str]]:
    if len(values) != 3:
        raise ValueError(f"expected three values, got {values!r}")
    if len(set(int(v) for v in values)) != 3 and not break_ties:
        return None
    idxs = sorted(range(3), key=lambda i: (-int(values[i]), i))
    return tuple(DND_ITEMS[i] for i in idxs)  # type: ignore[return-value]


def ordering_to_values_543(ordering: Sequence[str]) -> Tuple[float, float, float]:
    weights = {ordering[0]: 5.0, ordering[1]: 4.0, ordering[2]: 3.0}
    return tuple(weights[it] for it in DND_ITEMS)  # type: ignore[return-value]


def context_total_value(counts: Sequence[int], values: Sequence[float]) -> float:
    if len(counts) != 3 or len(values) != 3:
        raise ValueError(f"DND counts/values must have length 3, got {counts!r}, {values!r}")
    return float(sum(int(c) * float(v) for c, v in zip(counts, values)))


def validate_total_points(
    counts: Sequence[int],
    values: Sequence[float],
    *,
    role: str,
    expected: float = DND_TOTAL_POINTS,
) -> None:
    total = context_total_value(counts, values)
    if abs(total - float(expected)) > 1e-9:
        raise ValueError(
            f"DND {role} values must total {expected:g} points under counts {tuple(counts)}, "
            f"got {total:g} from values {tuple(values)}"
        )


def ordering_index(ordering: Sequence[str]) -> int:
    from itertools import permutations

    orderings = list(permutations(DND_ITEMS))
    return orderings.index(tuple(ordering))


def _extract_tag(line: str, tag: str) -> str:
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", line, re.S)
    if not m:
        raise ValueError(f"missing <{tag}> in DND line")
    return m.group(1).strip()


def _parse_context(body: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    toks = [int(x) for x in body.split()]
    if len(toks) != 6:
        raise ValueError(f"DND context must have 6 ints, got {body!r}")
    return (toks[0], toks[2], toks[4]), (toks[1], toks[3], toks[5])


def _parse_output(body: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], bool]:
    pairs = [(int(m.group("idx")), int(m.group("count"))) for m in _OUTPUT_RE.finditer(body)]
    if not pairs and "<" in body and ">" in body:
        return (0, 0, 0), (0, 0, 0), False
    if len(pairs) != 6:
        raise ValueError(f"DND output must contain 6 item assignments, got {body!r}")
    first = [0, 0, 0]
    second = [0, 0, 0]
    for idx, count in pairs[:3]:
        first[idx] = count
    for idx, count in pairs[3:]:
        second[idx] = count
    return tuple(first), tuple(second), True  # type: ignore[return-value]


def _parse_dialogue(body: str) -> Tuple[Tuple[DNDTurn, ...], Optional[str]]:
    turns: List[DNDTurn] = []
    selection_speaker: Optional[str] = None
    for raw in body.split("<eos>"):
        raw = raw.strip()
        if not raw:
            continue
        m = re.match(r"^(YOU|THEM):\s*(.*?)\s*$", raw, flags=re.S)
        if not m:
            raise ValueError(f"bad DND dialogue turn: {raw!r}")
        speaker = m.group(1)
        text = " ".join(m.group(2).split())
        is_selection = "<selection>" in text
        if is_selection:
            text = "<selection>"
            selection_speaker = speaker
        turns.append(DNDTurn(speaker=speaker, text=text, is_selection=is_selection))
    return tuple(turns), selection_speaker


def _pair_key(
    *,
    counts: Sequence[int],
    self_values: Sequence[int],
    partner_values: Sequence[int],
    dialogue: Sequence[DNDTurn],
) -> str:
    # Ignore perspective-specific YOU/THEM labels so the two views of the same
    # dialogue usually share a key.
    text = tuple(t.text for t in dialogue)
    ctxs = sorted([tuple(self_values), tuple(partner_values)])
    payload = json.dumps([tuple(counts), ctxs, text], sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def parse_dnd_line(line: str, *, split: str, line_index: int) -> DNDRecord:
    input_body = _extract_tag(line, "input")
    dialogue_body = _extract_tag(line, "dialogue")
    output_body = _extract_tag(line, "output")
    partner_body = _extract_tag(line, "partner_input")

    counts, self_values = _parse_context(input_body)
    partner_counts, partner_values = _parse_context(partner_body)
    if tuple(partner_counts) != tuple(counts):
        raise ValueError(
            f"count mismatch between input and partner_input: {counts} vs {partner_counts}"
        )
    validate_total_points(counts, self_values, role="self")
    validate_total_points(counts, partner_values, role="partner")
    output_self, output_partner, output_valid = _parse_output(output_body)
    dialogue, selection_speaker = _parse_dialogue(dialogue_body)
    self_ordering = values_to_ordering(self_values)
    partner_ordering = values_to_ordering(partner_values)
    pair = _pair_key(
        counts=counts,
        self_values=self_values,
        partner_values=partner_values,
        dialogue=dialogue,
    )
    return DNDRecord(
        split=split,
        line_index=int(line_index),
        dialogue_id=f"{split}_{line_index:05d}",
        pair_key=pair,
        raw_line=line.rstrip("\n"),
        counts=tuple(counts),
        self_values=tuple(self_values),
        partner_values=tuple(partner_values),
        self_ordering=self_ordering,
        partner_ordering=partner_ordering,
        self_ordering_tiebreak=values_to_ordering(self_values, break_ties=True),  # type: ignore[arg-type]
        partner_ordering_tiebreak=values_to_ordering(partner_values, break_ties=True),  # type: ignore[arg-type]
        self_tie=self_ordering is None,
        partner_tie=partner_ordering is None,
        output_self=output_self,
        output_partner=output_partner,
        output_valid=output_valid,
        dialogue=dialogue,
        selection_speaker=selection_speaker,
    )


def parse_split_file(path: Path, *, split: str) -> List[DNDRecord]:
    records: List[DNDRecord] = []
    with Path(path).open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                records.append(parse_dnd_line(line, split=split, line_index=i))
    return records


def download_raw_split(split: str, output_dir: Path, *, overwrite: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{split}.txt"
    if out.exists() and not overwrite:
        return out
    url = f"{RAW_BASE_URL}/{split}.txt"
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    out.write_bytes(data)
    return out


def ensure_raw_splits(output_dir: Path, *, overwrite: bool = False) -> Dict[str, Path]:
    return {
        split: download_raw_split(split, output_dir, overwrite=overwrite)
        for split in ("train", "val", "test")
    }


def records_to_jsonl(records: Iterable[DNDRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_json(), ensure_ascii=False) + "\n")


def compute_stats(records: Sequence[DNDRecord]) -> Dict[str, Any]:
    n = len(records)
    strict_partner = [r for r in records if r.partner_ordering is not None]
    strict_self = [r for r in records if r.self_ordering is not None]
    strict_both = [r for r in records if r.strict_both]
    order_dist = Counter(
        ">".join(r.partner_ordering) for r in strict_partner if r.partner_ordering
    )
    valid_outputs = [r for r in records if r.output_valid]
    opp_turns = []
    selection_depths = []
    selection_speakers = Counter()
    for r in records:
        k = 0
        for t in r.dialogue:
            if t.is_selection:
                break
            if t.speaker == "THEM":
                k += 1
        opp_turns.append(k)
        selection_depths.append(k)
        if r.selection_speaker:
            selection_speakers[r.selection_speaker] += 1
    k_support = {
        str(k): sum(1 for x in opp_turns if x >= k)
        for k in range(1, 6)
    }
    return {
        "n_records": n,
        "strict_self": len(strict_self),
        "strict_partner": len(strict_partner),
        "strict_both": len(strict_both),
        "valid_output": len(valid_outputs),
        "invalid_output": n - len(valid_outputs),
        "agreement_rate": len(valid_outputs) / n if n else None,
        "self_tie": n - len(strict_self),
        "partner_tie": n - len(strict_partner),
        "self_tie_rate": (n - len(strict_self)) / n if n else None,
        "partner_tie_rate": (n - len(strict_partner)) / n if n else None,
        "partner_ordering_distribution": dict(sorted(order_dist.items())),
        "k_support_all": k_support,
        "selection_speaker_counts": dict(selection_speakers),
        "mean_opp_utterances_before_selection": (
            sum(selection_depths) / len(selection_depths) if selection_depths else None
        ),
    }


def group_records_by_pair(records: Sequence[DNDRecord]) -> Dict[str, List[DNDRecord]]:
    out: Dict[str, List[DNDRecord]] = {}
    for rec in records:
        out.setdefault(rec.pair_key, []).append(rec)
    return out


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/results/dnd_transfer/main/data"))
    p.add_argument("--overwrite-raw", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    raw_dir = args.output_dir / "raw"
    raw_paths = ensure_raw_splits(raw_dir, overwrite=args.overwrite_raw)
    stats: Dict[str, Any] = {"raw_paths": {k: str(v) for k, v in raw_paths.items()}}
    for split, path in raw_paths.items():
        records = parse_split_file(path, split=split)
        records_to_jsonl(records, args.output_dir / f"dnd_parsed_{split}.jsonl")
        split_stats = compute_stats(records)
        stats[split] = split_stats
        strict_partner = [r for r in records if r.partner_ordering is not None]
        strict_both = [r for r in records if r.strict_both]
        records_to_jsonl(
            strict_partner,
            args.output_dir / f"dnd_{split}_strict_partner_orderings.jsonl",
        )
        records_to_jsonl(
            strict_both,
            args.output_dir / f"dnd_{split}_strict_both_orderings.jsonl",
        )
    stats_path = args.output_dir / "dnd_data_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "stats": str(stats_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
