"""Per-turn evaluation harness for CaSiNo agents.

Where ``opponent_model.metrics.evaluate_opponent_model`` snapshots an
agent's *priority ordering* every k=1..5 opponent utterances, this module
walks every dialogue turn-by-turn and asks the agent four questions
whenever the perspective agent is the one speaking:

    1. accept / reject the most-recent un-resolved Submit-Deal (if any)
    2. counter-bid: a 6-int allocation [Food/Water/Firewood for self,
       same for opponent] that the agent would propose
    3. strategy label(s) for the utterance the agent would emit (against
       the CaSiNo 10-tag strategy taxonomy)
    4. posterior over the 6 priority-ordering hypotheses

Ground truth comes from the held-out human's actual turn t:

    * accept gold = ``Accept-Deal`` (True) / ``Reject-Deal`` or
      ``Walk-Away`` (False); other utterances skipped.
    * bid gold    = the ``issue2youget`` / ``issue2theyget`` block on a
      ``Submit-Deal`` turn.
    * strategy gold = the comma-separated tags from
      ``casino_ann.json`` (matched to chat_logs by text).
    * posterior gold = one-hot over the 6 hypotheses, picked from the
      opponent's ``value2issue``.

Metrics:

    * ``accept_f1``         binary F1 on accept (positive class = accept)
    * ``bid_cosine``        mean cosine similarity over Submit-Deal turns
    * ``strategy_macro_f1`` macro F1 over the 10-tag label set
    * ``brier``             mean *normalized* multiclass Brier
                            (1/K * sum_k (p_k - 1{k=true})^2), in [0, 1]

The ``TurnLevelAgent`` Protocol below defines the agent contract; see
``opponent_model.turn_agents`` for adapters around ``HybridAgent`` and
``sft_8b.predict.SftModelFn``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import (
    Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol,
    Sequence, Tuple,
)

import numpy as np

from opponent_model.hypotheses import HYPOTHESES, ITEMS

logger = logging.getLogger("opponent_model.turn_level_metrics")


# ── Constants ──────────────────────────────────────────────────────────────


CASINO_STRATEGIES: Tuple[str, ...] = (
    "small-talk",
    "self-need",
    "other-need",
    "no-need",
    "elicit-pref",
    "promote-coordination",
    "vouch-fair",
    "showing-empathy",
    "uv-part",
    "non-strategic",
)

DEAL_ACTIONS: frozenset[str] = frozenset(
    {"Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away"}
)
RESOLVING_ACTIONS: frozenset[str] = frozenset(
    {"Accept-Deal", "Reject-Deal", "Walk-Away"}
)


# ── Agent contract ─────────────────────────────────────────────────────────


class TurnLevelAgent(Protocol):
    """Contract expected by :func:`turn_level_eval`.

    Implementations should be stateless across calls (the harness does not
    promise any particular call order beyond walking dialogues
    sequentially). Each call gets the *full* history up to (but not
    including) the turn being predicted, so an agent that needs to
    accumulate state should rebuild it from ``history`` each time.
    """

    def predict_turn(
        self,
        *,
        history: List[Mapping[str, Any]],
        my_role: str,
        opp_role: str,
        my_priorities: Mapping[str, str],
        my_reasons: Mapping[str, str],
        pending_offer: Optional[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Return a dict with any subset of the keys:

            ``accept``    -> Optional[bool]                  (decision on pending)
            ``bid``       -> Optional[Mapping[str, int]]     (counter offer; my-receive)
                              -- accepted shapes: ``{"Food": .., "Water": .., "Firewood": ..}``
                              -- or ``{"self": {...}, "opp": {...}}`` for the full split
            ``strategy``  -> Optional[Sequence[str]]         (multi-label CaSiNo tags)
            ``posterior`` -> Optional[Sequence[float]]       (length-6 over HYPOTHESES)

        Missing/None keys are interpreted as "agent abstains" and the
        corresponding metric is *not* updated for that turn.
        """
        ...


# ── Helpers: ground-truth extraction ───────────────────────────────────────


def accept_label_from_turn(turn: Mapping[str, Any]) -> Optional[bool]:
    """Map a chat_logs turn to {True, False, None}.

    True iff the turn is ``Accept-Deal``; False iff it is ``Reject-Deal``
    or ``Walk-Away``. None for everything else (including utterances and
    Submit-Deal — those are not accept/reject decisions).
    """
    text = turn.get("text", "")
    if text == "Accept-Deal":
        return True
    if text in ("Reject-Deal", "Walk-Away"):
        return False
    return None


def bid_from_turn(
    turn: Mapping[str, Any],
    *,
    target_role: str,
) -> Optional[np.ndarray]:
    """Extract the 6-dim bid vector from a Submit-Deal turn.

    Vector layout (deterministic order):

        ``[target_food, target_water, target_firewood,
           other_food,  other_water,  other_firewood]``

    where ``target`` is what the perspective agent would *receive*.
    Returns ``None`` for non-Submit-Deal turns or malformed task_data.
    """
    if turn.get("text") != "Submit-Deal":
        return None
    td = turn.get("task_data") or {}
    if "issue2youget" not in td or "issue2theyget" not in td:
        return None

    if turn.get("id") == target_role:
        target = td["issue2youget"]
        other = td["issue2theyget"]
    else:
        target = td["issue2theyget"]
        other = td["issue2youget"]

    try:
        arr = np.array(
            [int(target.get(it, 0)) for it in ITEMS]
            + [int(other.get(it, 0)) for it in ITEMS],
            dtype=float,
        )
    except (TypeError, ValueError):
        return None
    return arr


def coerce_bid_vector(
    bid_pred: Any,
    *,
    target_self: bool = True,
) -> Optional[np.ndarray]:
    """Normalize an agent's bid output into the same 6-dim vector layout.

    Accepted shapes:
        * ``{"Food": int, "Water": int, "Firewood": int}`` — interpreted
          as the *receiving* side of the perspective agent. The other
          side is filled with ``3 - x`` per item (CaSiNo invariant: each
          item totals to 3).
        * ``{"self": {...}, "opp": {...}}`` — each block is a 3-item dict.
        * ``{"issue2youget": {...}, "issue2theyget": {...}}`` — same as
          a chat_logs Submit-Deal task_data.
        * Length-6 sequence in the canonical order — passed through.
    """
    if bid_pred is None:
        return None

    if isinstance(bid_pred, (list, tuple, np.ndarray)):
        arr = np.asarray(bid_pred, dtype=float).flatten()
        if arr.shape == (6,):
            return arr
        if arr.shape == (3,):
            return _complete_bid(arr, target_self=target_self)
        return None

    if not isinstance(bid_pred, Mapping):
        return None

    if "self" in bid_pred and "opp" in bid_pred:
        try:
            self_arr = np.array(
                [int(bid_pred["self"].get(it, 0)) for it in ITEMS], dtype=float,
            )
            opp_arr = np.array(
                [int(bid_pred["opp"].get(it, 0)) for it in ITEMS], dtype=float,
            )
        except (TypeError, ValueError):
            return None
        return np.concatenate([self_arr, opp_arr])

    if "issue2youget" in bid_pred and "issue2theyget" in bid_pred:
        try:
            self_arr = np.array(
                [int(bid_pred["issue2youget"].get(it, 0)) for it in ITEMS], dtype=float,
            )
            opp_arr = np.array(
                [int(bid_pred["issue2theyget"].get(it, 0)) for it in ITEMS], dtype=float,
            )
        except (TypeError, ValueError):
            return None
        return np.concatenate([self_arr, opp_arr])

    if all(it in bid_pred for it in ITEMS):
        try:
            arr = np.array([int(bid_pred[it]) for it in ITEMS], dtype=float)
        except (TypeError, ValueError):
            return None
        return _complete_bid(arr, target_self=target_self)

    return None


def _complete_bid(arr3: np.ndarray, *, target_self: bool) -> np.ndarray:
    """Given a 3-dim allocation for one side, infer the other side as 3-x."""
    other = np.clip(3.0 - arr3, 0.0, 3.0)
    if target_self:
        return np.concatenate([arr3, other])
    return np.concatenate([other, arr3])


def strategy_labels_from_annotation(
    ann_value: Any,
) -> Optional[List[str]]:
    """Parse a single annotation cell into a clean list of CaSiNo tags."""
    if ann_value is None:
        return None
    if isinstance(ann_value, (list, tuple)):
        return [str(t).strip() for t in ann_value if str(t).strip()]
    if not isinstance(ann_value, str):
        return None
    return [t.strip() for t in ann_value.split(",") if t.strip()]


def build_annotation_lookup(
    annotations: Sequence[Any],
    chat_logs: Sequence[Mapping[str, Any]],
) -> Dict[int, List[str]]:
    """Match a per-utterance annotation list to chat_logs indices.

    The CaSiNo annotation file lists one ``[utterance_text, tags]`` entry
    per *natural* utterance (skipping Submit-Deal/Accept-Deal/etc.) in
    order. Most of the time we can walk both lists in lockstep, but a
    handful of dialogues have minor whitespace mismatches; we tolerate
    those by matching on stripped/lowered text and falling back to
    positional matching when needed.
    """
    lookup: Dict[int, List[str]] = {}
    if not annotations:
        return lookup

    ai = 0
    n = len(annotations)
    for ci, log in enumerate(chat_logs):
        text = (log.get("text") or "").strip()
        if not text or text in DEAL_ACTIONS:
            continue
        if ai >= n:
            break

        ann = annotations[ai]
        if not isinstance(ann, (list, tuple)) or len(ann) < 2:
            ai += 1
            continue

        ann_text = (ann[0] or "").strip() if isinstance(ann[0], str) else ""
        if ann_text == text or ann_text.lower() == text.lower():
            tags = strategy_labels_from_annotation(ann[1])
            if tags is not None:
                lookup[ci] = tags
            ai += 1
        else:
            tags = strategy_labels_from_annotation(ann[1])
            if tags is not None:
                lookup[ci] = tags
            ai += 1
    return lookup


def last_pending_offer(
    history: Sequence[Mapping[str, Any]],
    *,
    perspective: str,
) -> Optional[Dict[str, Any]]:
    """Most recent un-resolved Submit-Deal in ``history``.

    Walks backward from the end of history. Returns ``None`` if a
    resolving action (Accept/Reject/Walk-Away) is encountered first.

    ``perspective`` is the agent making the decision; we attach
    ``"to_perspective"`` indicating whether the offer was *theirs* or
    the opponent's so callers can decide whether to even predict accept
    (you don't accept your own offer).
    """
    for turn in reversed(history):
        text = turn.get("text", "")
        if text in RESOLVING_ACTIONS:
            return None
        if text == "Submit-Deal":
            td = turn.get("task_data") or {}
            if "issue2youget" not in td:
                continue
            proposer = turn.get("id")
            return {
                "proposer": proposer,
                "to_perspective": proposer != perspective,
                "task_data": td,
            }
    return None


def true_hypothesis_index(true_ordering: Sequence[str]) -> Optional[int]:
    target = tuple(true_ordering)
    for i, h in enumerate(HYPOTHESES):
        if tuple(h) == target:
            return i
    return None


# ── Metric primitives ──────────────────────────────────────────────────────


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _binary_f1(tp: int, fp: int, fn: int) -> float:
    if (tp + fp + fn) == 0:
        return float("nan")
    if tp == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def binary_accept_f1(
    pairs: Sequence[Tuple[bool, bool]],
) -> Dict[str, float]:
    """F1 for the positive class ('accept'); also report per-class & accuracy."""
    if not pairs:
        return {"f1": float("nan"), "accuracy": float("nan"),
                "precision": float("nan"), "recall": float("nan"),
                "support": 0}
    tp = fp = fn = tn = 0
    for pred, gold in pairs:
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif not pred and gold:
            fn += 1
        else:
            tn += 1
    f1 = _binary_f1(tp, fp, fn)
    acc = (tp + tn) / len(pairs)
    p = tp / (tp + fp) if (tp + fp) else float("nan")
    r = tp / (tp + fn) if (tp + fn) else float("nan")
    return {
        "f1": float(f1) if not isinstance(f1, float) or not np.isnan(f1) else float("nan"),
        "accuracy": float(acc),
        "precision": float(p) if not (isinstance(p, float) and np.isnan(p)) else float("nan"),
        "recall": float(r) if not (isinstance(r, float) and np.isnan(r)) else float("nan"),
        "support": len(pairs),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def macro_f1_multilabel(
    pred_labels: Sequence[Sequence[str]],
    true_labels: Sequence[Sequence[str]],
    *,
    label_set: Sequence[str],
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """Macro F1 over a fixed label set (each label = its own binary problem)."""
    per_label: Dict[str, Dict[str, float]] = {}
    f1_scores: List[float] = []
    for label in label_set:
        tp = fp = fn = 0
        for pred, true in zip(pred_labels, true_labels):
            in_pred = label in (pred or ())
            in_true = label in (true or ())
            if in_pred and in_true:
                tp += 1
            elif in_pred and not in_true:
                fp += 1
            elif not in_pred and in_true:
                fn += 1
        f1 = _binary_f1(tp, fp, fn)
        per_label[label] = {
            "f1": f1,
            "support": tp + fn,
            "tp": tp, "fp": fp, "fn": fn,
        }
        if not np.isnan(f1):
            f1_scores.append(f1)
    macro = float(np.mean(f1_scores)) if f1_scores else float("nan")
    return macro, per_label


def normalized_brier(
    posterior: np.ndarray,
    true_index: int,
) -> float:
    """Mean over classes of (p_k - 1{k=true})^2, in [0, 1]."""
    n = posterior.shape[0]
    one_hot = np.zeros(n, dtype=float)
    one_hot[true_index] = 1.0
    return float(np.mean((posterior - one_hot) ** 2))


# ── Per-turn record ────────────────────────────────────────────────────────


@dataclass
class TurnRecord:
    """One per-turn snapshot fed to ``on_record`` and stored in the result."""

    dialogue_id: Any
    perspective: str
    opp_role: str
    turn_index: int
    speaker: str
    turn_text: str
    is_action: bool
    pending_offer: Optional[Dict[str, Any]] = None
    pred: Dict[str, Any] = field(default_factory=dict)
    true: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Main loop ──────────────────────────────────────────────────────────────


def turn_level_eval(
    dialogues: Iterable[Mapping[str, Any]],
    agent: TurnLevelAgent,
    *,
    perspectives: Sequence[str] = ("mturk_agent_1", "mturk_agent_2"),
    label_set: Sequence[str] = CASINO_STRATEGIES,
    annotations_by_dialogue: Optional[Mapping[Any, Sequence[Any]]] = None,
    on_record: Optional[Callable[[TurnRecord], None]] = None,
    max_dialogues: Optional[int] = None,
) -> Dict[str, Any]:
    """Walk each held-out dialogue turn-by-turn for a given agent.

    For each turn ``t`` where ``perspective`` is the speaker, the agent
    is asked to predict accept / bid / strategy / posterior given
    ``chat_logs[:t]`` and the most-recent un-resolved Submit-Deal (the
    "pending offer"). Predictions are scored against the human's actual
    turn ``t``.

    Args:
        dialogues: held-out CaSiNo dialogues (with ``chat_logs`` and
            ``participant_info``). Optionally with ``annotations`` for
            strategy labels, or pass them via ``annotations_by_dialogue``.
        agent: any object satisfying :class:`TurnLevelAgent`.
        perspectives: which roles to evaluate (default = both).
        label_set: strategy taxonomy for macro F1 (default = CaSiNo 10).
        annotations_by_dialogue: ``{dialogue_id -> annotations list}``
            from ``casino_ann.json``. Falls back to dialogue-embedded
            annotations.
        on_record: optional streaming callback (e.g. JSONL writer).
        max_dialogues: cap on dialogues processed (debug / smoke).

    Returns:
        Dict with metric summary, support counts, and ``records`` list.
    """
    accept_pairs: List[Tuple[bool, bool]] = []
    bid_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    strat_pred: List[List[str]] = []
    strat_true: List[List[str]] = []
    brier_vals: List[float] = []
    records: List[TurnRecord] = []

    perspectives = tuple(perspectives)
    if not 1 <= len(perspectives) <= 2:
        raise ValueError(
            f"perspectives must be a 1- or 2-tuple of role ids; got {perspectives!r}"
        )

    # Canonical CaSiNo role pair — used to infer the opponent when the
    # caller restricts `perspectives` to a single role (e.g. for an
    # apples-to-apples replay comparison where the baseline only has
    # data for one side).
    _ROLE_PAIR = ("mturk_agent_1", "mturk_agent_2")

    def _infer_opp(role: str) -> str:
        if len(perspectives) == 2:
            return perspectives[1] if role == perspectives[0] else perspectives[0]
        return _ROLE_PAIR[1] if role == _ROLE_PAIR[0] else _ROLE_PAIR[0]

    for di, dialogue in enumerate(dialogues):
        if max_dialogues is not None and di >= max_dialogues:
            break

        did = dialogue.get("dialogue_id", di)
        chat_logs = dialogue.get("chat_logs", [])
        pinfo = dialogue.get("participant_info", {})
        anns = (
            (annotations_by_dialogue or {}).get(did)
            or dialogue.get("annotations")
            or []
        )

        for perspective in perspectives:
            opp_role = _infer_opp(perspective)
            try:
                my_priorities = pinfo[perspective]["value2issue"]
                opp_priorities = pinfo[opp_role]["value2issue"]
            except KeyError as e:
                logger.warning(
                    "dialogue %s: missing participant_info key %s; skipping perspective %s",
                    did, e, perspective,
                )
                continue
            my_reasons = pinfo[perspective].get("value2reason", {})

            true_ordering = [
                opp_priorities["High"],
                opp_priorities["Medium"],
                opp_priorities["Low"],
            ]
            true_idx = true_hypothesis_index(true_ordering)

            ann_lookup = build_annotation_lookup(anns, chat_logs)

            history: List[Dict[str, Any]] = []
            for t, turn in enumerate(chat_logs):
                if turn.get("id") != perspective:
                    history.append(dict(turn))
                    continue

                pending = last_pending_offer(history, perspective=perspective)

                try:
                    pred_full = agent.predict_turn(
                        history=list(history),
                        my_role=perspective,
                        opp_role=opp_role,
                        my_priorities=dict(my_priorities),
                        my_reasons=dict(my_reasons),
                        pending_offer=pending,
                    ) or {}
                except Exception:
                    logger.exception(
                        "agent.predict_turn raised at dialogue %s turn %d "
                        "(perspective %s); skipping turn.",
                        did, t, perspective,
                    )
                    history.append(dict(turn))
                    continue

                pred = dict(pred_full)
                gold_accept = accept_label_from_turn(turn)
                gold_bid = bid_from_turn(turn, target_role=perspective)
                gold_strat = ann_lookup.get(t)

                pred_accept = pred.get("accept")
                pred_bid = coerce_bid_vector(pred.get("bid"), target_self=True)
                pred_strat_raw = pred.get("strategy")
                pred_strat = (
                    [str(x).strip() for x in pred_strat_raw if str(x).strip()]
                    if pred_strat_raw is not None else None
                )
                pred_posterior_raw = pred.get("posterior")
                pred_posterior: Optional[np.ndarray] = None
                if pred_posterior_raw is not None:
                    arr = np.asarray(pred_posterior_raw, dtype=float).flatten()
                    if arr.shape == (len(HYPOTHESES),) and np.all(arr >= 0):
                        s = float(arr.sum())
                        if s > 0:
                            pred_posterior = arr / s

                # ── Accept F1 ── (only on Accept/Reject/Walk-Away turns
                # where there was actually an offer to respond to)
                if (
                    gold_accept is not None
                    and pred_accept is not None
                    and pending is not None
                    and pending.get("to_perspective", False)
                ):
                    accept_pairs.append((bool(pred_accept), bool(gold_accept)))

                # ── Bid cosine ── (only on Submit-Deal turns)
                if gold_bid is not None and pred_bid is not None:
                    bid_pairs.append((pred_bid, gold_bid))

                # ── Strategy macro-F1 ── (only on natural utterance turns)
                if (
                    gold_strat is not None
                    and pred_strat is not None
                    and turn.get("text") not in DEAL_ACTIONS
                ):
                    strat_true.append(list(gold_strat))
                    strat_pred.append(list(pred_strat))

                # ── Brier ── (always when posterior is exposed)
                if pred_posterior is not None and true_idx is not None:
                    brier_vals.append(normalized_brier(pred_posterior, true_idx))

                rec = TurnRecord(
                    dialogue_id=did,
                    perspective=perspective,
                    opp_role=opp_role,
                    turn_index=t,
                    speaker=turn.get("id", ""),
                    turn_text=turn.get("text", ""),
                    is_action=turn.get("text") in DEAL_ACTIONS,
                    pending_offer=pending,
                    pred={
                        "accept":    pred_accept,
                        "bid":       (pred_bid.tolist() if pred_bid is not None else None),
                        "strategy":  pred_strat,
                        "posterior": (
                            pred_posterior.tolist()
                            if pred_posterior is not None else None
                        ),
                    },
                    true={
                        "accept":   gold_accept,
                        "bid":      (gold_bid.tolist() if gold_bid is not None else None),
                        "strategy": gold_strat,
                        "true_ordering": true_ordering,
                        "true_hypothesis_index": true_idx,
                    },
                )
                records.append(rec)
                if on_record is not None:
                    on_record(rec)

                history.append(dict(turn))

    summary = aggregate_turn_metrics(
        accept_pairs=accept_pairs,
        bid_pairs=bid_pairs,
        strategy_pairs=(strat_pred, strat_true),
        brier_vals=brier_vals,
        label_set=label_set,
    )
    summary["n_records"] = len(records)
    summary["records"] = records
    return summary


def aggregate_turn_metrics(
    *,
    accept_pairs: Sequence[Tuple[bool, bool]],
    bid_pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    strategy_pairs: Tuple[Sequence[Sequence[str]], Sequence[Sequence[str]]],
    brier_vals: Sequence[float],
    label_set: Sequence[str] = CASINO_STRATEGIES,
) -> Dict[str, Any]:
    """Compute the four headline metrics from already-bucketed pairs."""
    accept_metrics = binary_accept_f1(list(accept_pairs))

    bid_cos_vals = [cosine_similarity(p, g) for p, g in bid_pairs]
    bid_cos_mean = (
        float(np.mean(bid_cos_vals)) if bid_cos_vals else float("nan")
    )

    strat_pred, strat_true = strategy_pairs
    strat_macro_f1, per_label = macro_f1_multilabel(
        strat_pred, strat_true, label_set=label_set,
    )

    brier_mean = (
        float(np.mean(list(brier_vals))) if brier_vals else float("nan")
    )

    return {
        "accept": accept_metrics,
        "bid_cosine": {
            "mean": bid_cos_mean,
            "support": len(bid_pairs),
        },
        "strategy_macro_f1": {
            "macro_f1": strat_macro_f1,
            "support": len(strat_true),
            "per_label": per_label,
        },
        "brier": {
            "mean": brier_mean,
            "support": len(list(brier_vals)),
        },
    }


def format_turn_level_summary(result: Mapping[str, Any]) -> str:
    """One-paragraph human-readable rendering of the headline metrics."""
    acc = result["accept"]
    bid = result["bid_cosine"]
    strat = result["strategy_macro_f1"]
    bri = result["brier"]
    n = result.get("n_records", 0)

    lines = [
        f"turn_level_eval: {n} per-turn predictions",
        "",
        f"  accept F1            {acc['f1']:>7.3f}   (n={acc['support']}, "
        f"P={acc['precision']:.3f} R={acc['recall']:.3f}, acc={acc['accuracy']:.3f})",
        f"  bid cosine           {bid['mean']:>7.3f}   (n={bid['support']})",
        f"  strategy macro-F1    {strat['macro_f1']:>7.3f}   (n={strat['support']}, "
        f"|labels|={len(strat['per_label'])})",
        f"  Brier (posterior)    {bri['mean']:>7.3f}   (n={bri['support']})",
    ]
    return "\n".join(lines)


__all__ = [
    "CASINO_STRATEGIES",
    "DEAL_ACTIONS",
    "TurnLevelAgent",
    "TurnRecord",
    "accept_label_from_turn",
    "bid_from_turn",
    "coerce_bid_vector",
    "build_annotation_lookup",
    "last_pending_offer",
    "true_hypothesis_index",
    "cosine_similarity",
    "binary_accept_f1",
    "macro_f1_multilabel",
    "normalized_brier",
    "turn_level_eval",
    "aggregate_turn_metrics",
    "format_turn_level_summary",
]
