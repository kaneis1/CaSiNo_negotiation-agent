"""Protocol-3 re-evaluation of the Structured-CoT 70B baseline.

Where ``run_protocol1.py`` lets the baseline self-play until it accepts
(which terminates its own trajectory and systematically under-samples
gold decision turns — see ``structured_cot/baseline_weaknesses.md``),
this runner walks the *gold* CaSiNo chat_logs turn-by-turn and asks the
baseline "what would you decide here, given the human's history up to
this turn?" at every decision-relevant turn.

Scope (cost control).
    We query the 70B only on mt1 chat_logs turns whose ``text`` is one of
    ``{Submit-Deal, Accept-Deal, Reject-Deal, Walk-Away}``. Those are the
    only turns ``opponent_model.turn_level_eval`` will count for accept F1
    (Accept/Reject/Walk-Away) and bid cosine (Submit-Deal). Skipping the
    utterance-only turns cuts the call count from ~900 to ~173, which
    gets us matched-support numbers in a single 1-2 h H100 run instead
    of ~8 h. Strategy macro-F1 is already a template-utterance artifact
    in the Bayesian teacher (and the primary metrics the field reports
    are accept F1 + bid), so we don't regenerate strategy labels here.

Output artifacts.
    structured_cot/results/protocol3_70b_decision_only/
        turns.jsonl       one line per queried gold turn; schema matches
                          the Protocol-1 ``turns.jsonl`` so the existing
                          ``StructuredCoTReplayAgent`` adapter can read
                          it without modification.
        summary.json      aggregate stats (n_queried, parse-error rate,
                          total wallclock).
        args.json         this run's CLI arguments.
        parse_failures.jsonl   only populated on parse errors.

The resulting ``turns.jsonl`` is a drop-in replacement for the P1 log
when scoring through ``opponent_model.turn_eval_run.py --agent
structured_cot_replay``: same dialogues, same harness, matched support.

Usage:
    python -m structured_cot.run_protocol3_baseline \\
        --data          data/casino_test.json \\
        --num-dialogues 150 \\
        --backend       llama_70b \\
        --run-name      protocol3_70b_decision_only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from structured_cot.agent import ActResult, StructuredCoTAgent
from structured_cot.llm_client import DummyStructuredLLM, StructuredLLMClient
from structured_cot.run_protocol1 import (
    ITEMS,
    _coerce_counts,
    _get_arguments,
    _get_priorities,
    _ground_truth_decision,
    _opponent_role,
    is_deal_action,
    render_agent_counter_offer,
    render_opponent_deal_action,
)

logger = logging.getLogger("structured_cot.run_protocol3_baseline")

DECISION_ACTIONS = ("Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away")


# ── History rendering: turn gold chat_logs[0..t-1] into (speaker, text) ─────


def render_agent_turn_from_gold(turn: Mapping[str, Any]) -> str:
    """Render a gold *agent* turn (mt1 when mt1 is the perspective) in the
    same shape the P1 trajectory would have produced. If the gold turn is
    a Submit-Deal by the agent, we render it as an agent counter-offer so
    the LLM sees a symmetric representation. Otherwise we pass the text
    through (natural utterances, Accept/Reject/Walk-Away)."""
    text = (turn.get("text") or "").strip()
    if text.startswith("Submit-Deal"):
        td = turn.get("task_data") or {}
        # ``issue2youget`` on a Submit-Deal is what the SUBMITTER keeps.
        # The agent in P1 renders "I propose: I get X; you get Y", so
        # we match that semantics exactly.
        you_get = _coerce_counts(td.get("issue2youget", {}))
        return render_agent_counter_offer(you_get)
    if text.startswith("Accept-Deal"):
        return "Accept-Deal — I accept the proposal on the table."
    if text.startswith("Reject-Deal"):
        return "Reject-Deal — I reject the proposal."
    if text.startswith("Walk-Away"):
        return "Walk-Away — I am walking away from the negotiation."
    return text or "(said nothing)"


def build_gold_history(
    chat_logs: List[Mapping[str, Any]],
    *,
    up_to_index: int,
    agent_role: str,
) -> List[Tuple[str, str]]:
    """Build the (speaker, text) history the LLM should see at chat_logs[up_to_index].

    Speakers: "me" when the turn was the agent's in gold, "opp" otherwise.
    Rendering matches the Protocol-1 conventions (see
    ``render_opponent_deal_action`` and ``render_agent_counter_offer``)
    so the 70B encounters the same surface form it was prompted with
    under Protocol 1.
    """
    hist: List[Tuple[str, str]] = []
    for j in range(up_to_index):
        t = chat_logs[j]
        speaker = t.get("id")
        if speaker == agent_role:
            hist.append(("me", render_agent_turn_from_gold(t)))
        else:
            hist.append(("opp", render_opponent_deal_action(t)))
    return hist


def last_pending_offer_to_agent(
    chat_logs: List[Mapping[str, Any]],
    *,
    up_to_index: int,
    agent_role: str,
) -> Optional[Dict[str, int]]:
    """Most recent un-resolved Submit-Deal by the opponent in the gold prefix.

    Returns ``{Food, Water, Firewood}`` from the AGENT's receiving side
    (``issue2theyget`` on the opponent's Submit-Deal), matching how
    ``run_protocol1.py`` tracks ``opp_last_offer``. Returns ``None`` if
    the most recent deal action was a resolution (Accept/Reject/Walk-Away)
    or if no Submit-Deal appears.
    """
    for j in range(up_to_index - 1, -1, -1):
        t = chat_logs[j]
        text = (t.get("text") or "").strip()
        if text in ("Accept-Deal", "Reject-Deal", "Walk-Away"):
            return None
        if text.startswith("Submit-Deal"):
            if t.get("id") == agent_role:
                # Our own Submit-Deal is not "to us"; keep walking.
                continue
            td = t.get("task_data") or {}
            return _coerce_counts(td.get("issue2theyget", {}))
    return None


# ── Main runner ────────────────────────────────────────────────────────────


def _build_agent(dialogue: Mapping[str, Any], agent_role: str, llm: Any,
                 *, max_tokens: int, temperature: float,
                 parse_log: Optional[Path]) -> StructuredCoTAgent:
    pinfo = dialogue.get("participant_info") or {}
    return StructuredCoTAgent(
        priorities=_get_priorities(pinfo.get(agent_role) or {}),
        arguments=_get_arguments(pinfo.get(agent_role) or {}),
        llm_client=llm,
        max_tokens=max_tokens,
        temperature=temperature,
        parse_log_path=parse_log,
    )


def query_one_turn(
    dialogue: Mapping[str, Any],
    turn_index: int,
    agent: StructuredCoTAgent,
    *,
    agent_role: str,
    opp_role: str,
    agent_priorities: Dict[str, str],
    opp_priorities: Dict[str, str],
) -> Dict[str, Any]:
    """Call the 70B on gold history[:turn_index] and return a trace row."""
    chat_logs = list(dialogue.get("chat_logs") or [])
    history = build_gold_history(
        chat_logs, up_to_index=turn_index, agent_role=agent_role,
    )
    pending = last_pending_offer_to_agent(
        chat_logs, up_to_index=turn_index, agent_role=agent_role,
    )

    gold_turn = chat_logs[turn_index]
    ground_truth = _ground_truth_decision(gold_turn)
    ground_truth_utterance = (gold_turn.get("text") or "").strip()

    result: ActResult = agent.act(history, pending_offer=pending)
    parsed = result.parsed
    decision = parsed.get("decision") or {}

    return {
        "dialogue_id":               dialogue.get("dialogue_id"),
        "turn_index":                turn_index,
        "agent_turn_index":          None,  # P3 doesn't have a self-trajectory index
        "agent_role":                agent_role,
        "agent_priorities":          agent_priorities,
        "opponent_priorities":       opp_priorities,
        "history_before":            [list(p) for p in history],
        "pending_offer":             pending,
        "raw_output_first":          result.raw_first,
        "raw_output_retry":          result.raw_retry,
        "retried":                   result.retried,
        "fell_back":                 result.fell_back,
        "parse_errors":              result.parse_errors,
        "elapsed_seconds":           result.elapsed_seconds,
        "parsed_observation":        parsed.get("observation"),
        "parsed_opponent_inference": parsed.get("opponent_inference"),
        "parsed_plan":               parsed.get("plan"),
        "parsed_utterance":          parsed.get("utterance"),
        "parsed_decision":           decision,
        "ground_truth_decision":     ground_truth,
        "ground_truth_utterance":    ground_truth_utterance,
        "gold_turn_text":            ground_truth_utterance,
    }


def run(
    dialogues: List[Mapping[str, Any]],
    *,
    agent_role: str,
    llm: Any,
    out_dir: Path,
    max_tokens: int,
    temperature: float,
    num_dialogues: Optional[int] = None,
) -> Dict[str, Any]:
    """Iterate gold chat_logs at decision-only mt1 turns; call the 70B."""
    out_dir.mkdir(parents=True, exist_ok=True)
    turns_path = out_dir / "turns.jsonl"
    parse_log = out_dir / "parse_failures.jsonl"

    n_queried = 0
    n_retried = 0
    n_fell_back = 0
    n_skipped_nonmatch = 0
    dialogue_stats: Dict[str, int] = {}
    start = time.time()

    with turns_path.open("w", encoding="utf-8") as fout:
        for di, dialogue in enumerate(dialogues):
            if num_dialogues is not None and di >= num_dialogues:
                break
            did = dialogue.get("dialogue_id", di)
            pinfo = dialogue.get("participant_info") or {}
            if agent_role not in pinfo:
                n_skipped_nonmatch += 1
                continue
            opp_role = _opponent_role(agent_role)
            agent_priorities = _get_priorities(pinfo.get(agent_role) or {})
            opp_priorities = _get_priorities(pinfo.get(opp_role) or {})

            agent = _build_agent(
                dialogue, agent_role, llm,
                max_tokens=max_tokens, temperature=temperature,
                parse_log=parse_log,
            )

            chat_logs = list(dialogue.get("chat_logs") or [])
            n_d = 0
            for t, turn in enumerate(chat_logs):
                if turn.get("id") != agent_role:
                    continue
                text = (turn.get("text") or "").strip()
                is_decision = any(text.startswith(a) for a in DECISION_ACTIONS)
                if not is_decision:
                    continue

                row = query_one_turn(
                    dialogue, t, agent,
                    agent_role=agent_role, opp_role=opp_role,
                    agent_priorities=agent_priorities,
                    opp_priorities=opp_priorities,
                )
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                n_queried += 1
                n_d += 1
                if row.get("retried"):
                    n_retried += 1
                if row.get("fell_back"):
                    n_fell_back += 1

                # Progress log every 25 calls.
                if n_queried % 25 == 0:
                    elapsed = time.time() - start
                    rate = n_queried / elapsed if elapsed > 0 else 0.0
                    logger.info(
                        "progress: %d calls, %.1f s elapsed, %.2f calls/s "
                        "(d%s, t%d)",
                        n_queried, elapsed, rate, did, t,
                    )

            dialogue_stats[str(did)] = n_d

    summary = {
        "n_dialogues":       len(dialogue_stats),
        "n_queried":         n_queried,
        "n_retried":         n_retried,
        "n_fell_back":       n_fell_back,
        "n_parse_retries":   n_retried,
        "n_parse_fallbacks": n_fell_back,
        "elapsed_seconds":   time.time() - start,
        "per_dialogue_queries": dialogue_stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/casino_test.json"))
    ap.add_argument("--num-dialogues", type=int, default=150)
    ap.add_argument("--agent-role", default="mturk_agent_1",
                    choices=("mturk_agent_1", "mturk_agent_2"))
    ap.add_argument("--backend", choices=("llama_70b", "dummy"),
                    default="llama_70b")
    ap.add_argument("--model-id", default=None,
                    help="Override --backend llama_70b's model id.")
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--run-name", default="protocol3_70b_decision_only")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-6s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    logger.info("Args: %s", vars(args))

    out_dir = Path("structured_cot/results") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(
        json.dumps({**vars(args), "data": str(args.data)}, indent=2, default=str)
    )

    dialogues = json.loads(args.data.read_text())
    logger.info("Loaded %d dialogues from %s", len(dialogues), args.data)

    if args.backend == "dummy":
        llm: Any = DummyStructuredLLM()
    else:
        llm = StructuredLLMClient(
            model_id=args.model_id or "meta-llama/Llama-3.3-70B-Instruct",
            default_max_tokens=args.max_tokens,
            default_temperature=args.temperature,
            seed=args.seed,
        )

    summary = run(
        dialogues,
        agent_role=args.agent_role,
        llm=llm,
        out_dir=out_dir,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        num_dialogues=args.num_dialogues,
    )

    logger.info(
        "Done. %d queries in %.1f s (%.2f s/call). retries=%d fallbacks=%d",
        summary["n_queried"],
        summary["elapsed_seconds"],
        summary["elapsed_seconds"] / max(1, summary["n_queried"]),
        summary["n_retried"],
        summary["n_fell_back"],
    )


if __name__ == "__main__":
    main()
