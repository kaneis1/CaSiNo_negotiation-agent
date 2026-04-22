"""Protocol 2 driver: Structured-CoT agent vs retrieval opponent.

Unlike Protocol 1 (opponent replayed verbatim from the test set, cannot
react) and unlike pure self-play (both sides are the same model), the
opponent here is a kNN-retriever over ``data/casino_train.json`` bucketed
by the opponent's priority ordering. Every opponent turn comes from
real human negotiation text, but is selected in response to *our agent's*
utterances, so the dialogue is actually adaptive.

What this produces is what the spec asked for: **end-to-end outcomes**.
Per-dialogue we record the final deal (if any), both sides' CaSiNo points,
the agent's Pareto-normalized share, termination reason, and turn count;
aggregate metrics are in ``summary.json``.

Usage:
    python -m structured_cot.run_protocol2 \
        --test-data  data/casino_test.json \
        --train-data data/casino_train.json \
        --num-dialogues 150 \
        --backend llama_70b
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from structured_cot.agent import StructuredCoTAgent
from structured_cot.llm_client import DummyStructuredLLM, StructuredLLMClient
from structured_cot.retrieval_opponent import (
    DEAL_ACTIONS,
    ITEMS,
    WALKAWAY_POINTS,
    RetrievalOpponent,
    build_retrieval_pool,
    load_training_corpus,
    pareto_max_self,
    points_for,
)

logger = logging.getLogger("structured_cot.run_protocol2")


MAX_AGENT_TURNS = 25  # matches Protocol 1's safety cap


# ── Priorities + rendering ────────────────────────────────────────────────


def _get_priorities_triple(info: Mapping[str, Any]) -> Tuple[str, str, str]:
    p = (info or {}).get("value2issue") or {}
    if not set(p.keys()) >= {"High", "Medium", "Low"}:
        raise ValueError(f"Cannot interpret value2issue: {p}")
    return (p["High"], p["Medium"], p["Low"])


def _get_priorities_dict(info: Mapping[str, Any]) -> Dict[str, str]:
    p = (info or {}).get("value2issue") or {}
    if not set(p.keys()) >= {"High", "Medium", "Low"}:
        raise ValueError(f"Cannot interpret value2issue: {p}")
    return {k: p[k] for k in ("High", "Medium", "Low")}


def _get_arguments(info: Mapping[str, Any]) -> Dict[str, str]:
    v2r = (info or {}).get("value2reason") or {}
    if set(v2r.keys()) >= {"High", "Medium", "Low"}:
        return {k: v2r[k] for k in ("High", "Medium", "Low")}
    return {k: "" for k in ("High", "Medium", "Low")}


def _coerce_counts(td_map: Mapping[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for it in ITEMS:
        v = td_map.get(it, 0) if isinstance(td_map, Mapping) else 0
        try:
            out[it] = int(v)
        except (TypeError, ValueError):
            out[it] = 0
    return out


def _render_opponent_turn(turn: Mapping[str, Any]) -> str:
    """Render a retrieved opponent turn into a single history line for the agent."""
    text = (turn.get("text") or "").strip()
    td = turn.get("task_data") or {}
    if text.startswith("Submit-Deal"):
        opp_gets = _coerce_counts(td.get("issue2youget", {}))
        you_get = _coerce_counts(td.get("issue2theyget", {}))
        return (
            f"Submit-Deal — Opponent proposes: "
            f"you get Food={you_get['Food']}, Water={you_get['Water']}, "
            f"Firewood={you_get['Firewood']}; "
            f"they get Food={opp_gets['Food']}, Water={opp_gets['Water']}, "
            f"Firewood={opp_gets['Firewood']}."
        )
    if text.startswith("Accept-Deal"):
        return "Accept-Deal — Opponent accepts the proposal on the table."
    if text.startswith("Reject-Deal"):
        return "Reject-Deal — Opponent rejects the proposal."
    if text.startswith("Walk-Away"):
        return "Walk-Away — Opponent walked away (both sides get the 5-point fallback)."
    return text


def _render_agent_offer(co: Mapping[str, int]) -> str:
    you = _coerce_counts(co)
    opp = {it: 3 - you[it] for it in ITEMS}
    return (
        f"Submit-Deal — I propose: "
        f"I get Food={you['Food']}, Water={you['Water']}, Firewood={you['Firewood']}; "
        f"you get Food={opp['Food']}, Water={opp['Water']}, Firewood={opp['Firewood']}."
    )


def _counter_offer_is_legal(co: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(co, Mapping):
        return False
    for it in ITEMS:
        v = co.get(it)
        if not isinstance(v, int) or v < 0 or v > 3:
            return False
    return True


# ── Single-dialogue driver ─────────────────────────────────────────────────


def simulate_dialogue(
    *,
    dialogue: Mapping[str, Any],
    agent: StructuredCoTAgent,
    opponent: RetrievalOpponent,
    agent_priorities_triple: Tuple[str, str, str],
    opp_priorities_triple:   Tuple[str, str, str],
    max_turns: int = MAX_AGENT_TURNS,
) -> Dict[str, Any]:
    dialogue_id = dialogue.get("dialogue_id")
    agent.reset()

    history: List[Tuple[str, str]] = []
    trace: List[Dict[str, Any]] = []

    agent_last_offer: Optional[Dict[str, int]] = None  # what agent receives
    opp_last_offer:   Optional[Dict[str, int]] = None  # what agent is offered

    outcome = "timeout"
    final_deal_agent: Optional[Dict[str, int]] = None   # counts agent receives
    agent_turns = 0
    turn_index = 0

    # Opponent opens (CaSiNo convention: mturk_agent_2 usually starts).
    opp_first = opponent.respond(priorities=opp_priorities_triple, context=[])
    first_text = (opp_first.get("text") or "").strip()
    if first_text:
        rendered = _render_opponent_turn(opp_first)
        history.append(("opp", rendered))
        trace.append({
            "turn_index":     turn_index,
            "speaker":        "opp",
            "type":           "opener",
            "raw_text":       first_text,
            "rendered":       rendered,
            "retrieved_from": opp_first.get("retrieved_from"),
        })
        turn_index += 1
        if first_text.startswith("Submit-Deal"):
            td = opp_first.get("task_data") or {}
            opp_last_offer = _coerce_counts(td.get("issue2theyget", {}))

    while agent_turns < max_turns:
        # ── Agent turn ───────────────────────────────────────────────────
        pending = dict(opp_last_offer) if opp_last_offer else None
        result = agent.act(history, pending_offer=pending)
        parsed = result.parsed
        decision = parsed.get("decision") or {}
        action = decision.get("action")
        co = decision.get("counter_offer")

        agent_event: Dict[str, Any] = {
            "turn_index":      turn_index,
            "speaker":         "agent",
            "type":            "utterance",
            "action":          action,
            "counter_offer":   co,
            "retried":         result.retried,
            "fell_back":       result.fell_back,
            "elapsed_seconds": result.elapsed_seconds,
            "parse_errors":    result.parse_errors,
            "parsed_plan":     parsed.get("plan"),
            "parsed_utterance": parsed.get("utterance"),
            "parsed_observation": parsed.get("observation"),
            "parsed_opponent_inference": parsed.get("opponent_inference"),
        }

        if action == "reject" and _counter_offer_is_legal(co):
            agent_msg = _render_agent_offer(co)
            agent_last_offer = _coerce_counts(co)
            opp_last_offer = None  # agent's new proposal overwrites pending
        elif action == "reject":
            agent_msg = (parsed.get("utterance") or "").strip() or "(agent said nothing)"
        elif action == "accept":
            if opp_last_offer is not None:
                agent_msg = "Accept-Deal — I accept the proposal on the table."
                final_deal_agent = dict(opp_last_offer)
                outcome = "agent_accepted"
                agent_event["type"] = "accept"
                history.append(("me", agent_msg))
                trace.append(agent_event)
                agent_turns += 1
                turn_index += 1
                break
            # Phantom accept: agent tried to accept with no offer on the table.
            agent_msg = "Accept-Deal — (no offer was on the table)"
            outcome = "agent_phantom_accept"
            agent_event["type"] = "phantom_accept"
            history.append(("me", agent_msg))
            trace.append(agent_event)
            agent_turns += 1
            turn_index += 1
            break
        elif action == "walkaway":
            agent_msg = "Walk-Away — I am walking away."
            outcome = "agent_walkaway"
            agent_event["type"] = "walkaway"
            history.append(("me", agent_msg))
            trace.append(agent_event)
            agent_turns += 1
            turn_index += 1
            break
        else:
            agent_msg = (parsed.get("utterance") or "").strip() or "(agent said nothing)"

        history.append(("me", agent_msg))
        trace.append(agent_event)
        agent_turns += 1
        turn_index += 1

        # ── Opponent turn ────────────────────────────────────────────────
        opp_reply = opponent.respond(
            priorities=opp_priorities_triple,
            context=history,
        )
        opp_text = (opp_reply.get("text") or "").strip()
        rendered = _render_opponent_turn(opp_reply)
        history.append(("opp", rendered))
        opp_event: Dict[str, Any] = {
            "turn_index":     turn_index,
            "speaker":        "opp",
            "type":           "utterance",
            "raw_text":       opp_text,
            "rendered":       rendered,
            "retrieved_from": opp_reply.get("retrieved_from"),
        }
        turn_index += 1

        if opp_text.startswith("Submit-Deal"):
            td = opp_reply.get("task_data") or {}
            opp_last_offer = _coerce_counts(td.get("issue2theyget", {}))
            opp_event["type"] = "submit"
            trace.append(opp_event)
            continue

        if opp_text.startswith("Accept-Deal"):
            opp_event["type"] = "accept"
            trace.append(opp_event)
            if agent_last_offer is not None:
                final_deal_agent = dict(agent_last_offer)
                outcome = "opp_accepted"
            else:
                outcome = "opp_accept_without_offer"
            break

        if opp_text.startswith("Walk-Away"):
            opp_event["type"] = "walkaway"
            trace.append(opp_event)
            outcome = "opp_walkaway"
            break

        if opp_text.startswith("Reject-Deal"):
            opp_event["type"] = "reject"
            trace.append(opp_event)
            # Reject clears any pending offer the opponent may have had on record
            # (the opponent rejected it). Agent proposal still stands.
            continue

        trace.append(opp_event)

    if agent_turns >= max_turns and outcome == "timeout":
        outcome = "max_turns_exceeded"

    # ── Score the outcome ─────────────────────────────────────────────────
    agent_points: int
    opp_points:   int
    if final_deal_agent is not None:
        agent_counts = final_deal_agent
        opp_counts = {it: 3 - agent_counts[it] for it in ITEMS}
        agent_points = points_for(agent_counts, agent_priorities_triple)
        opp_points   = points_for(opp_counts,   opp_priorities_triple)
    else:
        agent_counts = None
        opp_counts = None
        agent_points = WALKAWAY_POINTS
        opp_points   = WALKAWAY_POINTS

    pareto_agent = pareto_max_self(agent_priorities_triple, opp_priorities_triple)
    pareto_opp   = pareto_max_self(opp_priorities_triple,   agent_priorities_triple)

    summary = {
        "dialogue_id":            dialogue_id,
        "outcome":                outcome,
        "agent_priorities":       list(agent_priorities_triple),
        "opponent_priorities":    list(opp_priorities_triple),
        "agent_counts":           agent_counts,
        "opp_counts":             opp_counts,
        "agent_points":           agent_points,
        "opp_points":             opp_points,
        "joint_points":           agent_points + opp_points,
        "pareto_max_agent":       pareto_agent,
        "pareto_max_opp":         pareto_opp,
        "agent_pareto_share":     agent_points / pareto_agent,
        "opp_pareto_share":       opp_points / pareto_opp,
        "n_agent_turns":          agent_turns,
        "n_total_turns":          turn_index,
        "n_parse_retries":        sum(1 for e in trace if e.get("retried")),
        "n_parse_fallbacks":      sum(1 for e in trace if e.get("fell_back")),
        "agreement":              final_deal_agent is not None,
        "walkaway":               outcome in ("agent_walkaway", "opp_walkaway"),
    }
    return {"summary": summary, "trace": trace}


# ── CLI driver ─────────────────────────────────────────────────────────────


def _build_llm(backend: str, *, model_id: Optional[str], seed: int):
    if backend == "dummy":
        return DummyStructuredLLM()
    if backend == "llama_70b":
        return StructuredLLMClient(seed=seed)
    if backend == "llama_8b":
        from structured_cot.llm_client import LLAMA_31_8B_DEFAULT
        return StructuredLLMClient(model_id=LLAMA_31_8B_DEFAULT, seed=seed)
    if backend == "custom":
        if not model_id:
            raise ValueError("--backend custom requires --model-id")
        return StructuredLLMClient(model_id=model_id, seed=seed)
    raise ValueError(f"Unknown backend: {backend}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-data",  type=Path, default=Path("data/casino_test.json"))
    parser.add_argument("--train-data", type=Path, default=Path("data/casino_train.json"))
    parser.add_argument("--num-dialogues", type=int, default=150)
    parser.add_argument("--backend", default="llama_70b",
                        choices=["llama_70b", "llama_8b", "dummy", "custom"])
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--max-tokens",  type=int,   default=800)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--retrieval-temperature", type=float, default=0.7,
                        help="Sampling temperature over TF-IDF cosine top-K. "
                             "0 = argmax.")
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--retrieval-context-turns", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max-turns", type=int, default=MAX_AGENT_TURNS)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--output-root", type=Path,
                        default=Path("structured_cot/results"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"protocol2_{args.backend}_{timestamp}"
    out_dir = args.output_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing run artifacts to %s", out_dir)

    with (out_dir / "args.json").open("w") as f:
        json.dump(
            {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
             "timestamp": timestamp},
            f, indent=2,
        )

    logger.info("Loading test + train data …")
    with args.test_data.open() as f:
        test_dialogues = json.load(f)
    train_corpus = load_training_corpus(args.train_data)
    logger.info("Test: %d dialogues; Train: %d dialogues.",
                len(test_dialogues), len(train_corpus))

    if args.shuffle:
        rng = random.Random(args.seed)
        test_dialogues = list(test_dialogues)
        rng.shuffle(test_dialogues)
    selected = test_dialogues[: args.num_dialogues]
    logger.info("Selected %d / %d test dialogues.",
                len(selected), len(test_dialogues))

    logger.info("Building retrieval pool + indices lazily …")
    pool = build_retrieval_pool(train_corpus, context_turns=args.retrieval_context_turns)
    opponent = RetrievalOpponent(
        pool,
        context_turns=args.retrieval_context_turns,
        top_k=args.retrieval_top_k,
        temperature=args.retrieval_temperature,
        seed=args.seed,
    )

    llm = _build_llm(args.backend, model_id=args.model_id, seed=args.seed)
    parse_log = out_dir / "parse_failures.jsonl"

    summaries_path = out_dir / "dialogues.jsonl"
    trace_path     = out_dir / "traces.jsonl"

    agg: Dict[str, Any] = {
        "n_dialogues":            0,
        "n_agreement":            0,
        "n_walkaway":             0,
        "n_phantom_accept":       0,
        "n_max_turns":            0,
        "outcomes":               {},
        "agent_points":           [],
        "opp_points":             [],
        "agent_pareto_shares":    [],
        "opp_pareto_shares":      [],
        "joint_points":           [],
        "agent_turns":            [],
        "parse_retries":          0,
        "parse_fallbacks":        0,
    }

    t_start = time.time()
    with summaries_path.open("w") as sf, trace_path.open("w") as tf:
        for idx, dialogue in enumerate(selected):
            did = dialogue.get("dialogue_id")
            info = dialogue.get("participant_info", {}) or {}
            try:
                agent_pri_triple = _get_priorities_triple(info.get("mturk_agent_1") or {})
                opp_pri_triple   = _get_priorities_triple(info.get("mturk_agent_2") or {})
            except Exception as e:
                logger.warning("Skipping %s: %s", did, e)
                continue

            agent = StructuredCoTAgent(
                priorities=_get_priorities_dict(info.get("mturk_agent_1") or {}),
                arguments=_get_arguments(info.get("mturk_agent_1") or {}),
                llm_client=llm,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                parse_log_path=parse_log,
            )

            logger.info("[%d/%d] dialogue_id=%s  agent=%s vs opp=%s",
                        idx + 1, len(selected), did, agent_pri_triple, opp_pri_triple)

            try:
                res = simulate_dialogue(
                    dialogue=dialogue,
                    agent=agent,
                    opponent=opponent,
                    agent_priorities_triple=agent_pri_triple,
                    opp_priorities_triple=opp_pri_triple,
                    max_turns=args.max_turns,
                )
            except Exception:
                logger.exception("Failed on dialogue %s; continuing", did)
                continue

            s = res["summary"]
            sf.write(json.dumps(s) + "\n"); sf.flush()
            tf.write(json.dumps({"dialogue_id": did, "trace": res["trace"]}) + "\n")
            tf.flush()

            agg["n_dialogues"] += 1
            agg["n_agreement"] += int(s["agreement"])
            agg["n_walkaway"]  += int(s["walkaway"])
            agg["n_phantom_accept"] += int(s["outcome"] == "agent_phantom_accept")
            agg["n_max_turns"] += int(s["outcome"] == "max_turns_exceeded")
            agg["outcomes"][s["outcome"]] = agg["outcomes"].get(s["outcome"], 0) + 1
            agg["agent_points"].append(s["agent_points"])
            agg["opp_points"].append(s["opp_points"])
            agg["agent_pareto_shares"].append(s["agent_pareto_share"])
            agg["opp_pareto_shares"].append(s["opp_pareto_share"])
            agg["joint_points"].append(s["joint_points"])
            agg["agent_turns"].append(s["n_agent_turns"])
            agg["parse_retries"] += s["n_parse_retries"]
            agg["parse_fallbacks"] += s["n_parse_fallbacks"]

    def _mean(xs):
        return (sum(xs) / len(xs)) if xs else float("nan")

    summary = {
        "n_dialogues":              agg["n_dialogues"],
        "agreement_rate":           agg["n_agreement"] / agg["n_dialogues"]
                                    if agg["n_dialogues"] else float("nan"),
        "walkaway_rate":            agg["n_walkaway"] / agg["n_dialogues"]
                                    if agg["n_dialogues"] else float("nan"),
        "phantom_accept_rate":      agg["n_phantom_accept"] / agg["n_dialogues"]
                                    if agg["n_dialogues"] else float("nan"),
        "max_turns_rate":           agg["n_max_turns"] / agg["n_dialogues"]
                                    if agg["n_dialogues"] else float("nan"),
        "mean_agent_points":        _mean(agg["agent_points"]),
        "mean_opp_points":          _mean(agg["opp_points"]),
        "mean_joint_points":        _mean(agg["joint_points"]),
        "mean_agent_pareto_share":  _mean(agg["agent_pareto_shares"]),
        "mean_opp_pareto_share":    _mean(agg["opp_pareto_shares"]),
        "mean_agent_turns":         _mean(agg["agent_turns"]),
        "parse_retries":            agg["parse_retries"],
        "parse_fallbacks":          agg["parse_fallbacks"],
        "parse_fallback_rate":      (agg["parse_fallbacks"] / sum(agg["agent_turns"]))
                                    if sum(agg["agent_turns"]) else float("nan"),
        "outcomes":                 agg["outcomes"],
        "wallclock_seconds":        time.time() - t_start,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print(f"Run: {out_dir}")
    print(f"  dialogues            : {summary['n_dialogues']}")
    print(f"  agreement rate       : {summary['agreement_rate']:.2%}")
    print(f"  walkaway rate        : {summary['walkaway_rate']:.2%}")
    print(f"  phantom-accept rate  : {summary['phantom_accept_rate']:.2%}")
    print(f"  max-turns rate       : {summary['max_turns_rate']:.2%}")
    print(f"  mean agent points    : {summary['mean_agent_points']:.2f}")
    print(f"  mean opp points      : {summary['mean_opp_points']:.2f}")
    print(f"  mean joint points    : {summary['mean_joint_points']:.2f}")
    print(f"  mean agent Pareto    : {summary['mean_agent_pareto_share']:.2%}")
    print(f"  mean opp Pareto      : {summary['mean_opp_pareto_share']:.2%}")
    print(f"  mean agent turns/dlg : {summary['mean_agent_turns']:.1f}")
    print(f"  parse fallback rate  : {summary['parse_fallback_rate']:.2%}")
    print(f"  outcomes             : {summary['outcomes']}")
    print(f"  wallclock            : {summary['wallclock_seconds']:.1f}s")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
