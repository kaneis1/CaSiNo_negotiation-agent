"""Protocol 1 smoke test: self-play one side, replay the other from logs.

Usage
-----
    python -m structured_cot.run_protocol1 \
        --data data/casino_test.json \
        --num-dialogues 10 \
        --agent-role mturk_agent_1 \
        --backend llama_70b                      # or 'dummy' for CPU smoke

Output artifacts (one run = one timestamped subdirectory):
    structured_cot/results/protocol1_<timestamp>/
        turns.jsonl                  # one line per agent turn, full trace
        dialogues.jsonl              # one line per dialogue, summary
        parse_failures.jsonl         # only populated on parse errors
        summary.json                 # aggregate stats (parse-error rate, etc.)
        args.json                    # this run's CLI arguments
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from structured_cot.agent import ActResult, StructuredCoTAgent
from structured_cot.llm_client import DummyStructuredLLM, StructuredLLMClient

logger = logging.getLogger("structured_cot.run_protocol1")

ITEMS = ("Food", "Water", "Firewood")
MAX_AGENT_TURNS = 25  # safety cap against runaway loops


# ── Deal action helpers ────────────────────────────────────────────────────


def _coerce_counts(td_map: Mapping[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for it in ITEMS:
        v = td_map.get(it, 0)
        try:
            out[it] = int(v)
        except (TypeError, ValueError):
            out[it] = 0
    return out


def render_opponent_deal_action(turn: Mapping[str, Any]) -> str:
    """Render a Submit-/Accept-/Reject-/Walk-Away turn as text the agent sees.

    When the opponent submits a deal, ``issue2youget`` / ``issue2theyget``
    are from the OPPONENT's perspective. We flip so the agent reads:
    "Opponent proposes: you get ..., they get ...".
    """
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
        return "Accept-Deal — Opponent accepts the last proposal on the table."
    if text.startswith("Reject-Deal"):
        return "Reject-Deal — Opponent rejects the last proposal."
    if text.startswith("Walk-Away"):
        return "Walk-Away — Opponent walked away (both sides get the 5-point fallback)."
    return text


def render_agent_counter_offer(co: Mapping[str, int]) -> str:
    you = _coerce_counts(co)
    opp = {it: 3 - you[it] for it in ITEMS}
    return (
        f"Submit-Deal — I propose: "
        f"I get Food={you['Food']}, Water={you['Water']}, Firewood={you['Firewood']}; "
        f"you get Food={opp['Food']}, Water={opp['Water']}, Firewood={opp['Firewood']}."
    )


def is_deal_action(turn: Mapping[str, Any]) -> bool:
    t = (turn.get("text") or "").strip()
    return t.startswith(("Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away"))


def counter_offer_is_legal(co: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(co, Mapping):
        return False
    for it in ITEMS:
        v = co.get(it)
        if not isinstance(v, int):
            return False
        if v < 0 or v > 3:
            return False
    return True


# ── Ground-truth extraction for the human decision at each agent turn ──────


def _ground_truth_decision(turn: Mapping[str, Any]) -> Dict[str, Any]:
    text = (turn.get("text") or "").strip()
    td = turn.get("task_data") or {}
    if text.startswith("Submit-Deal"):
        you_get = _coerce_counts(td.get("issue2youget", {}))
        return {"action": "reject", "counter_offer": you_get, "raw_text": text}
    if text.startswith("Accept-Deal"):
        return {"action": "accept", "counter_offer": None, "raw_text": text}
    if text.startswith("Walk-Away"):
        return {"action": "walkaway", "counter_offer": None, "raw_text": text}
    if text.startswith("Reject-Deal"):
        return {"action": "reject", "counter_offer": None, "raw_text": text}
    return {"action": None, "counter_offer": None, "raw_text": text}


# ── Dialogue replay ────────────────────────────────────────────────────────


def _opponent_role(agent_role: str) -> str:
    return "mturk_agent_2" if agent_role == "mturk_agent_1" else "mturk_agent_1"


def _get_priorities(info: Mapping[str, Any]) -> Dict[str, str]:
    """Map CaSiNo's value2issue into our High/Medium/Low dict.

    Handles both orientations seen in the wild.
    """
    v2i = info.get("value2issue") or {}
    if set(v2i.keys()) >= {"High", "Medium", "Low"}:
        return {k: v2i[k] for k in ("High", "Medium", "Low")}
    if set(v2i.values()) >= {"High", "Medium", "Low"}:
        flipped = {str(lvl): str(item) for item, lvl in v2i.items()}
        return {k: flipped[k] for k in ("High", "Medium", "Low")}
    raise ValueError(f"Cannot interpret value2issue: {v2i}")


def _get_arguments(info: Mapping[str, Any]) -> Dict[str, str]:
    v2r = info.get("value2reason") or {}
    if set(v2r.keys()) >= {"High", "Medium", "Low"}:
        return {k: v2r[k] for k in ("High", "Medium", "Low")}
    return {k: "" for k in ("High", "Medium", "Low")}


def replay_protocol1(
    dialogue: Mapping[str, Any],
    agent: StructuredCoTAgent,
    agent_role: str,
    *,
    max_turns: int = MAX_AGENT_TURNS,
) -> Dict[str, Any]:
    """Run one dialogue in Protocol 1 mode.

    Returns a summary dict + the per-turn trace list so the driver can
    write both dialogue-level and turn-level JSONL files.
    """
    opp_role = _opponent_role(agent_role)
    chat_logs = list(dialogue.get("chat_logs") or [])
    dialogue_id = dialogue.get("dialogue_id")
    logs_len = len(chat_logs)

    # Ground truth priorities for logging (agent does NOT see opponent's).
    participant_info = dialogue.get("participant_info") or {}
    agent_info = participant_info.get(agent_role) or {}
    opp_info = participant_info.get(opp_role) or {}
    agent_priorities = _get_priorities(agent_info)
    opp_priorities = _get_priorities(opp_info)

    agent.reset()
    # History is the list of (speaker, utterance) pairs fed to build_prompt.
    history: List[Tuple[str, str]] = []
    turn_traces: List[Dict[str, Any]] = []

    i = 0
    agent_turns_taken = 0
    outcome = "timeout"  # unless we hit a terminal state
    agent_final_decision: Optional[Dict[str, Any]] = None
    opp_last_offer: Optional[Dict[str, int]] = None  # tracks pending offer

    while i < logs_len:
        turn = chat_logs[i]
        speaker = turn.get("id")

        if speaker == opp_role:
            text = render_opponent_deal_action(turn)
            history.append(("opp", text))
            t_text = (turn.get("text") or "").strip()
            if t_text.startswith("Submit-Deal"):
                td = turn.get("task_data") or {}
                opp_last_offer = _coerce_counts(td.get("issue2theyget", {}))  # what agent is offered
            elif t_text.startswith("Accept-Deal"):
                outcome = "opp_accepted_agent_offer"
                break
            elif t_text.startswith("Walk-Away"):
                outcome = "opp_walkaway"
                break
            i += 1
            continue

        # It's the agent's turn.
        if agent_turns_taken >= max_turns:
            outcome = "max_turns_exceeded"
            break

        pending = {"Food": opp_last_offer["Food"],
                   "Water": opp_last_offer["Water"],
                   "Firewood": opp_last_offer["Firewood"]} if opp_last_offer else None

        ground_truth = _ground_truth_decision(turn)
        ground_truth_utterance = (turn.get("text") or "").strip()

        result: ActResult = agent.act(history, pending_offer=pending)
        parsed = result.parsed
        decision = parsed.get("decision") or {}
        action = decision.get("action")
        co = decision.get("counter_offer")

        if action == "reject" and co is not None and counter_offer_is_legal(co):
            agent_message = render_agent_counter_offer(co)
        elif action == "reject":
            agent_message = (parsed.get("utterance") or "").strip() or "(agent said nothing)"
        elif action == "accept":
            agent_message = "Accept-Deal — I accept the proposal on the table."
        elif action == "walkaway":
            agent_message = "Walk-Away — I am walking away from the negotiation."
        else:
            agent_message = (parsed.get("utterance") or "").strip() or "(agent said nothing)"

        history.append(("me", agent_message))

        turn_traces.append({
            "dialogue_id":            dialogue_id,
            "turn_index":             i,
            "agent_turn_index":       agent_turns_taken,
            "agent_role":             agent_role,
            "agent_priorities":       agent_priorities,
            "opponent_priorities":    opp_priorities,  # ground-truth, NOT shown to agent
            "history_before":         [list(p) for p in history[:-1]],
            "pending_offer":          pending,
            "raw_output_first":       result.raw_first,
            "raw_output_retry":       result.raw_retry,
            "retried":                result.retried,
            "fell_back":              result.fell_back,
            "parse_errors":           result.parse_errors,
            "elapsed_seconds":        result.elapsed_seconds,
            "parsed_observation":     parsed.get("observation"),
            "parsed_opponent_inference": parsed.get("opponent_inference"),
            "parsed_plan":            parsed.get("plan"),
            "parsed_utterance":       parsed.get("utterance"),
            "parsed_decision":        decision,
            "ground_truth_decision":  ground_truth,
            "ground_truth_utterance": ground_truth_utterance,
            "counter_offer_legal":    counter_offer_is_legal(co) if co is not None else None,
        })
        agent_turns_taken += 1

        if action == "accept":
            agent_final_decision = dict(decision)
            outcome = "agent_accepted"
            break
        if action == "walkaway":
            agent_final_decision = dict(decision)
            outcome = "agent_walkaway"
            break
        # reject continues; consume the opponent's next turn next loop iter
        if action == "reject" and co is not None:
            opp_last_offer = None  # agent just made its own proposal
        i += 1

    # If we exhausted chat_logs without a terminal event above:
    if outcome == "timeout" and i >= logs_len:
        outcome = "chat_log_exhausted"

    n_agent_turns = len(turn_traces)
    n_parse_retried = sum(1 for t in turn_traces if t["retried"])
    n_fell_back = sum(1 for t in turn_traces if t["fell_back"])
    n_illegal_co = sum(
        1 for t in turn_traces if t["counter_offer_legal"] is False
    )

    summary = {
        "dialogue_id":      dialogue_id,
        "agent_role":       agent_role,
        "opponent_role":    opp_role,
        "agent_priorities": agent_priorities,
        "opponent_priorities": opp_priorities,
        "n_agent_turns":    n_agent_turns,
        "n_retried":        n_parse_retried,
        "n_fell_back":      n_fell_back,
        "n_illegal_counter_offer": n_illegal_co,
        "outcome":          outcome,
        "agent_final_decision": agent_final_decision,
        "total_elapsed_seconds": sum(t["elapsed_seconds"] for t in turn_traces),
    }
    return {"summary": summary, "turns": turn_traces}


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
    parser.add_argument("--data", type=Path, default=Path("data/casino_test.json"))
    parser.add_argument("--num-dialogues", type=int, default=10)
    parser.add_argument("--agent-role", default="mturk_agent_1",
                        choices=["mturk_agent_1", "mturk_agent_2"])
    parser.add_argument("--backend", default="llama_70b",
                        choices=["llama_70b", "llama_8b", "dummy", "custom"])
    parser.add_argument("--model-id", default=None,
                        help="Only used with --backend custom.")
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max-turns", type=int, default=MAX_AGENT_TURNS,
                        help="Safety cap on agent turns per dialogue.")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle dialogues before slicing --num-dialogues.")
    parser.add_argument("--output-root", type=Path,
                        default=Path("structured_cot/results"))
    parser.add_argument("--run-name", default=None,
                        help="Subdirectory name; defaults to timestamped.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    with args.data.open() as f:
        dialogues = json.load(f)
    if not isinstance(dialogues, list):
        raise ValueError(f"Expected list of dialogues at {args.data}")

    if args.shuffle:
        rng = random.Random(args.seed)
        dialogues = list(dialogues)
        rng.shuffle(dialogues)
    selected = dialogues[: args.num_dialogues]
    logger.info("Selected %d / %d dialogues", len(selected), len(dialogues))

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"protocol1_{args.backend}_{timestamp}"
    out_dir = args.output_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing run artifacts to %s", out_dir)

    with (out_dir / "args.json").open("w") as f:
        json.dump(
            {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
             "timestamp": timestamp},
            f, indent=2,
        )

    llm = _build_llm(args.backend, model_id=args.model_id, seed=args.seed)
    parse_log = out_dir / "parse_failures.jsonl"

    turns_path = out_dir / "turns.jsonl"
    dialogues_path = out_dir / "dialogues.jsonl"

    agg = {
        "n_dialogues":           0,
        "n_agent_turns":         0,
        "n_parse_retries":       0,
        "n_parse_fallbacks":     0,
        "n_illegal_counter_offers": 0,
        "outcomes":              {},
        "per_dialogue_elapsed":  [],
    }

    t_start = time.time()
    with turns_path.open("w") as turns_f, dialogues_path.open("w") as dlg_f:
        for idx, dialogue in enumerate(selected):
            dialogue_id = dialogue.get("dialogue_id")
            logger.info("[%d/%d] dialogue_id=%s", idx + 1, len(selected), dialogue_id)

            info = dialogue.get("participant_info", {}).get(args.agent_role, {})
            try:
                priorities = _get_priorities(info)
                arguments = _get_arguments(info)
            except Exception as e:
                logger.warning("Skipping %s: cannot parse priorities (%s)", dialogue_id, e)
                continue

            agent = StructuredCoTAgent(
                priorities=priorities,
                arguments=arguments,
                llm_client=llm,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                parse_log_path=parse_log,
            )

            try:
                result = replay_protocol1(
                    dialogue, agent, args.agent_role, max_turns=args.max_turns,
                )
            except Exception:
                logger.exception("Failed on dialogue %s; continuing", dialogue_id)
                continue

            for t in result["turns"]:
                turns_f.write(json.dumps(t) + "\n")
            turns_f.flush()
            dlg_f.write(json.dumps(result["summary"]) + "\n")
            dlg_f.flush()

            s = result["summary"]
            agg["n_dialogues"] += 1
            agg["n_agent_turns"] += s["n_agent_turns"]
            agg["n_parse_retries"] += s["n_retried"]
            agg["n_parse_fallbacks"] += s["n_fell_back"]
            agg["n_illegal_counter_offers"] += s["n_illegal_counter_offer"]
            agg["outcomes"][s["outcome"]] = agg["outcomes"].get(s["outcome"], 0) + 1
            agg["per_dialogue_elapsed"].append(s["total_elapsed_seconds"])

    agg["wallclock_seconds"] = time.time() - t_start
    if agg["n_agent_turns"] > 0:
        agg["parse_retry_rate"]    = agg["n_parse_retries"]    / agg["n_agent_turns"]
        agg["parse_fallback_rate"] = agg["n_parse_fallbacks"]  / agg["n_agent_turns"]
        agg["illegal_counter_offer_rate"] = (
            agg["n_illegal_counter_offers"] / agg["n_agent_turns"]
        )
    else:
        agg["parse_retry_rate"] = 0.0
        agg["parse_fallback_rate"] = 0.0
        agg["illegal_counter_offer_rate"] = 0.0

    with (out_dir / "summary.json").open("w") as f:
        json.dump(agg, f, indent=2)

    print("=" * 72)
    print(f"Run: {out_dir}")
    print(f"  dialogues:          {agg['n_dialogues']}")
    print(f"  agent turns:        {agg['n_agent_turns']}")
    print(f"  parse retry rate:   {agg['parse_retry_rate']:.3%}")
    print(f"  parse fallback rate:{agg['parse_fallback_rate']:.3%}")
    print(f"  illegal offers:     {agg['n_illegal_counter_offers']} "
          f"({agg['illegal_counter_offer_rate']:.3%})")
    print(f"  outcomes:           {agg['outcomes']}")
    print(f"  wallclock:          {agg['wallclock_seconds']:.1f}s")
    print("=" * 72)

    if agg["parse_fallback_rate"] > 0.05:
        logger.warning(
            "Parse fallback rate %.2f%% > 5%% — prompt likely needs work.",
            agg["parse_fallback_rate"] * 100.0,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
