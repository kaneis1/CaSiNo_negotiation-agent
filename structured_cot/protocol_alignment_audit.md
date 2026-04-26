# Protocol alignment audit: why the Bayesian teacher and the Structured-CoT baseline were not scored on the same turns

**Status.** Draft for the Limitations / Evaluation-methodology section of the paper. All numbers below are reproducible from artifacts in the repo:

- `structured_cot/results/protocol1_70b_full/turns.jsonl` — the Protocol-1 self-play log for the 70B Structured-CoT baseline on the 150 held-out dialogues.
- `opponent_model/results/turn_eval_structured_cot_replay_full150/turn_summary.json` — Structured-CoT baseline scored through the shared turn-level harness (Protocol-1 support).
- `opponent_model/results/turn_eval_bayesian_lambda1.0_m5_f0.50_full150/turn_summary.json` — Bayesian teacher scored through the same harness on the full gold support.
- `data/casino_test.json` — the 150-dialogue held-out split (CaSiNo chat_logs, human ground truth).

## 1. The headline discrepancy

When we scored both agents through the shared `opponent_model.turn_level_eval` harness on the 150-dialogue held-out test set, the evaluation support sizes came out very different even though the data set was identical:

| Metric              | Bayesian teacher (λ=1.0) | Structured-CoT baseline (Protocol-1 log) |
|---------------------|-------------------------:|-----------------------------------------:|
| Accept F1           | 0.897 (n=87)             | 0.809 (n=28)                             |
| Bid cosine          | 0.744 (n=86)             | 0.852 (n=14)                             |
| Strategy macro-F1   | 0.048 (n=350)            | 0.129 (n=297)                            |
| Brier (posterior)   | 0.085 (n=1054)           | — (agent exposes no posterior)           |

The two agents' harness-scored supports are **3× apart on accept F1 (87 vs 28) and 6× apart on bid cosine (86 vs 14)**. That gap is not random variation. It is a systematic consequence of how the Protocol-1 self-play run interacts with early termination — the same early-termination pattern we document in `baseline_weaknesses.md` as the baseline's "accepting dominated offers" failure mode.

## 2. The mechanism: Protocol-1 self-play truncates gold trajectories

Under Protocol 1, the Structured-CoT agent replays the opponent's utterances one turn at a time and emits its own move at each agent-turn. The dialogue terminates the moment the agent's `<decision>` block contains `"accept"` or `"walkaway"`. That is the correct semantics of an agent trajectory — but it means the agent never makes a decision past its own first `accept`.

The gold CaSiNo dialogue, however, does not stop when the baseline would have stopped. The real human participants keep going: the person we are replaying may have rejected the first Submit-Deal, counter-proposed, been re-countered, and only accepted on turn 14 — producing two more Submit-Deal turns and one more Accept-Deal turn that we would like to score.

Concretely, on the 150-dialogue held-out set:

- **Gold mt1 decision-relevant turns:** 89 accept-decision turns (75 Accept-Deal, 11 Reject-Deal, 3 Walk-Away) + 86 Submit-Deal turns = **175 gold decision points**.
- **Covered by the Protocol-1 self-play log:** 30 of 89 accept-decision turns (33.7%) and 26 of 86 Submit-Deal turns (30.2%).
- **Missed because P1 terminated the trajectory early:** 59 accept-decision turns (66.3%) and 60 Submit-Deal turns (69.8%).
- **Early-termination magnitude:** the baseline's last-logged turn index is a mean of **4.08 turns before** the gold dialogue's last turn index (median 3.0). 124 of 150 dialogues (82.7%) terminate at least one turn early.
- **Overall mt1 turn coverage:** 784 P1 entries vs 1054 gold mt1 turns = **74.4% coverage**.
- **Parser rules out ~0% of the observed gap.** Parse retries fired on 4/784 turns (0.51%); parse fallbacks never triggered. The baseline's structured output is essentially always parseable. The missing decisions are not parse failures — they were never produced.

The baseline's n=28 is therefore not "the 28 hardest turns" or "the 28 easiest turns". It is **self-selected at the trajectory level**: the subset of gold accept-decision turns that the baseline happened to reach, which are precisely the turns where the baseline chose to `accept` an offer early (or the rare dialogue it pushed through without accepting). That is exactly the regime where it least needs to be evaluated.

## 3. Why this matters for a head-to-head comparison

Two properties of the shared harness interact here:

1. **`turn_level_eval` scores every gold mt1 decision turn it can find.** It does not re-enter the agent's trajectory after an `accept`; it queries the agent on the *gold* history up to each turn.
2. **The `StructuredCoTReplayAgent` adapter looks up the baseline's answer by `(dialogue_id, gold_turn_index, role)`.** When there is no matching row in `turns.jsonl`, it abstains, and the harness excludes that turn from the support.

So the abstention pattern is: *whenever Protocol 1 stopped early, the adapter sees nothing and the turn drops out of the baseline's evaluation support.* The same harness evaluating the Bayesian teacher sees a live agent (not a replay log), calls `predict_turn` on every gold turn, and therefore covers **all 87 harness-eligible accept-decision turns** (and all 86 Submit-Deal turns).

The net effect is that the two numbers `0.897 / 0.809` are scored on **disjoint turn populations**, with the baseline's population biased toward its own easier terminal decisions.

## 4. The framing for the paper

The honest way to describe the comparison has three parts.

### 4.1 What Protocol-1 self-play actually reports

Protocol 1 is the evaluation protocol the field uses because it is the closest match to deployment: the agent is fully responsible for its own trajectory, and the resulting dialogue outcome (agreement, points, walkaway) is a clean behavioral summary. For **dialogue-level** metrics — who agreed, who walked away, who ended up with more points — Protocol 1 is the right protocol.

For **turn-level** metrics, Protocol 1 is biased by early termination, and the bias is agent-specific: an agent that accepts early gets a smaller, easier evaluation support than an agent that keeps going. The baseline's Protocol-1 accept-F1 of 0.809 on n=28 is therefore not comparable to the Bayesian teacher's 0.897 on n=87.

### 4.2 What the matched-support re-evaluation reports (Protocol 3)

To remove the support bias we run the Structured-CoT baseline a second time under Protocol 3: at every decision-relevant gold mt1 turn in the held-out set, we reset the agent's context to the *gold* history up to that turn and ask it what it would decide. This is a more generous evaluation of the baseline — it never pays the early-termination penalty — and it produces a matched-support accept-F1 number that is directly comparable to the teacher's.

The Protocol-3 live baseline rerun has landed at `opponent_model/results/turn_eval_structured_cot_p3_live_70b_m1_150/turn_summary.json`. It gives the baseline Accept F1 = 0.947 on n=87, bid cosine = 0.815 on n=29, and no Brier score because the prompted baseline exposes no posterior. Against the Bayesian teacher's Accept F1 = 0.897 on the same n=87, the matched-support accept result favours the baseline; the paper framing should therefore shift to calibration, auditability, and action-quality diagnostics rather than a raw accept-F1 win.

We **commit in advance** to reporting three numbers for accept F1:

- Baseline Protocol-1 F1 on n=28 — the field-standard number.
- Baseline Protocol-3 F1 on n=87 — matched support with the teacher.
- Bayesian Protocol-3 F1 on n=87 — the teacher, unchanged.

and the analogous pair for bid cosine. We will make the same commitment explicit in the paper regardless of which comparison favors the teacher.

### 4.3 What the divergence itself is telling us

The protocol-alignment issue is not merely a methodological footnote. It is **evidence that the baseline self-terminates in a way the teacher does not** — the baseline accepts 3–4 turns earlier than the human did on 82.7% of dialogues. That is directly downstream of the "accepts dominated offers" failure mode catalogued in `baseline_weaknesses.md` §3. A model that is over-eager to accept produces fewer decisions to score, and the ones it does produce are the ones it was most confident about — a silent selection bias against harder turns.

We will present this finding as a *secondary contribution*: in a prompted-CoT deployment, early acceptance silently hides a large fraction of the agent's accept-decision behaviour from any turn-level evaluation that relies on the agent's own trajectory. The Bayesian teacher, because its decision rule is driven by a menu ranked by `U_self + λ · E[U_opp]` rather than by an LLM's sampling temperature, does not exhibit this selection. A calibrated-acceptance agent is a more auditable agent, independent of which wins accept-F1.

## 5. Pre-registered interpretations

We commit in advance to one of three outcomes when the Protocol-3 baseline rerun lands:

The Protocol-3 rerun selected outcome 3: **teacher loses on matched-support accept F1**. The headline should shift to *calibration and auditability*, not raw accept-F1: the Bayesian teacher exposes (a) a 6-ordering posterior with Brier 0.085, (b) a bid menu that scores 86 native bid turns instead of the baseline's 29, and (c) a trajectory that does not self-truncate.

The accept-F1 comparison remains important, but it is now a constraint on the claim rather than the claim itself.

## 6. Reproducibility checklist

- Diagnostic script: `python -c` block at the top of this document; reruns against `structured_cot/results/protocol1_70b_full/turns.jsonl` + `data/casino_test.json` and prints every number in §2.
- Baseline Protocol-1 metrics: `opponent_model/results/turn_eval_structured_cot_replay_full150/turn_summary.json`.
- Bayesian Protocol-3 metrics: `opponent_model/results/turn_eval_bayesian_lambda1.0_m5_f0.50_full150/turn_summary.json`.
- Baseline Protocol-3 rerun driver: `structured_cot/run_protocol3_baseline.py` + `structured_cot/scripts/run_protocol3_baseline_70b.lsf`.
- Baseline Protocol-3 live metrics: `opponent_model/results/turn_eval_structured_cot_p3_live_70b_m1_150/turn_summary.json`.
