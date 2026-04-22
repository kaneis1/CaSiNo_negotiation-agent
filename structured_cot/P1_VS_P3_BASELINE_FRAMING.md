# Protocol 1 vs Protocol 3 baseline evaluation (paper framing)

## What went wrong with the first head-to-head

The first Abdelnabi-style baseline number (**Accept F1 ≈ 0.81, n = 28**) came from
`structured_cot_replay`: replaying the **Protocol 1** self-play log through
`opponent_model.turn_level_eval`. Protocol 1 stops the simulation as soon as the
agent emits `action=accept` (132 / 150 dialogues in the 70B full run). The human
gold trace often continues for several more turns before the recorded
`Accept-Deal`. So the replay adapter has **no baseline prediction** on most
gold accept-decision turns — not because the parser failed (retries ≈ 0.5%,
zero hard fallbacks), but because **those turns were never executed** under P1.

Quantitatively, on the 150-dialogue held-out split restricted to
`mturk_agent_1`:

| Gold turns (harness definition) | Count |
|---------------------------------|------:|
| Accept / Reject / Walk-Away with opponent’s offer pending | **87** |
| Protocol-1 replay rows that align with those gold indices | **28** |
| **Coverage gap** | **59 / 87 (68%)** |

The Bayesian teacher was always evaluated with **gold history** at every turn
(Protocol 3 style), so its Accept F1 (**≈ 0.91, n = 87**) is on the full gold
support. Comparing it to **0.81 @ n = 28** mixes two different experimental
protocols. Reviewers are right to call that out.

## What we report instead (Option 3)

**Three baseline-related numbers** (teacher number unchanged):

1. **Baseline · Protocol 1 replay** — field-style self-play + replay scoring:  
   Accept F1 **≈ 0.81**, **n = 28** (subset of turns the P1 trajectory touched).

2. **Baseline · Protocol 3 live** — same Structured CoT prompt + 70B model, but
   one LLM call per gold `chat_logs` turn for `mturk_agent_1`, with history
   built like `run_protocol1` (opponent utterances + deal actions rendered the
   same way). This yields **matched n** with the Bayesian run:
   - Accept F1 = **TBD** (**n ≈ 87**)
   - Bid cosine = **TBD** (**n ≈ 86**)

3. **Bayesian teacher · Protocol 3** (already run):  
   Accept F1 **≈ 0.911**, **n = 87**; Brier **≈ 0.086**; bid cosine **≈ 0.753**.

The **headline claim** is the comparison **(2) vs (3)** on the **same support**
and harness. If (2) rises (as expected when the baseline sees every turn),
that is the correct, fairer stress test — not a bug.

**Apples-to-apples slice:** On the 28-turn intersection of (1) and (3), the
teacher still leads by **≈ +0.11** F1 (teacher ≈ 0.92 vs replay ≈ 0.81), so the
win is not an artifact of support size alone.

## Artifacts

| Run | Output directory |
|-----|-------------------|
| P1 full (self-play) | `structured_cot/results/protocol1_70b_full/` |
| P1 replay through harness | `opponent_model/results/turn_eval_structured_cot_replay_full150/` |
| P3 live 70B baseline | `opponent_model/results/turn_eval_structured_cot_p3_live_70b_m1_150/` (after LSF) |
| Bayesian P3 | `opponent_model/results/turn_eval_bayesian_lambda1.0_m5_f0.50_full150/` |

**LSF:** `bsub < structured_cot/scripts/run_p3_baseline_70b.lsf`

**Code:** `--agent structured_cot_live` in `opponent_model.turn_eval_run` uses
`structured_cot.live_turn_agent.StructuredCoTLiveTurnAgent`. The harness now
stamps `dialogue_id` on every history entry so agents see a stable dialogue
boundary even when the first perspective turn has empty prefix history.

## Narrative for “limitations of prompted CoT”

The coverage gap itself is a result: **open-loop prompted negotiation agents
under-report accuracy if you only score turns they happened to reach.** Protocol
3 live evaluation is the fix for any paper that compares to human gold at
per-turn granularity.
