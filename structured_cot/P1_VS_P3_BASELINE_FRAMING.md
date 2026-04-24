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
(Protocol 3 style), so its Accept F1 is on the full gold support (see table
below). Comparing it to **0.81 @ n = 28** mixes two different experimental
protocols. Reviewers are right to call that out.

## What we report instead (Option 3)

**Three baseline-related numbers** (teacher = row 3):

| # | Condition | Accept F1 | n (accept) | Bid cosine | n (bid) | Notes |
|---|-----------|-----------|------------|------------|---------|--------|
| 1 | **Baseline · Protocol 1 replay** | **0.809** | **28** | **0.852** | **14** | `turn_eval_structured_cot_replay_full150` |
| 2 | **Baseline · Protocol 3 live (70B)** | **0.947** | **87** | **0.815** | **29** | `turn_eval_structured_cot_p3_live_70b_m1_150` |
| 3 | **Bayesian teacher · Protocol 3** | **0.897** | **87** | **0.744** | **86** | `turn_eval_bayesian_lambda1.0_m5_f0.50_full150` |

Row 1 replay numbers are from `opponent_model/results/turn_eval_structured_cot_replay_full150/turn_summary.json` (Accept F1 = 0.8085 printed as 0.809 in logs). Row 3 is from `opponent_model/results/turn_eval_bayesian_lambda1.0_m5_f0.50_full150/turn_summary.json`.

The **headline claim** for the paper is **row 2 vs row 3** on the **same support**
and harness. Row 2 now exists (see §Job 239149741 below).

**Apples-to-apples slice (replay vs teacher):** On the **28-turn** intersection
of Protocol-1 replay and full gold scoring, the teacher still leads by about
**+0.11** Accept F1 (teacher ≈ **0.92** vs replay ≈ **0.81** on shared keys —
recomputed from `turn_records.jsonl` in an earlier analysis). That shows the
win is not *only* an artifact of support size, but it does **not** replace the
need for row 2.

### Job 239149741 (P3 live 70B) — status

LSF job **239149741** completed successfully. Wall time **≈ 6.5 h**
(`elapsed_seconds` = 23 293). Output written 2026-04-24 06:37 to
`opponent_model/results/turn_eval_structured_cot_p3_live_70b_m1_150/`.

Row 2 is now filled from `turn_summary.json` (see table above).

### Bulletproof matched-support paragraph (template)

> We evaluate the Abdelnabi Structured-CoT baseline under two protocols. **Protocol 1 replay** (self-play trace scored on gold chat logs) yields Accept F1 **0.809** on **n = 28** accept-decision turns — only **32%** coverage of the **n = 87** gold accept turns — because the baseline often accepts early and never reaches later human decisions. **Protocol 3 live** runs the same 70B prompt on the **full gold prefix** at every `mturk_agent_1` turn, giving **Accept F1 = 0.947, n = 87** and **bid cosine = 0.815, n = 29**, directly comparable to our Bayesian teacher (**Accept F1 = 0.897, n = 87**; **bid cosine = 0.744, n = 86**). On accept-or-reject decisions the 70B prompted baseline edges the 8B Bayesian teacher by **+0.050** Accept F1; however, its bid parser yields scorable offers on only **29 / 86** turns where the teacher provides a bid, leaving bid-quality comparison under-powered. On the strict intersection of scored bid turns (via `compare_turn_eval_runs.py`) the teacher's richer bid coverage and lower Brier score may still justify its use as the primary opponent model.

## One-command comparison (in-repo)

From the repo root, after both runs have produced `turn_summary.json`:

```bash
python -m opponent_model.scripts.compare_turn_eval_runs \
  opponent_model/results/turn_eval_structured_cot_p3_live_70b_m1_150/turn_summary.json \
  opponent_model/results/turn_eval_bayesian_lambda1.0_m5_f0.50_full150/turn_summary.json \
  --label-a "P3 live 70B baseline" \
  --label-b "Bayesian λ=1 (8B teacher)"
```

Add **`--records-a`** / **`--records-b`** paths to the two `turn_records.jsonl`
files to print **matched-support** Accept F1 and bid cosine on the intersection
of turns where both agents have a scored prediction (same eligibility rule as
the harness).

## Artifacts

| Run | Output directory |
|-----|-------------------|
| P1 full (self-play) | `structured_cot/results/protocol1_70b_full/` |
| P1 replay through harness | `opponent_model/results/turn_eval_structured_cot_replay_full150/` |
| P3 live 70B baseline | `opponent_model/results/turn_eval_structured_cot_p3_live_70b_m1_150/` |
| Bayesian P3 | `opponent_model/results/turn_eval_bayesian_lambda1.0_m5_f0.50_full150/` |
| LSF transcript (239149741) | `structured_cot/results/lsf/p3_baseline_70b.239149741.out` / `.err` |

**LSF:** `bsub < structured_cot/scripts/run_p3_baseline_70b.lsf`

**Code:** `--agent structured_cot_live` in `opponent_model.turn_eval_run` uses
`structured_cot.live_turn_agent.StructuredCoTLiveTurnAgent`. The harness stamps
`dialogue_id` on every history entry so agents see a stable dialogue boundary
even when the first perspective turn has empty prefix history.

## Narrative for “limitations of prompted CoT”

The coverage gap itself is a result: **open-loop prompted negotiation agents
under-report accuracy if you only score turns they happened to reach.** Protocol
3 live evaluation is the fix for any paper that compares to human gold at
per-turn granularity.
