# Roadmap: Bayesian-Distilled Negotiation Agent with Style Conditioning

**Deadlines**
- Abstract: **April 30** (10 days from today)
- Full paper: **May 7** (17 days from today)

**Today: April 20, 2026**

---

## Guiding Principles

1. **Protect the benchmark.** Get the Abdelnabi-style structured CoT baseline running end-to-end before anything novel. Supervisor-directed, non-negotiable.
2. **Every day produces a tangible artifact.** Code committed, numbers logged, or paragraphs drafted. No abstract planning days.
3. **Cut from the tail, not the head.** If time slips, drop evaluation breadth and style ablations first; never drop the benchmark or Protocol 3.
4. **Pre-commit predictions before looking at results.** Protects against unconscious p-hacking and gives you a story regardless of outcome.

---

## Phase 0 — Setup and Sanity (April 20–21, 2 days)

### Day 1 (Apr 20, today)

**Morning: Data audit and splits (2–3 hours)**
- Count exact dialogues in your CaSiNo copy. Confirm ~1030 vs 1096 — investigate any discrepancy.
- Check that participant IDs are present in the metadata. You need these for participant-level splitting.
- Write `make_splits.py`: hold out 150 dialogues, stratified by integrative potential, split by **participant not dialogue**. Save split IDs to `splits/test_ids.json`, `splits/train_ids.json`. Commit.

**Afternoon: Quality score plumbing (2–3 hours)**
- Write `quality_score.py` implementing:
  ```
  Q_w(d) = w·ŝ_self_pareto + (1-w)·sat̂_opp − λ·max(0, τ−ŝ_opp)²
  ```
- Include Pareto normalization: enumerate 64 possible splits per dialogue, compute max achievable self-score given priority orderings, normalize.
- Apply minimum-signal filter: drop walkaways, drop dialogues <6 turns.
- Output: three ranked lists (w=0.2, 0.5, 0.8), each filtered to top-30%.

**Evening: Eyeball validation (45 min — do not skip)**
- Read 5 top-Q and 5 bottom-Q transcripts from each of the three lists.
- Ask: does top-Q-cooperative actually read as integrative? Does top-Q-competitive read as assertive-but-not-steamrolling? If a reader can't distinguish the three style-selected sets, your Q formula is broken — tune λ, τ, or the Pareto floor before proceeding.
- Write a 2-paragraph note in `validation_notes.md` with observations. This note will feed your paper's methodology section.

### Day 2 (Apr 21)

**Morning: Pre-commit predictions (1 hour)**
- Create `predictions.md`:
  - "Competitive agent (w=0.8) achieves ≥2 more points than cooperative (w=0.2) on Protocol 2."
  - "Cooperative agent achieves ≥0.3 higher Likert satisfaction than competitive."
  - "Distilled student posterior Brier score at final turn is lower than Abdelnabi baseline by ≥0.05."
  - Any others you want to test.
- **Commit this before looking at any training result.** Timestamp it.

**Afternoon–evening: Extend eval harness for Protocol 3 (4–5 hours)**
- Your current `eval_harness.py` computes end-to-end metrics. Extend it with a `turn_level_eval()` function:
  - Given a held-out dialogue and an agent, iterate through turns.
  - At each turn t, give the agent history up to t-1 and the turn-t offer (if any).
  - Record: agent's accept/reject decision, bid (if rejecting), strategy label (run your classifier), posterior (if agent exposes one).
  - Compare against human ground truth.
  - Output per-turn metrics: accept F1, bid cosine, strategy macro-F1, Brier score for posterior.
- This is your **highest-priority engineering task of the week**. Protocol 3 is what your distillation claim stands or falls on.

---

## Phase 1 — Benchmark Locked Down (April 22–24, 3 days)

### Day 3 (Apr 22)

**Structured CoT baseline implementation (full day)**
- Implement the `StructuredCoTAgent` you already have blueprinted. XML-tagged reasoning stages: `<observation>`, `<opponent_inference>`, `<plan>`, `<utterance>`.
- Use Llama-3.3-70B on your HPC allocation.
- Run on 10 held-out dialogues as a smoke test. Check outputs are parseable.

### Day 4 (Apr 23)

**Benchmark full run + weakness characterization**
- Run Abdelnabi baseline against all 150 held-out dialogues via Protocol 3 (turn-level).
- Also run Protocol 2 (retrieval-opponent) for end-to-end outcomes. Skip Protocol 2 if Protocol 3 is eating time — Protocol 3 is the one that matters.
- Produce a table: points, deal rate, accept F1, strategy macro-F1. This is your abstract's headline baseline number.

**Weakness log**
- Write `baseline_weaknesses.md` cataloging failure modes from reading ~15 baseline trajectories:
  - Where does it miss opponent priorities?
  - Where does it accept dominated offers?
  - Where does it propose dominated offers?
  - Does it expose belief? (No — that's your motivation.)
- This becomes the "limitations of prompted CoT" section in your paper.

### Day 5 (Apr 24)

**Bayesian teacher implementation**
- Likelihood function: reuse or adapt Chawla et al.'s hierarchical ranker. You already have ~40% item accuracy — do not try to improve this before the abstract.
- Posterior update: 6 hypotheses over priority orderings, Dirichlet prior (optionally parameterized by Big Five regression — cuttable), turn-by-turn Bayesian update using likelihood.
- Menu generator: enumerate 64 splits, compute E_θ[U(π)] under current posterior for each, return top-5 by utility + 3 epistemic actions (probe, argue, share).
- Test: on 5 held-out dialogues, print the posterior trajectory and check it converges roughly toward the true priority ordering.

---

## Phase 2 — Abstract-Ready (April 25–30, 6 days)

### Day 6 (Apr 25)

**Bayesian agent evaluation (Protocol 3)**
- Wire the Bayesian teacher into a minimal agent: posterior → menu → argmax utility action → template utterance.
- Run Protocol 3 on held-out set.
- Compare to Abdelnabi baseline. Does it win on accept F1? On bid similarity? On Brier score (baseline has no posterior, so this is automatic)?
- **If Bayesian agent underperforms baseline on accept F1, stop and debug.** This is your sanity check that the teacher is worth distilling from. Do not proceed to distillation if the teacher is broken.

### Day 7 (Apr 26)

**Distillation data preparation**
- For top-30% dialogues under each of three Q_w settings, run the Bayesian teacher over every turn.
- Log structured training examples:
  ```
  <history>...</history>
  <posterior>p(θ₁)=..., p(θ₆)=...</posterior>
  <menu>[top-5 utility + 3 epistemic, with scores]</menu>
  <selected_intent>counter</selected_intent>   ← from human
  <selected_content>(3,1,2)</selected_content> ← from human
  <style>competitive</style>
  <utterance>...</utterance>                    ← from human
  ```
- ~265 dialogues × ~12 turns average × 3 styles ≈ ~9500 training examples. Save as JSONL.

### Day 8 (Apr 27)

**SFT run on Llama-3.1-8B**
- LoRA fine-tuning, style token as prompt conditioning.
- Training target: all tagged fields (posterior regression, intent classification, content classification, utterance LM loss).
- Budget: one overnight run. If HPC queue is bad, reduce to one style (balanced) for the abstract and expand post-abstract.
- Save checkpoint.

### Day 9 (Apr 28)

**Distilled student evaluation**
- Protocol 3 on held-out set, one pass per style token.
- Log the four metrics: accept F1, bid similarity, strategy macro-F1, Brier trajectory.
- Produce the headline figure: Brier score as a function of turn index, three lines for baseline / Bayesian teacher / distilled student. If the student's curve is between teacher and baseline, distillation worked.
- If Protocol 2 harness is ready, run the retrieval-opponent experiment and produce the style tradeoff bar chart.

### Day 10 (Apr 29)

**Abstract writing day**
- Draft in this structure (300 words):
  1. Problem: LLM negotiators are fluent but belief-opaque.
  2. Approach: distill Bayesian policy into LLM with belief exposed as tagged text; condition on style token for controllable behavior.
  3. Setup: CaSiNo, Llama-3.1-8B student, Llama-3.3-70B Abdelnabi baseline.
  4. Headline result: distilled student matches baseline on accept F1 while achieving [X] Brier score improvement and demonstrating style-responsive behavior.
  5. Contribution: first CaSiNo agent with auditable posteriors and controllable style.
- Internal review with supervisor if possible.

### Day 11 (Apr 30) — ABSTRACT SUBMISSION

- Final polish, submit.
- Start afternoon on Phase 3.

---

## Phase 3 — Full Paper (May 1–7, 7 days)

### Day 12 (May 1)

**DPO preference pair construction**
- From SFT student, sample two continuations per training turn at temp=0.8.
- Score each by: (a) posterior calibration vs. Bayesian teacher, (b) expected utility under posterior, (c) style consistency.
- Pair winner/loser. Drop ties.
- This gives you the DPO training corpus.

### Day 13 (May 2)

**DPO training run**
- Overnight on HPC. Same LoRA config, DPO objective.
- If it tanks SFT performance (it sometimes does), revert to SFT-only for the paper and cut DPO to an appendix ablation. Don't get attached.

### Day 14 (May 3)

**Full ablation ladder**
- Run Protocol 3 on: (1) Abdelnabi baseline, (2) SFT without posterior exposure, (3) SFT with posterior exposure, (4) SFT + DPO.
- This is your ablation table. Each row isolates a hypothesis.
- If any step doesn't help, report it honestly — reviewers respect clean negative results.

### Day 15 (May 4)

**Protocol 2 full run + style tradeoff curve**
- Retrieval-augmented opponent negotiation, 100 games × 3 styles = 300 games.
- Produce the signature figure: scatter plot with self-score on x-axis, predicted opponent satisfaction on y-axis, three points for the three styles with error bars.
- Check pre-registered predictions from Day 2. Commit the "predictions met / not met" note.

### Day 16 (May 5)

**Paper writing: methods + results (full day)**
- Methods section: Bayesian teacher, distillation objective, menu interface, Q_w and selection, pre-registration.
- Results section: ablation table, Brier trajectory figure, style tradeoff curve, Protocol 3 metrics table.
- Keep figures minimal — three is plenty for an 8-page paper.

### Day 17 (May 6)

**Paper writing: intro, related work, discussion**
- Intro: lead with the belief-opaqueness problem, then your layered decomposition.
- Related work: Chen multi-issue self-play (why it doesn't transfer to language), Lewis (language drift), Abdelnabi/Fu/NegotiationArena (prompted baselines), He 2018 (modular precedent), Chawla opponent modeling (you reuse their ranker).
- Discussion: style as controllable axis, Big Five as future work or as a section if Dirichlet-Big-Five regression actually made it in.
- Limitations: teacher noise from ~40% item accuracy, CaSiNo's one-shot setting, English-only.

### Day 18 (May 7) — PAPER SUBMISSION

- Morning: final figure polish, bibliography check, supplementary material.
- Afternoon: submit with buffer.

---

## Contingency: Cut-Order if Time Slips

Cut in this order (most cuttable first):

1. **Big Five Dirichlet prior regression.** Use uniform prior. Mention as future work.
2. **Style variants 2 and 3.** Train and evaluate only the balanced agent. Mention style as future work.
3. **DPO stage.** Report SFT-only as main result. DPO becomes appendix experiment.
4. **Protocol 2 end-to-end evaluation.** Rely on Protocol 3 only. The paper still stands on turn-level agreement.
5. **Epistemic actions in the menu.** Use only utility-bearing actions (accept, counter-splits). Mention probe/argue as future work.

Never cut: Abdelnabi baseline, Bayesian teacher, SFT distillation, Protocol 3 evaluation, posterior exposure as tagged text.

---

## Daily Rituals

- **Start of day:** read your pre-committed predictions. Spend 2 minutes checking you haven't drifted from them.
- **End of day:** log what worked, what didn't, what you'll do tomorrow. Three bullet points. No more.
- **Every 3 days:** read 5 model outputs by hand. Metrics can look good while behavior is broken. Human eyeballs on trajectories catch things F1 scores miss.
- **Supervisor meetings:** bring the tradeoff curve and the ablation table. These two figures are your story.

---

## What You're Shipping, Explicitly

- 1 Bayesian teacher producing posteriors + menus per turn
- 1 distilled student emitting tagged intermediate quantities
- 1 style token for controllable behavior (3 values if time permits, 1 if not)
- 1 Protocol 3 evaluation harness producing 4 metrics
- 1 Protocol 2 retrieval-opponent experiment (optional but desirable)
- 1 ablation ladder (up to 4 rows)
- 1 pre-registered prediction document
- 1 paper

Everything else is scope creep. Hold the line.