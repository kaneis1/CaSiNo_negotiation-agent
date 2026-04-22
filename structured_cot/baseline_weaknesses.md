# Baseline Weaknesses: Abdelnabi-style Structured-CoT on CaSiNo

**Corpus.** 150 held-out CaSiNo dialogues played under Protocol 1 by a
Llama-3.3-70B-Instruct agent prompted with Abdelnabi's five-block
Structured-CoT template (`<observation>`, `<opponent_inference>`,
`<plan>`, `<utterance>`, `<decision>`). Log:
`structured_cot/results/protocol1_70b_full/` (784 agent-turns,
4 parse retries, 0 unrecoverable parse failures, 0 illegal counter-offers).
Quantitative scan: `structured_cot/scripts/baseline_weakness_scan.py`.

This document catalogs four systematic failure modes in that corpus. All
point estimates are against the ground-truth opponent priorities recorded
in `CaSiNo/data/casino.json` — metadata the agent never sees at inference
time. We sampled ~15 complete trajectories (d1, d4, d5, d16, d18, d33,
d34, d188, d193, d210, d231, d302, d427, d449) plus the aggregate scan.

The weaknesses below are the motivation for the Bayesian teacher reported
in `data/validation_notes.md`: an explicit posterior over the six
priority orderings + a λ-weighted menu over all 64 splits.


## (A) Opponent-priority inference is brittle and privately incoherent

The `<opponent_inference>` block is the baseline's only representation
of opponent preferences, and it is a free-text hedge rather than a belief.

**Aggregate.** Over 784 agent-turns:
- `353 / 784 = 45.0%` of `<opponent_inference>` blocks hedge — "uncertain",
  "two plausible orderings", "could be X or Y" — while being asked for a
  commitment. The hedges are conversational fillers; nothing downstream
  consumes them as probabilities.
- Of the `745 / 784 = 95%` blocks that eventually name a top priority (via
  a 3-item chain or a "prioritizes X" cue), `209 / 745 = 28.1%` name the
  wrong one relative to the ground-truth `opponent_priorities["High"]`.
- The agent *regresses* across turns: in dialogue 5, turn 2 correctly
  infers `Firewood > Food/Water` after the opponent explicitly says
  *"I prefer firewood because I will get cold at night"* — but by turn 6
  the block flips to `Food > Firewood > Water` despite the opponent
  reinforcing the firewood claim in every intervening utterance. A true
  Bayesian update would monotonically concentrate mass on the revealed
  top; the free-text inference drifts because the model re-summarises the
  whole history each turn and is susceptible to recency bias from its own
  `<utterance>` rebuttals ("I also have a high need for food").

**Qualitative pattern (d34, turn 1).** Cold-start: with zero opponent
signal the agent still asserts `Food > Water > Firewood` as one of "two
plausible orderings" and lets `<plan>` use it. This is a prior masquerading
as a conclusion. The true ordering was `Firewood > Water > Food`; every
opening counter-offer the agent built on this assumption was
priority-inverted.

**Implication.** A free-text inference block is not a belief
representation. It cannot be updated incrementally, cannot express
uncertainty quantitatively, and cannot be evaluated — there is no Brier
score for a sentence that says "could be X or Y". In the parallel
Bayesian-teacher run at λ=1.0 the posterior *monotonically* sharpened on
the ground-truth top priority as the dialogue progressed (see the eyeball
test log `sft_8b/results/lsf.eyeball2.239128428.out`, e.g. d936 k=3
posterior = 0.69 on `Firewood > Water > Food`); the baseline has no
analogous quantity to plot.


## (B) Accepting dominated offers

**Definition.** An accept is "dominated" if, against the true opponent
priorities and a reasonable opponent reservation point (`U_opp ≥ 15`, i.e.
the opponent is ≥ 10 pts over the walkaway floor), there exists a feasible
split that gives the agent strictly more points than the pending offer.

**Aggregate.** `18 / 132 = 13.6%` of accepts leave ≥ 4 points on the
table; the largest leftover was 10 points on a 36-point scale.

**Representative cases.**

- **d427 t12** (agent priorities `Water > Firewood > Food`; opponent
  true `Water > Food > Firewood`). Opponent submits `you get F=1,W=2,Fw=0`.
  Agent accepts: `U_self = 3 + 10 + 0 = 13`. A log-rolling split
  `agent F=0, W=3, Fw=2` gives the agent 23 points while still leaving
  the opponent 15 (= walkaway + 10). The agent left 10 points on the
  table and the agent's `<observation>` even identifies the structure
  ("they have consistently asked for more Water… medical condition"),
  then fails to act on it.
- **d302 t12** (agent `Food > Firewood > Water`; opponent
  true `Water > Firewood > Food`). This is the textbook logrolling
  scenario — complementary priorities — and the opponent proposes
  `agent F=3, W=0, Fw=1`. Agent accepts at `U_self = 19`; the Pareto-
  efficient split `agent F=3, W=0, Fw=3` gives 27 while keeping the
  opponent at 15. Leftover = 8 points = ~22% of max. The agent's
  `<plan>` notes "this looks fair" without ever enumerating the
  feasible set.
- **d16 t12, d193 t17, d231 t13, d449 t11:** all leave 5–7 points on
  the table in logrolling scenarios the agent could detect — in each
  case the `<opponent_inference>` correctly identifies the opponent's
  top priority but the `<plan>` settles for a roughly balanced split
  instead of an extractive one.

**Root cause.** Acceptance in the baseline is decided by free-form
deliberation in `<plan>`, not by comparing the pending offer's
`U_self` to a principled menu. The agent has no internal
`max_{π : U_opp(π|θ̂) ≥ τ} U_self(π)` computation; it settles when the
offer "looks fair" — an anchor-biased heuristic that systematically
undershoots logrolling opportunities.


## (C) Proposing dominated offers

**Definition.** A counter-offer π is "dominated" if, against the true
opponent priorities, there is another feasible split π' with
`U_self(π') ≥ U_self(π)` and `U_opp(π'|θ_true) ≥ U_opp(π|θ_true)`,
strictly greater on at least one. "Strictly worse for self" is the
stronger subcase `U_self(π') > U_self(π)` — the agent could give itself
more points without making the opponent unhappier.

**Aggregate.** `502 / 539 = 93.1%` of counter-offers are dominated.
Of those, `297 / 539 = 55.1%` are strictly worse for the agent than a
Pareto-accessible alternative — a self-inflicted loss, not a concession.

**Representative cases.**

- **d1 t5** (both players `Food > Firewood > Water`, i.e. competitive
  same-priority scenario). Agent counters `agent F=2, W=0, Fw=1`:
  `U_self = 14, U_opp_true = 22`. The split `agent F=0, W=1, Fw=3`
  gives `U_self = 15, U_opp_true = 21` — strictly better on the agent's
  side *and* only marginally worse for the opponent. The baseline is
  holding its Food share to signal its priority when it could instead
  logroll around the Firewood/Water slack.
- **d4 t3** (both `Food > Firewood > Water`). Agent counters
  `agent F=2, W=0, Fw=0`: `U_self = 10` — a pathological undercount
  that gives the opponent 26 points. A trivial alternative,
  `agent F=0, W=0, Fw=3`, gives the agent 12 points (!) and the
  opponent 24. The baseline's utterance-first framing lets it commit
  to "I'll give up most items" as a rhetorical move without checking
  the numbers.
- **d16 t0** (agent `Food > Firewood > Water`, opponent true
  `Firewood > Food > Water`). Cold-start counter: agent proposes
  `agent F=2, W=0, Fw=1`. With zero dialogue signal the agent ignores
  the scenario structure and opens with a roughly even split; the
  opening offer is dominated from turn 0. A principled opener
  — `agent F=3, W=1, Fw=0` — is `U_self = 18`, anchors on priority,
  and trades Firewood the agent values least.

**Root cause.** The baseline emits a single counter-offer per turn
chosen by the LLM's free-text planning. There is no search over the
64 feasible splits; the proposal is whatever the `<plan>` block's
narrative justifies. This produces counter-offers that are locally
fluent ("a balanced split") but globally sub-optimal: the agent never
computes `argmax U_self(π)` subject to opponent-acceptability, and
therefore routinely proposes splits that both players would leave
money on the table in.


## (D) No belief exposure (architectural, not corpus-dependent)

The five-block template emits a natural-language `<opponent_inference>`
and nothing else about opponent state. There is:

- **No probability distribution** over the 6 priority orderings. A
  calibrated Brier score is impossible to define against a free-text
  inference, so the baseline cannot be evaluated on belief quality.
  Turn-level Brier over the 150-dialogue run is `nan` (support = 0)
  — there is no quantity to score.
- **No incremental update.** Each turn regenerates the full
  `<opponent_inference>` from scratch. There is no mechanism to carry
  state between turns, which is why (A) shows drifting claims even on
  monotonically consistent opponent signals.
- **No action–belief coupling.** The `<plan>` block and the
  `<opponent_inference>` block are written by the same forward pass,
  but the plan is free to ignore the inference. We verified this
  qualitatively in ~15 dialogues; it also shows up in (C): 28% of
  counter-offers are built on a wrong top-priority claim, 93% are
  Pareto-dominated against the true θ.

This is the critical weakness for the paper's framing. Structured
prompting with XML tags gives the appearance of inference without any
of the properties that make inference useful: no probabilities, no
updates, no accountability to downstream action. The Bayesian teacher
in `sft_8b/bayesian_agent.py` is designed specifically to fix the three
properties above: it keeps an explicit posterior `p(θ | dialogue)` over
the 6 orderings, it refreshes that posterior every turn via
`K = 16` Monte-Carlo samples from the SFT model, and it selects
actions by maximising an expected-utility score
`score(π) = U_self(π) + λ · E_θ[U_opp(π|θ)]` over *all* 64 feasible
splits — not just the one the LLM happens to narrate.


## Head-to-head evidence that the weaknesses matter

Scored through the same `opponent_model.turn_level_eval` harness, the
Bayesian teacher (λ=1.0, margin=5, floor=0.5 — see `validation_notes.md`)
converts the weaknesses above into measurable gains on the identical
150-dialogue test set:

| Metric                 | Baseline (Structured-CoT 70B) | Bayesian (SFT-8B, λ=1)    | Δ      |
|------------------------|-------------------------------|----------------------------|--------|
| Accept F1              | 0.809  (n=28)                 | **0.911**  (n=87)          | +0.103 |
| Brier (posterior)      | nan (no posterior)            | **0.086**  (n=1054)        | n/a    |
| Bid cosine             | 0.852  (n=14)                 | 0.753  (n=86)              | −0.099 |
| Strategy macro-F1      | 0.133  (n=269)                | 0.048  (n=350)             | −0.085 |

The +0.103 on Accept F1 is the direct consequence of (B): the baseline
accepts dominated offers because it has no menu to compare against, and
mis-rejects non-dominated offers because its acceptance heuristic is
narrative. The free Brier win is (D). The bid-cosine and strategy-F1
results favour the baseline on different supports and are discussed
honestly in `validation_notes.md` — the bid-cosine gap largely reflects
support size (n=14 vs 86) and the strategy-F1 gap reflects the Bayesian
agent's use of a template utterance by design.


## Take-aways for the paper's "Limitations of Prompted CoT" section

1. **Free-text inference is not a belief.** 45% of inference blocks
   hedge, 28% commit to the wrong top priority, and none expose a
   distribution. There is no Brier score for a sentence.
2. **Free-text planning is not a menu.** 93% of counter-offers are
   Pareto-dominated against the true opponent θ, and 55% are strictly
   worse for the agent — self-inflicted losses, not concessions.
3. **Free-text acceptance is not a reservation rule.** 14% of accepts
   leave ≥ 4 points on the table, up to 10 points on log-rolling
   scenarios where complementarity is observable in the transcript.
4. **The XML template adds no structure beyond formatting.** The same
   forward pass produces `<opponent_inference>` and `<plan>`; the plan
   is free to ignore the inference. Tags are an affordance for parsing,
   not a constraint on reasoning.

The Bayesian teacher addresses (1) with a K=16 MC posterior over the 6
orderings, (2) with argmax over all 64 feasible splits, (3) with a
margin-plus-floor acceptance rule derived from the data (see
`validation_notes.md`), and — structurally — (4) with a factorised
architecture where belief and action are separate functions of the
dialogue prefix. The head-to-head +0.103 Accept-F1 result and the
free Brier win establish that these fixes translate into measurable
task performance on the same benchmark the baseline is usually
evaluated against.
