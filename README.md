# CaSiNo Bayesian-Distilled Negotiation Agent

This repository studies language-based negotiation agents on the CaSiNo dataset.
The current paper direction is:

1. Expose auditable opponent-belief posteriors during negotiation.
2. Distill a Bayesian teacher into an 8B student that emits tagged intermediate state.
3. Evaluate turn-level decisions with Protocol 3: accept/reject, bid similarity, strategy labels, and posterior Brier score.
4. Report SVO-conditioned lambda as a sensitivity/null-result analysis rather than a positive matched-SVO accept result.

The main surviving result is the Brier calibration story. The student-balanced
run stays below the uniform posterior reference at every turn
(`max per-turn Brier = 0.146 < 1/6`), while prompted baselines expose no
posterior to score.

## Repository Layout

| Path | Purpose |
|---|---|
| `CaSiNo/` | Local copy of the CaSiNo data and annotation files. |
| `data/` | Train/test splits and quality-filtered training subsets. |
| `structured_cot/` | Abdelnabi-style structured-CoT baseline, replay/live adapters, and baseline weakness notes. |
| `opponent_model/` | Turn-level evaluation harness, agents, bid extraction, metrics, and Day 9/10 analysis scripts. |
| `sft_8b/` | Bayesian teacher, SFT/distillation data builders, student parser/model code, and LSF job scripts. |
| `results/` | Earlier benchmark results for strategy and preference prediction. |
| `tests/` | Unit tests for parsers, bid extraction, and SFT data builders. |
| `roadmap.md` | Current project plan and paper-status reset. |

## Setup

Create the conda environment and install the repo in editable mode:

```bash
conda env create -f environment.yml
conda activate casino
pip install -e .
```

On Minerva, the canonical Python used in recent runs is:

```bash
/sc/arion/work/cuiz02/conda-envs/envs/casino/bin/python
```

Run tests:

```bash
python -m pytest tests
```

## Core Evaluation

The main entry point for turn-level evaluation is:

```bash
python -m opponent_model.turn_eval_run \
  --data data/casino_test.json \
  --output-dir opponent_model/results/turn_eval_smoke_uniform \
  --max-dialogues 5 \
  --agent uniform \
  --annotations CaSiNo/data/casino_ann.json
```

Important agents:

| Agent | Description |
|---|---|
| `structured_cot_live` | Live 70B structured-CoT baseline under Protocol 3. |
| `structured_cot_replay` | Replay adapter for saved Protocol 1 structured-CoT traces. |
| `bayesian` | SFT-backed Bayesian teacher with posterior, menu, and template utterance. |
| `distilled_student` | LoRA student that emits posterior, intent, content, and utterance tags. |
| `uniform` / `hybrid` / `sft` | Baselines and intermediate opponent-model variants. |

### Bayesian Teacher

The teacher scores candidate splits with an additive menu objective:

```text
score(pi) = U_self(pi) + lambda * E[U_opp(pi | theta)]
```

This is not a convex self/other interpolation. `lambda` is an opponent-utility
exchange rate, so its numerical scale matters. The SVO-conditioned runs compare
moderate rescaled lambdas against legacy boundary lambdas and swapped-lambda
counterfactuals.

Useful scripts:

```bash
# Full Protocol 3 Bayesian teacher job wrapper.
bash sft_8b/scripts/run_bayesian_protocol3.lsf

# SVO integrity smoke.
python -m opponent_model.scripts.check_svo_integrity_smoke

# SVO match-vs-mismatch accept diagnostic.
python -m opponent_model.scripts.diagnose_svo_accept_match_mismatch \
  --match-records opponent_model/results/turn_eval_bayesian_svo_lambda_rescaled_rerun_m5_f0.50_full150/turn_records.jsonl \
  --mismatch-records opponent_model/results/turn_eval_bayesian_svo_lambda_rescaled_mismatch_m5_f0.50_full150/turn_records.jsonl \
  --output-dir opponent_model/results/day10_svo_accept_diagnostic_rescaled
```

On the cluster, check GPU availability first:

```bash
ssh li04e01 "gpuavail"
```

Use only `gpu` or `gpuexpress`. Prefer `h100nvl` when available; use `h10080g`
when `h100nvl` is unavailable. Avoid relying on `bsub -env`; recent scripts use
inline environment variables in the submitted command body.

## Distillation And Student Evaluation

Distillation data is built from quality-filtered training subsets and Bayesian
teacher outputs:

```bash
bash sft_8b/scripts/run_distill_data.lsf
```

SFT training:

```bash
bash sft_8b/scripts/run_day8_sft_train.lsf
```

Student Protocol 3 evaluation:

```bash
bash sft_8b/scripts/run_day9_student_eval.lsf
```

The student output parser lives in `sft_8b/student_parser.py`; bid canonical
extraction for free-text utterances lives in `opponent_model/bid_extractor.py`.

## Current Locked Results

Key artifacts:

| Result | Artifact |
|---|---|
| Headline Brier numbers and plots | `opponent_model/results/day9_headline_artifacts/` |
| Balanced student full150 eval | `opponent_model/results/turn_eval_student_balanced_full150/turn_summary.json` |
| Structured-CoT P3 live baseline | `opponent_model/results/turn_eval_structured_cot_p3_live_70b_m1_150/turn_summary.json` |
| Bayesian teacher full150 eval | `opponent_model/results/turn_eval_bayesian_lambda1.0_m5_f0.50_full150/turn_summary.json` |
| SVO subgroup tables | `opponent_model/results/day10_svo_subgroup_legacy/` and `opponent_model/results/day10_svo_subgroup_rescaled/` |
| SVO accept diagnostics | `opponent_model/results/day10_svo_accept_diagnostic_*` |
| Big Five/style null checks | `opponent_model/results/day10_style_variant_strategy_distributions/` and `opponent_model/results/day10_human_bigfive_strategy_matrix/` |
| Current Section 3.2 draft | `opponent_model/results/day10_svo_subgroup_section3_1_draft.md` |

Locked Day 9/10 findings:

- Student-balanced max per-turn Brier is `0.146`, below the uniform reference
  `1/6 = 0.167`.
- Baseline Structured-CoT has strong Accept-F1 but no posterior distribution,
  so Brier is undefined for that baseline.
- Rescaled SVO match-vs-mismatch changed lambda on `84/87` accept-eligible
  turns and flipped `0/84` accept predictions; Accept-F1 is `0.768` in both
  runs.
- Legacy SVO match-vs-mismatch flipped `9/84` accept predictions, but the
  Accept-F1 delta is non-significant: `0.911` matched vs. `0.904` mismatched,
  delta `0.007`, Welch p=`0.840`.
- Big Five/style is cut from the main paper: fixed-style strategy distributions
  are indistinguishable (`chi^2 p=0.998`), and the human Big Five x strategy
  target matrix is near-zero (`max |r|=0.069`).

## Paper Framing

The current main-paper outline is:

1. **Brier calibration and posterior exposure.** The Bayesian teacher/student
   expose auditable posterior trajectories; prompted baselines do not.
2. **SVO lambda sensitivity.** SVO-conditioned lambda changes allocation
   behavior, but accept decisions are insensitive to SVO-conditioned lambda
   within the tested regimes.
3. **Documented nulls.** Big Five/style conditioning and SVO match-vs-mismatch
   Accept-F1 are reported as null or limitations results, not headline wins.

Do not claim that SVO matched conditioning improves Accept-F1. Use
"accept-decision insensitivity to SVO-conditioned lambda" unless a future
diagnostic supports a stronger mechanism.

## Development Notes

- Prefer `rg` for code search.
- Do not overwrite prior result directories; use fresh run tags for reruns.
- Treat zero-record result directories as invalid.
- Keep result claims tied to exact artifact paths and visible denominators.
- Preserve Protocol 3 as the main evaluation protocol; Protocol 2 is optional
  unless explicitly needed for a later paper section.
