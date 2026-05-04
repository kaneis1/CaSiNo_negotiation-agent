# CaSiNo Bayesian-Distilled Negotiation Agent

This repository studies language-based negotiation agents on the CaSiNo dataset.
The current paper direction is:

1. Expose auditable opponent-belief posteriors during negotiation.
2. Distill a Bayesian teacher into an 8B student that emits tagged intermediate state.
3. Evaluate turn-level decisions with Protocol 3: accept/reject, bid similarity, strategy labels, and posterior Brier score.
4. Report SVO-conditioned lambda as a sensitivity/null-result analysis rather than a positive matched-SVO accept result.

The main surviving result is the Brier calibration story. The balanced student
achieves mean Brier `0.114`, below the CaSiNo six-ordering uniform reference
`5/36 ~= 0.139`. At supported turn indices 0, 2, and 13, per-turn Brier
slightly exceeds the reference. The 70B structured-CoT baseline does not
natively expose a posterior, so we elicit one by self-consistency; its elicited
Brier is `0.194`.

## Repository Layout

| Path | Purpose |
|---|---|
| `src/casino_belief/` | Active Python package organized by paper workflow: belief, policy, student, training, evaluation, baselines, diagnostics, transfer, and LLM utilities. |
| `experiments/lsf/` | Paper experiment launchers grouped by training, evaluation, ablation, DND transfer, and structured-CoT baseline. |
| `data/casino/` | Local CaSiNo splits and quality-filtered subsets. Large JSON files are ignored for GitHub. |
| `data/dnd/` | Local Deal-or-No-Deal transfer data. Large JSONL/raw files are ignored for GitHub. |
| `artifacts/` | Local generated results, tables, figures, logs, and training metadata. This is ignored; paper-facing aliases live in `paper_artifacts/`. |
| `external/` | Local copies of upstream CaSiNo and prior-baseline materials used for comparison. This is ignored. |
| `paper_artifacts/` | Tracked reviewer overlay with manifest-first provenance for every paper table, figure, claim, and shared result. |
| `tests/` | Unit tests for parsers, bid extraction, DND transfer, auditability, and data builders. |
| `roadmap.md` | Current project plan and paper-status reset. |

## Setup

Create the conda environment and install the repo in editable mode:

```bash
conda env create -f environment.yml
conda activate casino
pip install -e .
```

On an institutional LSF cluster, set the Python interpreter explicitly when
submitting batch jobs, for example:

```bash
python
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

## Core Evaluation

The main entry point for turn-level evaluation is:

```bash
python -m casino_belief.evaluation.turn_eval_run \
  --data data/casino/casino_test.json \
  --output-dir artifacts/results/protocol3/turn_eval_smoke_uniform \
  --max-dialogues 5 \
  --agent uniform \
  --annotations external/casino_original/data/casino_ann.json
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
bash experiments/lsf/evaluation/run_bayesian_protocol3.lsf

# SVO integrity smoke.
python -m casino_belief.diagnostics.svo.check_svo_integrity_smoke

# SVO match-vs-mismatch accept diagnostic.
python -m casino_belief.diagnostics.svo.diagnose_svo_accept_match_mismatch \
  --match-records artifacts/results/svo/turn_eval_bayesian_svo_lambda_rescaled_rerun_m5_f0.50_full150/turn_records.jsonl \
  --mismatch-records artifacts/results/svo/turn_eval_bayesian_svo_lambda_rescaled_mismatch_m5_f0.50_full150/turn_records.jsonl \
  --output-dir artifacts/results/svo/accept_diagnostic_rescaled
```



## Distillation And Student Evaluation

Distillation data is built from quality-filtered training subsets and Bayesian
teacher outputs:

```bash
bash experiments/lsf/training/run_distill_data.lsf
```

SFT training:

```bash
bash experiments/lsf/training/run_day8_sft_train.lsf
```

Student Protocol 3 evaluation:

```bash
bash experiments/lsf/evaluation/run_day9_student_eval.lsf
```

The student output parser lives in `src/casino_belief/student/student_parser.py`; bid canonical
extraction for free-text utterances lives in `src/casino_belief/evaluation/bid_extractor.py`.

## Current Locked Results

Key artifacts:

| Result | Artifact |
|---|---|
| Headline Brier numbers and plots | `artifacts/results/posterior_quality/` |
| Balanced student full150 eval | `artifacts/results/protocol3/distilled_student_balanced_full150/turn_summary.json` |
| Structured-CoT P3 live baseline | `artifacts/results/protocol3/structured_cot_70b_full150/turn_summary.json` |
| Bayesian teacher full150 eval | `artifacts/results/protocol3/bayesian_teacher_full150/turn_summary.json` |
| SVO subgroup tables | `artifacts/results/svo/subgroup_legacy/` and `artifacts/results/svo/subgroup_rescaled/` |
| SVO accept diagnostics | `artifacts/results/svo/accept_diagnostic_*` |
| Big Five/style null checks | `artifacts/results/personality/style_variant_strategy_distributions/` and `artifacts/results/personality/human_bigfive_strategy_matrix/` |
| Current SVO draft | `artifacts/results/svo/svo_subgroup_section3_1_draft.md` |

Locked Day 9/10 findings:

- Student-balanced mean Brier is `0.114`, below the CaSiNo six-ordering
  uniform reference `5/36 ~= 0.139`; supported turns 0, 2, and 13 are slightly
  above the reference.
- Baseline Structured-CoT has strong Accept-F1 but no native posterior
  distribution; the paper reports an elicited self-consistency posterior Brier
  of `0.194`.
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
