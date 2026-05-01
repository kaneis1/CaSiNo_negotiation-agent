#!/bin/bash
set -eo pipefail

REPO_ROOT="/sc/arion/projects/lin_lab/complexbehavior/CaSino"
cd "$REPO_ROOT"

ROOT="${ROOT:-opponent_model/results/ablation_neurips2026}"
PRED="$ROOT/ablation_predictions.md"
mkdir -p "$ROOT/lsf"

if [[ -e "$PRED" && "${OVERWRITE_PREDICTIONS:-0}" != "1" ]]; then
  echo "Pre-registration file already exists: $PRED" >&2
  echo "Set OVERWRITE_PREDICTIONS=1 only if you intentionally want to replace it." >&2
  exit 2
fi

cat > "$PRED" <<EOF
# Ablation Predictions

Written before job submission.

- Timestamp local: $(date '+%Y-%m-%d %H:%M:%S %Z')
- Timestamp UTC: $(date -u '+%Y-%m-%d %H:%M:%S UTC')
- Root: $ROOT
- Data: data/casino_test.json
- Perspectives: mturk_agent_1

1. A1a Brier > 0.150.
2. A1b ground-truth direct-SFT Brier in [0.100, 0.130].
3. A3 rule likelihood Brier in [0.110, 0.150], with largest losses on small-talk and indirect preference turns.
4. A4 action-only Accept-F1 in [0.88, 0.93].
5. A2d adversarial-prefix intervention sensitivity > 50%.
6. A5 uniform planner bid cosine < 0.80.
7. A6 no-offer evidence hurts Brier more than A7 no-utterance evidence.
8. A8 optimal likelihood temperature in [10, 50].

Locked implementation notes:

- Incremental Bayes evidence unit is the newest opponent natural utterance only.
- Direct-SFT primary target is ground-truth smoothed one-hot with p_true=0.90 and p_false=0.02.
- Strategy-filtered evidence runs are restricted to annotated test dialogues.
- Prefix-injection runs include a same-posterior sanity condition before adversarial interpretation.
EOF

echo "Wrote pre-registration gate: $PRED"

echo "Submitting training matrix..."
TRAIN_OUT=$(bsub < sft_8b/scripts/run_ablation_train_matrix.lsf)
echo "$TRAIN_OUT"
TRAIN_JOB=$(echo "$TRAIN_OUT" | sed -n 's/.*Job <\([0-9][0-9]*\)>.*/\1/p')
if [[ -z "$TRAIN_JOB" ]]; then
  echo "Could not parse training job id from bsub output; not submitting dependent eval matrix." >&2
  exit 3
fi

echo "Submitting eval matrix after training job $TRAIN_JOB finishes..."
bsub -w "done($TRAIN_JOB)" < sft_8b/scripts/run_ablation_eval_matrix.lsf

echo "Submitting A8/A9 sweeps..."
bsub < sft_8b/scripts/run_ablation_sweeps.lsf

echo "Interventions and aggregation should run after the full-student eval finishes:"
echo "  bsub < sft_8b/scripts/run_ablation_interventions.lsf"
echo "  bsub < sft_8b/scripts/run_ablation_aggregate.lsf"
