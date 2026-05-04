#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/sc/arion/projects/lin_lab/complexbehavior/CaSino}"
cd "$REPO_ROOT"

ROOT="${ROOT:-artifacts/results/dnd_transfer/main}"
mkdir -p "$ROOT/lsf"

submit() {
  local label="$1"; shift
  echo "Submitting $label: $*" >&2
  local out
  out="$(bsub "$@" 2>&1)"
  echo "$out" >&2
  echo "$out" | sed -n 's/.*Job <\([0-9][0-9]*\)>.*/\1/p'
}

PREP_JOB=$(submit prepare < experiments/lsf/dnd_transfer/run_dnd_prepare.lsf)
RULE_JOB=$(submit rule -w "done($PREP_JOB)" < experiments/lsf/dnd_transfer/run_dnd_rule_eval.lsf)
ZERO_JOB=$(submit zero_eval -w "done($PREP_JOB)" -J "dnd_zero[1-4]%2" < experiments/lsf/dnd_transfer/run_dnd_eval_matrix.lsf)
TRAIN_JOB=$(submit fewshot_train -w "done($PREP_JOB)" < experiments/lsf/dnd_transfer/run_dnd_fewshot_train.lsf)
FEW_JOB=$(submit few_eval -w "done($TRAIN_JOB)" -J "dnd_few[5-7]%2" < experiments/lsf/dnd_transfer/run_dnd_eval_matrix.lsf)
AGG_JOB=$(submit aggregate -w "done($RULE_JOB) && done($ZERO_JOB) && done($FEW_JOB)" < experiments/lsf/dnd_transfer/run_dnd_aggregate.lsf)

cat > "$ROOT/lsf/submission_manifest.json" <<EOF
{
  "prepare": "$PREP_JOB",
  "rule": "$RULE_JOB",
  "zero_eval": "$ZERO_JOB",
  "fewshot_train": "$TRAIN_JOB",
  "fewshot_eval": "$FEW_JOB",
  "aggregate": "$AGG_JOB"
}
EOF

cat "$ROOT/lsf/submission_manifest.json"
