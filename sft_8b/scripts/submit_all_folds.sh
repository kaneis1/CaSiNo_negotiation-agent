#!/bin/bash
# Submit all K folds of the CV pipeline as independent LSF jobs.
#
# We DON'T use `bsub -env` to pass FOLD/N_SPLITS/SEED — on some LSF
# installations (including Minerva) `-env` silently fails to propagate
# user-defined variables, leaving the job's `${FOLD:-}` empty and tripping
# the safety guard in run_cv_fold.lsf. Instead, we generate a tiny per-fold
# wrapper LSF script via heredoc that:
#   1. Carries its own #BSUB directives (resources, walltime, log paths).
#   2. `export`s FOLD/N_SPLITS/SEED inside the job's shell.
#   3. `exec`s `bash sft_8b/scripts/run_cv_fold.lsf` (the inner pipeline —
#      its own #BSUB lines are just comments to bash and are ignored).
#
# Each job runs:  build data -> train -> eval -> save summary.json
# (see sft_8b/scripts/run_cv_fold.lsf for the full pipeline).
#
# Usage
# -----
#   bash sft_8b/scripts/submit_all_folds.sh                # K=5, seed=42
#   N_SPLITS=10 bash sft_8b/scripts/submit_all_folds.sh    # K=10
#   SEED=1234   bash sft_8b/scripts/submit_all_folds.sh    # different shuffle
#
# After all K jobs finish:
#   python -m sft_8b.cv_aggregate
#
# The script prints the K job ids it submits so you can monitor with:
#   bjobs                # all your jobs
#   bjobs <jobid>        # one job
#   tail -f sft_8b/results/cv/fold_<K>/fold.log

set -euo pipefail

REPO_ROOT="/sc/arion/projects/lin_lab/complexbehavior/CaSino"
cd "$REPO_ROOT"

N_SPLITS=${N_SPLITS:-5}
SEED=${SEED:-42}

mkdir -p sft_8b/results/cv

echo "=== Submitting ${N_SPLITS} CV folds (seed=${SEED}) ==="
JOBIDS=()
for ((k=0; k<N_SPLITS; k++)); do
    # Generate a per-fold wrapper inline. The #BSUB directives MUST be at
    # the very top of the heredoc body for bsub to parse them — that's why
    # we re-declare them here instead of cat-ing run_cv_fold.lsf directly.
    OUT=$(bsub <<EOF
#!/bin/bash
#BSUB -J sft8b_cv_fold_${k}
#BSUB -P acc_lin_lab
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -R "h100nvl rusage[mem=40000] span[hosts=1]"
#BSUB -n 4
#BSUB -W 12:00
#BSUB -o sft_8b/results/cv/lsf.fold_${k}.%J.out
#BSUB -e sft_8b/results/cv/lsf.fold_${k}.%J.err
#BSUB -L /bin/bash

export FOLD=${k}
export N_SPLITS=${N_SPLITS}
export SEED=${SEED}

exec bash sft_8b/scripts/run_cv_fold.lsf
EOF
)
    JID=$(echo "$OUT" | awk '{print $2}' | tr -d '<>')
    JOBIDS+=("$JID")
    printf "  fold %d -> jobid %s  (sft8b_cv_fold_%d)\n" "$k" "$JID" "$k"
done

echo ""
echo "All ${N_SPLITS} folds submitted: ${JOBIDS[*]}"
echo ""
echo "Monitor with:"
echo "  bjobs"
echo "  for k in \$(seq 0 $((N_SPLITS-1))); do echo \"--- fold \$k ---\"; tail -3 sft_8b/results/cv/fold_\$k/fold.log 2>/dev/null; done"
echo ""
echo "When all jobs finish, aggregate with:"
echo "  python -m sft_8b.cv_aggregate"
