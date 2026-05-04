#!/bin/bash
set -euo pipefail

# Submit ablation jobs one at a time: submit next only after the previous
# submitted job reaches RUN. Long-running phases still keep their real
# dependencies: eval waits for training DONE, interventions wait for eval DONE,
# aggregate waits for everything it summarizes.

REPO_ROOT="${REPO_ROOT:-/sc/arion/projects/lin_lab/complexbehavior/CaSino}"
cd "$REPO_ROOT"

ROOT="${ROOT:-artifacts/results/ablation/main}"
QUEUE="${QUEUE:-gpu}"
PROJECT="${PROJECT:-acc_lin_lab}"
GPU_MODELS="${GPU_MODELS:-h100nvl}"
POLL_SECONDS="${POLL_SECONDS:-30}"
START_TIMEOUT_SECONDS="${START_TIMEOUT_SECONDS:-120}"

if [[ "$ROOT" = /* ]]; then
    ROOT_ABS="$ROOT"
else
    ROOT_ABS="$REPO_ROOT/$ROOT"
fi

mkdir -p "$ROOT_ABS/lsf"
MANIFEST="$ROOT_ABS/lsf/streak_manifest.tsv"
LOG="$ROOT_ABS/lsf/streak_submitter.log"
touch "$MANIFEST" "$LOG"

trap 'log "streak submitter exiting with status $? at line $LINENO"' EXIT

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG" >&2
}

job_status() {
    local jid="$1"
    bjobs -w "$jid" 2>/dev/null | awk 'NR==2{print $3}'
}

job_line() {
    local jid="$1"
    bjobs -w "$jid" 2>/dev/null | awk 'NR==2{print}'
}

wait_until_running() {
    local jid="$1" label="$2"
    while true; do
        local line stat
        line="$(job_line "$jid" || true)"
        stat="$(echo "$line" | awk '{print $3}')"
        log "poll $label/$jid ${line:-'(not visible)'}"
        case "$stat" in
            RUN)
                return 0
                ;;
            DONE)
                log "$label/$jid reached DONE before a RUN poll caught it; continuing."
                return 0
                ;;
            EXIT)
                log "ERROR: $label/$jid exited before running successfully."
                bjobs -l "$jid" | sed -n '1,180p' | tee -a "$LOG" >&2 || true
                return 2
                ;;
            "")
                log "$label/$jid is no longer visible in bjobs; assuming an old completed seed job."
                return 0
                ;;
        esac
        sleep "$POLL_SECONDS"
    done
}

wait_until_done() {
    local jid="$1" label="$2"
    while true; do
        local line stat
        line="$(job_line "$jid" || true)"
        stat="$(echo "$line" | awk '{print $3}')"
        log "done-poll $label/$jid ${line:-'(not visible)'}"
        case "$stat" in
            DONE)
                return 0
                ;;
            EXIT)
                log "ERROR: $label/$jid exited."
                bjobs -l "$jid" | sed -n '1,180p' | tee -a "$LOG" >&2 || true
                return 2
                ;;
            "")
                # LSF can drop old DONE jobs from normal bjobs output; treat
                # disappearance after submission as non-fatal and let artifact
                # checks catch missing outputs later.
                return 0
                ;;
        esac
        sleep "$POLL_SECONDS"
    done
}

submit_task() {
    local label="$1" task="$2" mem="$3" wall="$4" script="$5"
    local models model_index model out rc jid start now elapsed line stat
    read -r -a models <<< "$GPU_MODELS"
    if [[ "${#models[@]}" -eq 0 ]]; then
        log "ERROR: GPU_MODELS is empty."
        return 3
    fi

    for model_index in "${!models[@]}"; do
        model="${models[$model_index]}"
        log "Submitting $label task=$task on queue=$QUEUE model=$model"
        out="$(
            bsub \
                -J "$label" \
                -P "$PROJECT" \
                -q "$QUEUE" \
                -gpu "num=1" \
                -R "$model rusage[mem=$mem] span[hosts=1]" \
                -n 1 \
                -W "$wall" \
                -o "$ROOT_ABS/lsf/${label}.%J.out" \
                -e "$ROOT_ABS/lsf/${label}.%J.err" \
                "cd $REPO_ROOT && ROOT=$ROOT TASK=$task bash $script" 2>&1
        )"
        rc=$?
        log "$out"
        if [[ "$rc" -ne 0 ]]; then
            log "bsub failed for $label on $model (rc=$rc)."
            continue
        fi
        jid="$(echo "$out" | sed -n 's/.*Job <\([0-9][0-9]*\)>.*/\1/p')"
        if [[ -z "$jid" ]]; then
            log "ERROR: could not parse job id for $label"
            return 3
        fi
        echo -e "$label\t$task\t$jid\t$script\t$QUEUE\t$model" >> "$MANIFEST"

        start="$(date +%s)"
        while true; do
            line="$(job_line "$jid" || true)"
            stat="$(echo "$line" | awk '{print $3}')"
            log "poll $label/$jid ${line:-'(not visible)'}"
            case "$stat" in
                RUN)
                    LAST_JID="$jid"
                    return 0
                    ;;
                DONE)
                    log "$label/$jid reached DONE before a RUN poll caught it; continuing."
                    LAST_JID="$jid"
                    return 0
                    ;;
                EXIT)
                    log "ERROR: $label/$jid exited before running successfully."
                    bjobs -l "$jid" | sed -n '1,180p' | tee -a "$LOG" >&2 || true
                    return 2
                    ;;
            esac

            now="$(date +%s)"
            elapsed=$((now - start))
            if [[ "$model_index" -lt $((${#models[@]} - 1)) && "$elapsed" -ge "$START_TIMEOUT_SECONDS" ]]; then
                log "$label/$jid still pending after ${elapsed}s on $model; killing and trying next model."
                bkill "$jid" | tee -a "$LOG" >&2 || true
                break
            fi
            sleep "$POLL_SECONDS"
        done
    done

    log "ERROR: all GPU model attempts failed for $label"
    return 4
}

collect_jobs_by_prefix() {
    local prefix="$1"
    awk -F'\t' -v p="$prefix" '$1 ~ p {print $3}' "$MANIFEST"
}

log "Starting streak submitter. ROOT=$ROOT QUEUE=$QUEUE GPU_MODELS='$GPU_MODELS'"

train_jobs=()
if [[ -n "${SEED_TRAIN1_JOB:-}" ]]; then
    log "Using existing train task 1 job: $SEED_TRAIN1_JOB"
    wait_until_running "$SEED_TRAIN1_JOB" "ab_train_1"
    echo -e "ab_train_1\t1\t$SEED_TRAIN1_JOB\texperiments/lsf/ablation/run_ablation_train_matrix.lsf\tseed" >> "$MANIFEST"
    train_jobs+=("$SEED_TRAIN1_JOB")
else
    submit_task ab_train_1 1 40G 08:00 experiments/lsf/ablation/run_ablation_train_matrix.lsf
    train_jobs+=("$LAST_JID")
fi

for task in 2 3 4 5; do
    seed_var="SEED_TRAIN${task}_JOB"
    if [[ -n "${!seed_var:-}" ]]; then
        jid="${!seed_var}"
        log "Using existing train task $task job: $jid"
        wait_until_running "$jid" "ab_train_$task"
        echo -e "ab_train_$task\t$task\t$jid\texperiments/lsf/ablation/run_ablation_train_matrix.lsf\tseed" >> "$MANIFEST"
        train_jobs+=("$jid")
    else
        submit_task "ab_train_$task" "$task" 40G 08:00 experiments/lsf/ablation/run_ablation_train_matrix.lsf
        train_jobs+=("$LAST_JID")
    fi
done

sweep_jobs=()
for task in 1 2 3 4 5 6 7 8 9 10 11; do
    submit_task "ab_sweep_$task" "$task" 40G 04:00 experiments/lsf/ablation/run_ablation_sweeps.lsf
    sweep_jobs+=("$LAST_JID")
done

log "Waiting for all training jobs to finish before eval."
for i in "${!train_jobs[@]}"; do
    wait_until_done "${train_jobs[$i]}" "train_$((i + 1))"
done

eval_jobs=()
for task in $(seq 1 22); do
    submit_task "ab_eval_$task" "$task" 40G 06:00 experiments/lsf/ablation/run_ablation_eval_matrix.lsf
    eval_jobs+=("$LAST_JID")
done

log "Waiting for eval and sweep jobs before interventions/aggregate."
for i in "${!eval_jobs[@]}"; do
    wait_until_done "${eval_jobs[$i]}" "eval_$((i + 1))"
done
for i in "${!sweep_jobs[@]}"; do
    wait_until_done "${sweep_jobs[$i]}" "sweep_$((i + 1))"
done

intervention_jobs=()
for task in 1 2 3; do
    submit_task "ab_intervene_$task" "$task" 20G 01:00 experiments/lsf/ablation/run_ablation_interventions.lsf
    intervention_jobs+=("$LAST_JID")
done
for i in "${!intervention_jobs[@]}"; do
    wait_until_done "${intervention_jobs[$i]}" "intervention_$((i + 1))"
done

submit_task ab_aggregate 1 20G 01:00 experiments/lsf/ablation/run_ablation_aggregate.lsf
aggregate_job="$LAST_JID"
wait_until_done "$aggregate_job" "aggregate"

log "Streak submitter finished."
