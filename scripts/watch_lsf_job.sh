#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <job_id> <submit_script> [poll_seconds]"
  echo "Example: $0 28029947 run_L5_myopic.sh 300"
  exit 2
fi

JOB_ID="$1"
SUBMIT_SCRIPT="$2"
POLL_SECONDS="${3:-300}"
MAX_RELAUNCHES="${MAX_RELAUNCHES:-1}"

RUN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$RUN_ROOT/logs/job-watch"
WATCH_LOG="$STATE_DIR/job_${JOB_ID}_watch.log"
mkdir -p "$STATE_DIR"

# Optional: set NOTIFY_EMAIL in the environment for mail notifications.
NOTIFY_EMAIL="${NOTIFY_EMAIL:-}"

log() {
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$ts] $*" | tee -a "$WATCH_LOG"
}

send_notice() {
  local subject="$1"
  local body="$2"
  log "$subject"

  if [[ -z "$NOTIFY_EMAIL" ]]; then
    return 0
  fi

  if command -v mail >/dev/null 2>&1; then
    if printf "%s\n" "$body" | mail -s "$subject" "$NOTIFY_EMAIL"; then
      return 0
    fi
    log "Local mail command failed; trying LSF notification fallback."
  fi

  if command -v bsub >/dev/null 2>&1; then
    local notify_job="notify-${JOB_ID}-$(date +%s)"
    local escaped_subject escaped_body
    escaped_subject="${subject//\"/\\\"}"
    escaped_body="${body//\"/\\\"}"
    (
      cd "$RUN_ROOT"
      bsub \
        -q hpc \
        -W 00:10 \
        -n 1 \
        -R "rusage[mem=1024]" \
        -u "$NOTIFY_EMAIL" \
        -N \
        -J "$notify_job" \
        -o "logs/job-watch/${notify_job}_%J.out" \
        -e "logs/job-watch/${notify_job}_%J.err" \
        "echo \"$escaped_subject\"; echo; echo \"$escaped_body\"" \
        >/dev/null
    ) || log "LSF notification fallback submission failed."
  fi
}

job_state() {
  bjobs -noheader -o "stat" "$1" 2>/dev/null | awk 'NR==1{print $1}'
}

job_file_from_bjobs_l() {
  local id="$1"
  local kind="$2"
  if [[ "$kind" == "out" ]]; then
    bjobs -l "$id" 2>/dev/null | sed -n 's/.*Output File <\([^>]*\)>.*/\1/p' | head -n1
  else
    bjobs -l "$id" 2>/dev/null | sed -n 's/.*Error File <\([^>]*\)>.*/\1/p' | head -n1
  fi
}

infer_log_paths() {
  local id="$1"
  local out_file err_file
  out_file="$(job_file_from_bjobs_l "$id" out || true)"
  err_file="$(job_file_from_bjobs_l "$id" err || true)"

  if [[ -z "$out_file" ]]; then
    out_file="$RUN_ROOT/logs/${id}.out"
  fi
  if [[ -z "$err_file" ]]; then
    err_file="$RUN_ROOT/logs/${id}.err"
  fi

  echo "$out_file|$err_file"
}

looks_successful() {
  local out_file="$1"
  local err_file="$2"

  if [[ -f "$out_file" ]] && grep -Eiq "completed successfully|run completed successfully" "$out_file"; then
    return 0
  fi

  # Some jobs report final success in stderr streams.
  if [[ -f "$err_file" ]] && grep -Eiq "completed successfully|run completed successfully" "$err_file"; then
    return 0
  fi

  return 1
}

handle_failure_and_maybe_patch() {
  local id="$1"
  local out_file="$2"
  local err_file="$3"

  local combined
  combined="$(mktemp)"
  {
    [[ -f "$out_file" ]] && tail -n 300 "$out_file"
    [[ -f "$err_file" ]] && tail -n 300 "$err_file"
  } > "$combined" || true

  log "Detected failed/unknown terminal state for job $id. Inspecting logs..."

  if grep -q "Directory cannot be locked" "$combined"; then
    log "Found stale Snakemake lock signature. Applying unlock patch workflow before relaunch."
    if [[ -x "$HOME/.pixi/bin/pixi" ]]; then
      (
        cd "$RUN_ROOT"
        "$HOME/.pixi/bin/pixi" run snakemake --unlock --configfile config/Myruns/config.L5-myopic.yaml
      ) || true
    fi
  fi

  rm -f "$combined"
}

relaunch_job() {
  local submit_script="$1"
  local out

  if [[ ! -f "$RUN_ROOT/$submit_script" ]]; then
    log "Submit script not found: $RUN_ROOT/$submit_script"
    return 1
  fi

  out="$(cd "$RUN_ROOT" && bsub < "$submit_script")"
  log "Relaunch output: $out"
  echo "$out" | sed -n 's/.*Job <\([0-9]\+\)>.*/\1/p' | head -n1
}

main() {
  local relaunch_count=0
  local current_job_id="$JOB_ID"

  log "Starting watcher for job $current_job_id using submit script '$SUBMIT_SCRIPT' (poll=${POLL_SECONDS}s)."

  while true; do
    local state
    state="$(job_state "$current_job_id" || true)"

    if [[ "$state" == "RUN" || "$state" == "PEND" || "$state" == "PSUSP" || "$state" == "USUSP" || "$state" == "SSUSP" ]]; then
      log "Job $current_job_id state=$state; next check in ${POLL_SECONDS}s."
      sleep "$POLL_SECONDS"
      continue
    fi

    local paths out_file err_file
    paths="$(infer_log_paths "$current_job_id")"
    out_file="${paths%%|*}"
    err_file="${paths##*|}"

    if looks_successful "$out_file" "$err_file"; then
      send_notice \
        "LSF job $current_job_id completed" \
        "Job $current_job_id appears complete. Out log: $out_file ; Err log: $err_file"
      break
    fi

    handle_failure_and_maybe_patch "$current_job_id" "$out_file" "$err_file"

    if (( relaunch_count >= MAX_RELAUNCHES )); then
      send_notice \
        "LSF job $current_job_id failed and watcher stopped" \
        "Job $current_job_id did not finish successfully and max relaunches ($MAX_RELAUNCHES) was reached. Check logs: $out_file and $err_file"
      break
    fi

    local new_job_id
    new_job_id="$(relaunch_job "$SUBMIT_SCRIPT")"
    if [[ -z "$new_job_id" ]]; then
      send_notice \
        "LSF relaunch failed" \
        "Unable to parse a new job ID from relaunch command for submit script '$SUBMIT_SCRIPT'."
      break
    fi

    relaunch_count=$((relaunch_count + 1))
    log "Relaunch successful. Old job=$current_job_id, new job=$new_job_id, relaunch_count=$relaunch_count"
    current_job_id="$new_job_id"
    sleep "$POLL_SECONDS"
  done
}

main "$@"