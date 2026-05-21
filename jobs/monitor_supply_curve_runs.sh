#!/usr/bin/env bash

set -euo pipefail

user_name="${1:-s240026}"
interval_seconds="${2:-300}"
log_file="${3:-/work3/s240459/pypsa-eur-thesis/logs/supply_curve_monitor.log}"
run_log_dir="/work3/s240459/pypsa-eur-thesis/logs/supply_curve"

mkdir -p "$(dirname "$log_file")"

while true; do
  ts="$(date '+%F %T %Z')"
  {
    echo "=== $ts ==="
    bjobs -w -u "$user_name" | awk 'NR == 1 || /sc-S/'
  } >> "$log_file"

  jobs="$(bjobs -w -u "$user_name" | awk 'NR > 1 && /sc-S/ {print $1}')"
  if [[ -z "$jobs" ]]; then
    echo "ALERT no active sc-S jobs remaining" >> "$log_file"
    exit 0
  fi

  for job_id in $jobs; do
    out_file="$(find "$run_log_dir" -maxdepth 1 -type f -name "*_${job_id}.out" -print -quit)"
    err_file="$(find "$run_log_dir" -maxdepth 1 -type f -name "*_${job_id}.err" -print -quit)"

    if [[ -n "$out_file" && -f "$out_file" ]]; then
      last_lines="$(tail -n 3 "$out_file" 2>/dev/null | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g')"
      echo "JOB $job_id OUT $out_file LAST $last_lines" >> "$log_file"
    fi

    if [[ -n "$out_file" || -n "$err_file" ]]; then
      if rg -n "Snakemake failed|Exiting because a job execution failed|WorkflowError|RuleException|Out Of Memory|Killed|TERM_|No space" ${out_file:+"$out_file"} ${err_file:+"$err_file"} >/dev/null 2>&1; then
        echo "ALERT failure markers detected for job $job_id" >> "$log_file"
      fi
    fi
  done

  sleep "$interval_seconds"
done
