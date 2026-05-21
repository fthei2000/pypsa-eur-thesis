#!/bin/bash
# Launch the job monitor as a background process on the login node.
# Usage: bash start_monitor.sh <job_id>
#   e.g. bash start_monitor.sh 28047225

set -euo pipefail

JOB_ID="${1:?Usage: bash start_monitor.sh <job_id>}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$REPO_DIR/logs/monitor_${JOB_ID}.log"

# Load API key from file if not already in env
if [[ -z "${ANTHROPIC_API_KEY:-}" ]] && [[ -f "$HOME/.anthropic_api_key" ]]; then
  export ANTHROPIC_API_KEY="$(cat "$HOME/.anthropic_api_key")"
fi

mkdir -p "$REPO_DIR/logs"

# Add user-local pip packages to path (for anthropic package)
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:${PYTHONPATH:-}"

echo "Starting monitor for job $JOB_ID"
echo "Log: $LOG_FILE"

nohup python3 "$REPO_DIR/monitor_job.py" "$JOB_ID" \
  > "$LOG_FILE" 2>&1 &

MONITOR_PID=$!
echo "Monitor PID: $MONITOR_PID"
echo "$MONITOR_PID" > "$REPO_DIR/logs/monitor_${JOB_ID}.pid"
echo "To follow: tail -f $LOG_FILE"
echo "To stop:   kill $MONITOR_PID"
