#!/usr/bin/env bash
set -euo pipefail

session_name="${1:-main-agent}"
config_path="${2:-configs/main_agent_llama31.yaml}"
log_path="runs/${session_name}_$(date +%Y%m%d_%H%M%S).log"

if tmux has-session -t "$session_name" 2>/dev/null; then
  echo "tmux session already exists: $session_name" >&2
  exit 1
fi

mkdir -p runs
tmux new-session -d -s "$session_name" \
  "bash scripts/run_all.sh '$config_path' '$log_path'"
tmux set-option -t "$session_name" remain-on-exit on
tmux set-environment -t "$session_name" TASK_REFUSAL_LOG "$log_path"

echo "started session: $session_name"
echo "log: $log_path"
echo "attach: tmux attach -t $session_name"
