#!/usr/bin/env bash
set -euo pipefail

session_name="${1:-main-agent}"
if ! tmux has-session -t "$session_name" 2>/dev/null; then
  echo "tmux session is not running: $session_name" >&2
  exit 1
fi

log_path="$(tmux show-environment -t "$session_name" TASK_REFUSAL_LOG | sed 's/^TASK_REFUSAL_LOG=//')"
gpu_line="$(tmux show-environment -t "$session_name" TASK_REFUSAL_GPU 2>/dev/null || true)"
gpu_id="${gpu_line#TASK_REFUSAL_GPU=}"
echo "session: $session_name"
echo "log: $log_path"
if [[ -n "$gpu_id" ]]; then
  echo "physical GPU: $gpu_id (cuda:0 inside the experiment)"
  nvidia-smi -i "$gpu_id" \
    --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader
else
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader
fi
tail -n 30 "$log_path"
