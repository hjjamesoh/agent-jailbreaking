#!/usr/bin/env bash
set -euo pipefail

session_name="${1:-main-agent}"
config_path="${2:-configs/main_agent_llama31.yaml}"
gpu_id="${3:-${CUDA_VISIBLE_DEVICES:-}}"
log_path="runs/${session_name}_$(date +%Y%m%d_%H%M%S).log"

if [[ -z "$gpu_id" ]]; then
  echo "usage: bash scripts/run_tmux.sh <session> <config> <physical_gpu_id>" >&2
  echo "example: bash scripts/run_tmux.sh exp1-refusal configs/main_agent_llama31.yaml 3" >&2
  exit 2
fi

if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
  echo "physical_gpu_id must be one integer GPU index, got: $gpu_id" >&2
  exit 2
fi

if tmux has-session -t "$session_name" 2>/dev/null; then
  echo "tmux session already exists: $session_name" >&2
  exit 1
fi

mkdir -p runs
tmux new-session -d -s "$session_name" \
  "env CUDA_VISIBLE_DEVICES='$gpu_id' bash scripts/run_all.sh '$config_path' '$log_path'"
tmux set-option -t "$session_name" remain-on-exit on
tmux set-environment -t "$session_name" TASK_REFUSAL_LOG "$log_path"
tmux set-environment -t "$session_name" TASK_REFUSAL_GPU "$gpu_id"

echo "started session: $session_name"
echo "physical GPU: $gpu_id (visible as cuda:0 inside the experiment)"
echo "log: $log_path"
echo "attach: tmux attach -t $session_name"
