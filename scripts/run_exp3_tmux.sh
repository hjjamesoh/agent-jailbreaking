#!/usr/bin/env bash
set -euo pipefail

session_name="${1:-exp3-agent-refusal}"
config_path="${2:-configs/exp3_agentalign_llama31.yaml}"
gpu_id="${3:-}"
if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
  echo "usage: bash scripts/run_exp3_tmux.sh <session> <config> <physical_gpu_id>" >&2
  exit 2
fi
if tmux has-session -t "$session_name" 2>/dev/null; then
  echo "tmux session already exists: $session_name" >&2
  exit 1
fi
log_path="runs/${session_name}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p runs
tmux new-session -d -s "$session_name" \
  "env CUDA_VISIBLE_DEVICES='$gpu_id' bash scripts/run_exp3_all.sh '$config_path' '$log_path'"
tmux set-option -t "$session_name" remain-on-exit on
tmux set-environment -t "$session_name" TASK_REFUSAL_LOG "$log_path"
tmux set-environment -t "$session_name" TASK_REFUSAL_GPU "$gpu_id"
echo "started session: $session_name"
echo "physical GPU: $gpu_id (visible as cuda:0 inside the experiment)"
echo "log: $log_path"
echo "attach: tmux attach -t $session_name"
