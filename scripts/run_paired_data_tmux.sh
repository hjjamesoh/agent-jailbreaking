#!/usr/bin/env bash
set -euo pipefail

session_name="${1:-paired-data-v1}"
gpu_ids="${2:-}"
stage="${3:-all}"
generator_model="${4:-Qwen/Qwen2.5-32B-Instruct}"
batch_size="${5:-4}"

if [[ ! "$gpu_ids" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "usage: bash scripts/run_paired_data_tmux.sh <session> <gpu-list> [stage] [model] [batch]" >&2
  exit 2
fi
if [[ ! "$stage" =~ ^(sources|collect|plan|generate|finalize|audit|all)$ ]]; then
  echo "invalid stage: $stage" >&2
  exit 2
fi
if tmux has-session -t "$session_name" 2>/dev/null; then
  echo "tmux session already exists: $session_name" >&2
  exit 1
fi

log_path="runs/${session_name}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p runs
tmux new-session -d -s "$session_name" \
  "env CUDA_VISIBLE_DEVICES='$gpu_ids' bash scripts/run_paired_data_all.sh '$log_path' '$stage' '$generator_model' '$batch_size'"
tmux set-option -t "$session_name" remain-on-exit on
tmux set-environment -t "$session_name" TASK_REFUSAL_LOG "$log_path"
tmux set-environment -t "$session_name" TASK_REFUSAL_GPU "$gpu_ids"

echo "started session: $session_name"
echo "physical GPU(s): $gpu_ids (renumbered from cuda:0 inside the process)"
echo "stage: $stage"
echo "generator: $generator_model"
echo "log: $log_path"
echo "attach: tmux attach -t $session_name"
