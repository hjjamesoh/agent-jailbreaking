#!/usr/bin/env bash
set -euo pipefail
session="cast-stage1-v1"
tmux has-session -t "$session"
log="$(tmux show-environment -t "$session" CAST_STAGE1_LOG)"
log="${log#CAST_STAGE1_LOG=}"
gpu="$(tmux show-environment -t "$session" CAST_STAGE1_GPU)"
gpu="${gpu#CAST_STAGE1_GPU=}"
echo "session: $session"
echo "log: $log"
tmux list-panes -t "$session" -F 'dead=#{pane_dead} exit=#{pane_dead_status} command=#{pane_current_command}'
nvidia-smi -i "$gpu" --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
tail -n 15 "$log"
