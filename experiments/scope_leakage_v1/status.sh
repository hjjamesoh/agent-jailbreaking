#!/usr/bin/env bash
set -euo pipefail
stage="${1:-calibrate}"
case "$stage" in calibrate|evaluate) ;; *) echo "Expected calibrate or evaluate" >&2; exit 2;; esac
session="scope-leak-v1-${stage}"
tmux has-session -t "$session"
log="$(tmux show-environment -t "$session" SCOPE_LOG)"
log="${log#SCOPE_LOG=}"
gpu="$(tmux show-environment -t "$session" SCOPE_GPU)"
gpu="${gpu#SCOPE_GPU=}"
echo "session: $session"
echo "log: $log"
tmux list-panes -t "$session" -F 'dead=#{pane_dead} exit=#{pane_dead_status} command=#{pane_current_command}'
nvidia-smi -i "$gpu" --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
tail -n 15 "$log"
