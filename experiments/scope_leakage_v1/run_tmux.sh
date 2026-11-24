#!/usr/bin/env bash
# Dedicated session/output. Never stop or rename another experiment's session.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
stage="${1:-calibrate}"
gpu="${2:-0}"
if [[ "$stage" != calibrate && "$stage" != evaluate ]]; then
  echo "Usage: PYTHON=/path/to/python bash experiments/scope_leakage_v1/run_tmux.sh calibrate|evaluate GPU" >&2
  exit 2
fi
if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
  echo "Specify exactly one physical GPU index (for example 0)." >&2
  exit 2
fi
session="scope-leak-v1-${stage}"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "Session already exists: $session. Inspect it; do not launch a duplicate." >&2
  exit 1
fi
python_bin="${PYTHON:-python}"
python_bin="$("$python_bin" -c 'import sys; print(sys.executable)')"
"$python_bin" -c 'import torch, transformers; assert torch.cuda.is_available(); print("Python/GPU dependencies OK")'
mkdir -p runs/scope_leakage_v1/logs
log="$root/runs/scope_leakage_v1/logs/${stage}_$(date +%Y%m%d_%H%M%S).log"
launcher="${log%.log}.sh"
# All interpolated shell arguments use printf %q. No eval or user code is accepted.
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\n'
  printf 'cd %q\n' "$root"
  printf 'export CUDA_VISIBLE_DEVICES=%q\n' "$gpu"
  printf 'export PYTHONUNBUFFERED=1\n'
  printf '%q %q prepare\n' "$python_bin" experiments/scope_leakage_v1/run.py
  if [[ "$stage" == calibrate ]]; then
    printf '%q %q extract --local-only\n' "$python_bin" experiments/scope_leakage_v1/run.py
    printf '%q %q calibrate --local-only\n' "$python_bin" experiments/scope_leakage_v1/run.py
  else
    printf '%q %q evaluate --local-only\n' "$python_bin" experiments/scope_leakage_v1/run.py
  fi
  printf 'echo "STAGE_COMPLETE: %s"\n' "$stage"
} > "$launcher"
printf -v command 'bash %q > %q 2>&1' "$launcher" "$log"
# Start paused via a small launcher, then keep the pane for exit-status inspection.
tmux new-session -d -s "$session" -c "$root"
tmux set-option -t "$session" remain-on-exit on
tmux set-environment -t "$session" SCOPE_LOG "$log"
tmux set-environment -t "$session" SCOPE_GPU "$gpu"
tmux send-keys -t "$session" "$command; exit" C-m
echo "session: $session"
echo "log: $log"
echo "root: $root"
echo "physical GPU: $gpu"
echo "Inspect: bash experiments/scope_leakage_v1/status.sh $stage"
