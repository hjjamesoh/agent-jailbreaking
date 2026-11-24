#!/usr/bin/env bash
# Stage 1 only. GPU 3 is the default. Does not touch any older session.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
gpu="${1:-3}"
if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then echo "Expected one GPU index" >&2; exit 2; fi
session="cast-stage1-v1"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "Session already exists: $session. Inspect status before resuming." >&2
  exit 1
fi
python_bin="${PYTHON:-$root/runs/cast_stage1_v1/env/bin/python}"
if [[ ! -x "$python_bin" ]]; then echo "Missing Python: $python_bin. Follow README setup." >&2; exit 1; fi
CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -c 'import torch,transformers,sklearn,rich,matplotlib; assert torch.cuda.is_available(); print("Dependencies ready")'
mkdir -p runs/cast_stage1_v1/logs
log="$root/runs/cast_stage1_v1/logs/first_$(date +%Y%m%d_%H%M%S).log"
launcher="${log%.log}.sh"
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\n'
  printf 'cd %q\n' "$root"
  printf 'export CUDA_VISIBLE_DEVICES=%q\nexport PYTHONUNBUFFERED=1\n' "$gpu"
  for stage in prepare download extract evaluate report; do
    printf 'echo %q\n' "STAGE_START: $stage"
    printf '%q %q %q --gpu %q' "$python_bin" experiments/cast_stage1_v1/run.py "$stage" "$gpu"
    if [[ "$stage" == extract || "$stage" == evaluate ]]; then printf ' --local-only'; fi
    printf '\n'
  done
  printf 'echo "STAGE1_COMPLETE: review required; no scope experiment was started"\n'
} > "$launcher"
printf -v command 'bash %q > %q 2>&1' "$launcher" "$log"
tmux new-session -d -s "$session" -c "$root"
tmux set-option -t "$session" remain-on-exit on
tmux set-environment -t "$session" CAST_STAGE1_LOG "$log"
tmux set-environment -t "$session" CAST_STAGE1_GPU "$gpu"
tmux send-keys -t "$session" "$command; exit" C-m
echo "session: $session"
echo "physical GPU: $gpu (cuda:0 inside this process)"
echo "root: $root"
echo "log: $log"
