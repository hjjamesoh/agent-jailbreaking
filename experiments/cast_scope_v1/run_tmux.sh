#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
gpu="${1:-3}"
mode="${2:-screen}"
alpha="${3:-0.2}"
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Expected GPU index" >&2; exit 2; }
[[ "$mode" == screen || "$mode" == run || "$mode" == all ]] || { echo "mode: screen/run/all" >&2; exit 2; }
python_bin="${PYTHON:-$root/runs/cast_stage1_v1/env/bin/python}"
[[ -x "$python_bin" ]] || { echo "Missing stage1 Python: $python_bin" >&2; exit 1; }
session="cast-scope-v1-$mode"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "Session already exists: $session; no session was modified." >&2
  exit 1
fi
mkdir -p runs/cast_scope_v1/logs
log="$root/runs/cast_scope_v1/logs/${mode}_$(date +%Y%m%d_%H%M%S).log"
launcher="${log%.log}.sh"
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\n'
  printf 'cd %q\n' "$root"
  printf 'export PYTHONUNBUFFERED=1\n'
  printf '%q %q %q --gpu %q --alpha %q --local-only\n' \
    "$python_bin" experiments/cast_scope_v1/run.py "$mode" "$gpu" "$alpha"
} > "$launcher"
printf -v command 'bash %q > %q 2>&1' "$launcher" "$log"
tmux new-session -d -s "$session" -c "$root"
tmux set-option -t "$session" remain-on-exit on
tmux set-environment -t "$session" CAST_SCOPE_LOG "$log"
tmux send-keys -t "$session" "$command; exit" C-m
echo "session: $session"
echo "physical GPU: $gpu"
echo "log: $log"
echo "Watch: tail -f $log"
