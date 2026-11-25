#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
gpu="${1:-3}"
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Expected physical GPU index" >&2; exit 2; }
session="cast-oracle-v2"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "Session already exists: $session. No session changed." >&2
  exit 1
fi
python_bin="${PYTHON:-$root/runs/cast_stage1_v1/env/bin/python}"
[[ -x "$python_bin" ]] || { echo "Missing stage1 Python: $python_bin" >&2; exit 1; }
mkdir -p runs/cast_oracle_v2/logs
log="$root/runs/cast_oracle_v2/logs/all_$(date +%Y%m%d_%H%M%S).log"
launcher="${log%.log}.sh"
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\n'
  printf 'cd %q\nexport PYTHONUNBUFFERED=1\n' "$root"
  printf '%q %q all --gpu %q --local-only\n' "$python_bin" experiments/cast_oracle_v2/run.py "$gpu"
  printf 'echo "CAST_ORACLE_COMPLETE"\n'
} > "$launcher"
printf -v command 'bash %q > %q 2>&1' "$launcher" "$log"
tmux new-session -d -s "$session" -c "$root"
tmux set-option -t "$session" remain-on-exit on
tmux set-environment -t "$session" CAST_ORACLE_LOG "$log"
tmux send-keys -t "$session" "$command; exit" C-m
echo "session: $session"
echo "physical GPU: $gpu"
echo "log: $log"
echo "Watch: tail -f $log"
