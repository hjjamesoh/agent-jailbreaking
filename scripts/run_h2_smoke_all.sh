#!/usr/bin/env bash
set -euo pipefail

config_path="${1:-configs/h2_agent_step_llama31.yaml}"
log_path="${2:-runs/h2_smoke_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$(dirname "$log_path")"
python -u scripts/check_gpu_env.py 2>&1 | tee -a "$log_path"
python -u scripts/run_h2_smoke.py --config "$config_path" 2>&1 | tee -a "$log_path"
