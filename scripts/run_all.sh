#!/usr/bin/env bash
set -euo pipefail

config_path="${1:-configs/main_agent_llama31.yaml}"
log_path="${2:-runs/main_agent_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$(dirname "$log_path")"
python -u scripts/check_gpu_env.py | tee -a "$log_path"
python -u scripts/run_main.py --config "$config_path" --stage all 2>&1 | tee -a "$log_path"
