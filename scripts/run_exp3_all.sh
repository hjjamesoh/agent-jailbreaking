#!/usr/bin/env bash
set -euo pipefail

config_path="${1:-configs/exp3_agentalign_llama31.yaml}"
log_path="${2:-runs/exp3_$(date +%Y%m%d_%H%M%S).log}"
stage="${3:-all}"

mkdir -p "$(dirname "$log_path")"
python -u scripts/check_gpu_env.py 2>&1 | tee -a "$log_path"
python -u scripts/prepare_exp3_data.py 2>&1 | tee -a "$log_path"
python -u scripts/run_exp3.py --config "$config_path" --stage "$stage" 2>&1 | tee -a "$log_path"
