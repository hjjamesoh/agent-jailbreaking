#!/usr/bin/env bash
set -euo pipefail

log_path="${1:-runs/paired-data-$(date +%Y%m%d_%H%M%S).log}"
stage="${2:-all}"
generator_model="${3:-Qwen/Qwen2.5-32B-Instruct}"
batch_size="${4:-4}"

source_root="data/raw/controlled_pairs_v1"
work_dir="data/processed/controlled_pairs_v1"
mkdir -p "$(dirname "$log_path")"

run_builder() {
  python -u scripts/build_paired_agent_dataset.py "$1" \
    --source-root "$source_root" \
    --work-dir "$work_dir" \
    --generator-model "$generator_model" \
    --batch-size "$batch_size" 2>&1 | tee -a "$log_path"
}

python -u scripts/check_gpu_env.py 2>&1 | tee -a "$log_path"

case "$stage" in
  sources)
    python -u scripts/prepare_paired_sources.py --root "$source_root" \
      2>&1 | tee -a "$log_path"
    ;;
  collect|plan|generate|finalize|audit)
    run_builder "$stage"
    ;;
  all)
    python -u scripts/prepare_paired_sources.py --root "$source_root" \
      2>&1 | tee -a "$log_path"
    run_builder collect
    run_builder plan
    run_builder generate
    run_builder finalize
    ;;
  *)
    echo "invalid stage: $stage" >&2
    exit 2
    ;;
esac
