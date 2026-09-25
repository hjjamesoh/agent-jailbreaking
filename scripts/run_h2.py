from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

from task_refusal.modeling import LlamaHarness
from task_refusal.progress import log_event
from task_refusal.step_comparison import StepComparisonPipeline, load_step_config

STAGES = ("prepare", "extract", "analyze", "all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare standard-chat and agent-step refusal directions."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/h2_agent_step_llama31.yaml"),
    )
    parser.add_argument("--stage", choices=STAGES, default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_step_config(args.config)
    log_event(
        "h2_process_start",
        pid=os.getpid(),
        python=sys.version.split()[0],
        platform=platform.platform(),
        config=str(args.config),
        stage=args.stage,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    stages = ("prepare", "extract", "analyze") if args.stage == "all" else (args.stage,)
    pipeline = StepComparisonPipeline(config)
    if "prepare" in stages:
        pipeline.prepare()
    if "extract" in stages:
        pipeline.model_harness = LlamaHarness(config.model)
        pipeline.extract()
    if "analyze" in stages:
        pipeline.analyze()
    log_event("h2_process_complete", stage=args.stage, output=str(config.output_dir))


if __name__ == "__main__":
    main()
