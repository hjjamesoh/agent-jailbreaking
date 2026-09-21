from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

from task_refusal.config import load_config
from task_refusal.data import prepare_manifests
from task_refusal.modeling import LlamaHarness
from task_refusal.pipeline import ExperimentPipeline
from task_refusal.progress import log_event


STAGES = ("prepare", "filter", "extract", "select", "evaluate", "all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the main agent refusal-direction experiment.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/main_agent_llama31.yaml")
    )
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument(
        "--categories",
        nargs="*",
        help="Optional direction subset. Use exact AgentHazard category names or global.",
    )
    parser.add_argument("--force-filter", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    log_event(
        "process_start",
        pid=os.getpid(),
        python=sys.version.split()[0],
        platform=platform.platform(),
        config=str(args.config),
        stage=args.stage,
        categories=args.categories,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    stages = (
        ("prepare", "filter", "extract", "select", "evaluate")
        if args.stage == "all"
        else (args.stage,)
    )
    if "prepare" in stages:
        prepare_manifests(config.data)
        log_event("stage_complete", stage="prepare", output=str(config.data.manifest_dir))
        if len(stages) == 1:
            return
    harness = LlamaHarness(config.model)
    pipeline = ExperimentPipeline(config, harness)
    for stage in stages:
        if stage == "prepare":
            continue
        log_event("stage_start", stage=stage)
        if stage == "filter":
            pipeline.filter_and_split(force=args.force_filter)
        elif stage == "extract":
            pipeline.extract(args.categories)
        elif stage == "select":
            pipeline.select(args.categories)
        elif stage == "evaluate":
            pipeline.evaluate()
    log_event("process_complete", stage=args.stage, output=str(config.output_dir))


if __name__ == "__main__":
    main()
