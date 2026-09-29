#!/usr/bin/env python3
"""Run the preregistered causal AgentAlign category-direction experiment."""

from __future__ import annotations

import argparse
import platform
import sys

from task_refusal.exp3_pipeline import Exp3Pipeline
from task_refusal.progress import log_event


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp3_agentalign_llama31.yaml")
    parser.add_argument("--stage", choices=("all", "smoke", "preflight", "extract", "proxy",
                                            "validate", "test", "external", "posttool"),
                        default="all")
    args = parser.parse_args()
    log_event("exp3_process_start", config=args.config, stage=args.stage,
              python=sys.version.split()[0], platform=platform.platform())
    pipeline = Exp3Pipeline(args.config)
    pipeline.run(args.stage)
    log_event("exp3_process_complete", config=args.config, stage=args.stage,
              output=str(pipeline.output))


if __name__ == "__main__":
    main()
