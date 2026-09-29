from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

from task_refusal.h2_steering_smoke import run_h2_steering_smoke
from task_refusal.modeling import LlamaHarness
from task_refusal.progress import log_event
from task_refusal.step_comparison import load_step_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quickly add saved H2 directions and inspect full next-agent responses."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/h2_agent_step_llama31.yaml"),
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        default=["llm_reference", "agent_all"],
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--per-step-per-label", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_step_config(args.config)
    log_event(
        "h2_smoke_process_start",
        pid=os.getpid(),
        python=sys.version.split()[0],
        platform=platform.platform(),
        config=str(args.config),
        directions=args.directions,
        alphas=args.alphas,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    harness = LlamaHarness(config.model)
    report = run_h2_steering_smoke(
        config,
        harness,
        direction_names=args.directions,
        alphas=args.alphas,
        per_step_per_label=args.per_step_per_label,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
    )
    print("condition\tlabel\tn\trefusal\tdelta\ttool_like\tempty")
    for row in report["summary"]:
        delta = row["refusal_phrase_change"]
        print(
            f"{row['condition']}\t{row['label']}\t{row['n']}\t"
            f"{row['refusal_phrase_rate']:.3f}\t"
            f"{'' if delta is None else f'{delta:+.3f}'}\t"
            f"{row['tool_call_like_rate']:.3f}\t{row['empty_rate']:.3f}"
        )
    destination = args.output_dir or config.output_dir / "steering_smoke"
    print(f"summary: {destination / 'summary.json'}")
    print(f"manual review: {destination / 'manual_review.html'}")
    log_event("h2_smoke_process_complete", output=str(destination))


if __name__ == "__main__":
    main()
