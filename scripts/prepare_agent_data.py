from __future__ import annotations

import argparse
import json
from pathlib import Path

from task_refusal.config import load_config
from task_refusal.data import prepare_manifests
from task_refusal.progress import log_event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare existing agent trajectory manifests.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/main_agent_llama31.yaml")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    log_event("manifest_preparation_start", config=str(args.config))
    summary = prepare_manifests(config.data)
    log_event(
        "manifest_preparation_complete",
        output=str(config.data.manifest_dir),
        summary=summary,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
