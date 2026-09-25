from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from task_refusal.progress import log_event

BASE_URL = "https://raw.githubusercontent.com/andyrdt/refusal_direction/main/dataset/splits"
FILENAMES = (
    "harmful_train.json",
    "harmful_test.json",
    "harmless_train.json",
    "harmless_test.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the official Arditi et al. refusal-direction splits."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/refusal_direction/dataset/splits"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate(path: Path) -> int:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a non-empty JSON list: {path}")
    if any(
        not isinstance(row, dict) or not str(row.get("instruction", "")).strip() for row in rows
    ):
        raise ValueError(f"One or more rows lack an instruction: {path}")
    return len(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for filename in FILENAMES:
        output = args.output_dir / filename
        if output.exists() and not args.force:
            count = validate(output)
            log_event("h2_reference_data_present", file=str(output), rows=count)
        else:
            url = f"{BASE_URL}/{filename}"
            log_event("h2_reference_data_download_start", url=url, output=str(output))
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            output.write_bytes(payload)
            count = validate(output)
            log_event("h2_reference_data_download_complete", file=str(output), rows=count)
        summary[filename] = count
    log_event("h2_reference_data_ready", output=str(args.output_dir), files=summary)


if __name__ == "__main__":
    main()
