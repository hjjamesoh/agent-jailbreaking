#!/usr/bin/env python3
"""Download the public sources used to author controlled paired prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download

from task_refusal.progress import log_event


AGENT_SAFETY_REVISION = "74feea8de601b3a1449a93fcf70017fe61556f73"
BEAVERTAILS_REPO = "PKU-Alignment/BeaverTails"
BEAVERTAILS_REVISION = "8401fe609d288129cc684a9b3be6a93e41cfe678"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-c", f"safe.directory={path.resolve()}", "-C", str(path),
         "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw/controlled_pairs_v1")
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    beaver = root / "BeaverTails"
    revision = BEAVERTAILS_REVISION
    log_event("paired_source_download_start", dataset=BEAVERTAILS_REPO,
              revision=revision)
    snapshot_download(
        repo_id=BEAVERTAILS_REPO, repo_type="dataset", revision=revision,
        local_dir=beaver,
        allow_patterns=[
            "round0/330k/*.jsonl.xz", "*.json", "*.jsonl", "*.csv",
            "README.md", "LICENSE*",
        ],
    )

    safety = root / "Agent-SafetyBench"
    if not safety.exists():
        subprocess.run([
            "git", "clone", "https://github.com/thu-coai/Agent-SafetyBench.git",
            str(safety),
        ], check=True)
    subprocess.run([
        "git", "-c", f"safe.directory={safety.resolve()}", "-C", str(safety),
        "fetch", "origin", AGENT_SAFETY_REVISION,
    ], check=True)
    subprocess.run([
        "git", "-c", f"safe.directory={safety.resolve()}", "-C", str(safety),
        "checkout", "--detach", AGENT_SAFETY_REVISION,
    ], check=True)
    if _git_head(safety) != AGENT_SAFETY_REVISION:
        raise ValueError("Agent-SafetyBench checkout is not at the pinned revision")

    files = []
    for dataset, directory in (("BeaverTails", beaver), ("Agent-SafetyBench", safety)):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and (
                path.suffix.lower() in {".json", ".jsonl", ".csv"}
                or path.name.lower().endswith(".jsonl.xz")
            ):
                files.append({
                    "dataset": dataset,
                    "path": str(path.relative_to(root)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                })
    manifest = {
        "sources": {
            "BeaverTails": {
                "repo": BEAVERTAILS_REPO,
                "revision": revision,
                "license": "CC-BY-NC-4.0",
                "use": "topical source seeds; prompts are not exported verbatim",
            },
            "Agent-SafetyBench": {
                "repo": "thu-coai/Agent-SafetyBench",
                "revision": AGENT_SAFETY_REVISION,
                "license": "see pinned upstream repository",
                "use": "agent-risk source seeds and environment inspiration",
            },
            "AgentHarm": {
                "repo": "ai-safety-institute/AgentHarm",
                "downloaded": False,
                "use": "held out for external evaluation; not expanded into authoring data",
                "reason": "upstream requests that examples not be used for training",
            },
        },
        "files": files,
    }
    output = root / "source_manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log_event("paired_sources_ready", manifest=str(output), files=len(files))


if __name__ == "__main__":
    main()
