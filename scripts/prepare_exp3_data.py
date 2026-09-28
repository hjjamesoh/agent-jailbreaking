#!/usr/bin/env python3
"""Fetch only pinned public AgentAlign assets; no Docker or real services."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download

from task_refusal.agent_simulator import AGENTALIGN_CODE_REVISION, verify_simulator_source
from task_refusal.agentalign_data import (
    AGENTALIGN_DATA_SHA256, AGENTALIGN_REVISION, HARM_CATEGORIES, SPLITS,
    balanced_tool_pairs, find_supported_split, load_tasks,
)
from task_refusal.progress import log_event

SAFETYBENCH_REVISION = "74feea8de601b3a1449a93fcf70017fe61556f73"
# Hash of the raw LF bytes at SAFETYBENCH_REVISION.  Do not compute this from
# a Windows checkout: Git may rewrite JSON line endings to CRLF.
SAFETYBENCH_DATA_SHA256 = "59dd0333001ef767766d803e97086ec02af0fbf7ff1f7070b3797863b0dacbe2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_normalized_lf(path: Path) -> str:
    """Hash a Git text asset independent of checkout line-ending policy."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/AgentAlign_exp3")
    parser.add_argument("--summary", default="data/processed/exp3/support.json")
    parser.add_argument("--external-repo", default="data/raw/AgentSafetyBench_exp3")
    args = parser.parse_args()
    root = Path(args.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    dataset = root / "agent_align_data_v3.json"
    repo = root / "source"
    if not dataset.exists():
        log_event("exp3_dataset_download_start", revision=AGENTALIGN_REVISION)
        cached = Path(hf_hub_download(
            repo_id="jc-ryan/AgentAlign", repo_type="dataset",
            filename="agent_align_data_v3.json", revision=AGENTALIGN_REVISION,
        ))
        if _sha256(cached) != AGENTALIGN_DATA_SHA256:
            raise ValueError("Downloaded AgentAlign dataset checksum mismatch.")
        shutil.copyfile(cached, dataset)
    if _sha256(dataset) != AGENTALIGN_DATA_SHA256:
        raise ValueError("Local AgentAlign dataset checksum mismatch.")
    if not repo.exists():
        log_event("exp3_simulator_clone_start", revision=AGENTALIGN_CODE_REVISION)
        subprocess.run(["git", "clone", "https://github.com/jc-ryan/AgentAlign.git",
                        str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach",
                        AGENTALIGN_CODE_REVISION], check=True)
    verify_simulator_source(repo)
    external = Path(args.external_repo)
    if not external.exists():
        log_event("exp3_external_clone_start", revision=SAFETYBENCH_REVISION)
        subprocess.run(["git", "clone", "https://github.com/thu-coai/Agent-SafetyBench.git",
                        str(external)], check=True)
        subprocess.run(["git", "-C", str(external), "checkout", "--detach",
                        SAFETYBENCH_REVISION], check=True)
    external_head = subprocess.run(
        ["git", "-c", f"safe.directory={external.resolve()}", "-C", str(external),
         "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
    ).stdout.strip()
    if external_head != SAFETYBENCH_REVISION or _sha256_normalized_lf(
        external / "data" / "released_data.json") != SAFETYBENCH_DATA_SHA256:
        raise ValueError("Agent-SafetyBench external release differs from the pinned source.")
    tasks = load_tasks(dataset)
    seed, _assigned, grouped, summary = find_supported_split(
        tasks, initial_seed=42, max_seeds=1000,
        minimum_per_side={"train": 64, "val": 16, "test": 16},
    )
    balanced_counts = {}
    for category in HARM_CATEGORIES:
        balanced_counts[category] = {}
        for split in SPLITS:
            harmful, benign = balanced_tool_pairs(
                grouped[category, split, True], grouped[category, split, False],
                max_per_side=100000, seed=seed,
            )
            count = min(len(harmful), len(benign))
            balanced_counts[category][split] = count
            if count < {"train": 60, "val": 16, "test": 16}[split]:
                raise ValueError(f"Insufficient exactly tool-balanced {category}/{split}: {count}")
    summary["balanced_pair_counts"] = balanced_counts
    summary.update({"seed": seed, "dataset_sha256": AGENTALIGN_DATA_SHA256,
                    "dataset_revision": AGENTALIGN_REVISION,
                    "simulator_revision": AGENTALIGN_CODE_REVISION,
                    "external_revision": SAFETYBENCH_REVISION,
                    "external_data_sha256": SAFETYBENCH_DATA_SHA256})
    output = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log_event("exp3_data_ready", summary=str(output), seed=seed,
              unique_tasks=summary["unique_tasks"])


if __name__ == "__main__":
    main()
