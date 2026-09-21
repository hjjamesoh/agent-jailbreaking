from __future__ import annotations

import json
import zipfile
from pathlib import Path

from task_refusal.data import load_agenthazard_states, load_agentlens_states


CATEGORIES = ("System_RCE", "Data_Exfiltration")


def test_agentlens_deduplicates_and_keeps_task_groups(tmp_path: Path) -> None:
    row = {
        "rollout_id": 7,
        "task_id": 3,
        "task_description": "Do the same task",
        "step_number": 1,
        "input": [{"role": "user", "content": "hello"}],
        "output": "done",
        "label": 0,
    }
    first = tmp_path / "train.json"
    second = tmp_path / "test.json"
    first.write_text(json.dumps([row]), encoding="utf-8")
    second.write_text(json.dumps([row]), encoding="utf-8")
    states = load_agentlens_states((first, second), 42, (0.7, 0.15, 0.15))
    assert len(states) == 1
    assert states[0].messages[0]["role"] == "user"


def test_agenthazard_group_split_never_leaks(tmp_path: Path) -> None:
    dataset = []
    trace_rows = []
    for index in range(40):
        category = CATEGORIES[index % 2]
        dataset.append(
            {
                "id": index,
                "category": category,
                "jailbreak_method": "Direct",
                "original_id": index // 2,
            }
        )
        trace_rows.append(
            {
                **dataset[-1],
                "query": f"task {index}",
                "turns": [
                    {"turn_idx": 1, "input": "step one", "output": "result one"},
                    {"turn_idx": 2, "input": "step two", "output": "result two"},
                ],
            }
        )
    dataset_path = tmp_path / "dataset.json"
    trace_path = tmp_path / "trace.jsonl"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    trace_path.write_text(
        "\n".join(json.dumps(row) for row in trace_rows), encoding="utf-8"
    )
    states = load_agenthazard_states(
        dataset_path,
        (str(trace_path),),
        CATEGORIES,
        42,
        (0.7, 0.15, 0.15),
    )
    splits_by_group: dict[str, set[str]] = {}
    for state in states:
        splits_by_group.setdefault(state.group_id, set()).add(state.split)
    assert all(len(splits) == 1 for splits in splits_by_group.values())
    assert {state.category for state in states} == set(CATEGORIES)


def test_agenthazard_reads_official_zip_archives(tmp_path: Path) -> None:
    dataset = [
        {
            "id": 1,
            "category": "System_RCE",
            "jailbreak_method": "Direct",
            "original_id": 1,
        },
        {
            "id": 2,
            "category": "Data_Exfiltration",
            "jailbreak_method": "Direct",
            "original_id": 2,
        },
    ]
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    archive_path = tmp_path / "official-traces.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for row in dataset:
            trace = {
                **row,
                "query": f"task {row['id']}",
                "turns": [
                    {"turn_idx": 1, "input": "step", "output": "result"}
                ],
            }
            archive.writestr(
                f"model/trajectory_{row['id']}.jsonl", json.dumps(trace) + "\n"
            )
        # Official archives can contain macOS AppleDouble resource forks whose
        # names end in .jsonl even though their contents are binary.
        archive.writestr("__MACOSX/model/._trajectory_1.jsonl", b"\x00\xa3\x81\x00")
    states = load_agenthazard_states(
        dataset_path,
        (str(archive_path),),
        CATEGORIES,
        42,
        (0.7, 0.15, 0.15),
    )
    assert len(states) == 2
    assert all(".zip!" in str(state.provenance) for state in states)
