from __future__ import annotations

import glob
import hashlib
import io
import json
import random
import re
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from task_refusal.config import DataConfig


Message = dict[str, str]
Context = list[Message]


@dataclass(frozen=True)
class AgentState:
    state_id: str
    episode_id: str
    group_id: str
    source: str
    split: str
    step: int
    messages: tuple[Message, ...]
    output: str
    harm_label: int | None = None
    category: str | None = None
    attack_strategy: str | None = None
    original_id: str | None = None
    provenance: str | None = None

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "AgentState":
        return cls(
            state_id=str(row["state_id"]),
            episode_id=str(row["episode_id"]),
            group_id=str(row["group_id"]),
            source=str(row["source"]),
            split=str(row["split"]),
            step=int(row["step"]),
            messages=tuple(
                {"role": str(message["role"]), "content": str(message["content"])}
                for message in row["messages"]
            ),
            output=str(row.get("output", "")),
            harm_label=None if row.get("harm_label") is None else int(row["harm_label"]),
            category=row.get("category"),
            attack_strategy=row.get("attack_strategy"),
            original_id=None if row.get("original_id") is None else str(row["original_id"]),
            provenance=row.get("provenance"),
        )

    def context(self) -> Context:
        return [dict(message) for message in self.messages]


def _canonical_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _digest(value: str, length: int = 20) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assign_split(group_id: str, seed: int, ratios: Sequence[float]) -> str:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < ratios[0]:
        return "train"
    if value < ratios[0] + ratios[1]:
        return "val"
    return "test"


def _clean_messages(messages: Sequence[dict[str, Any]]) -> tuple[Message, ...]:
    cleaned: list[Message] = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        content = message.get("content", "")
        if not role or content is None:
            continue
        cleaned.append({"role": role, "content": str(content)})
    if not cleaned:
        raise ValueError("An agent state cannot have an empty message history.")
    return tuple(cleaned)


def load_agentlens_states(
    paths: Sequence[Path], seed: int, ratios: Sequence[float]
) -> list[AgentState]:
    states: list[AgentState] = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"AgentLens file not found: {path}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Expected a JSON list in {path}")
        for index, row in enumerate(rows):
            messages = _clean_messages(row["input"])
            output = str(row.get("output", ""))
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"Unexpected AgentLens label {label!r} in {path}")
            task_description = _canonical_text(str(row.get("task_description", "")))
            group_id = f"mas-task-{_digest(task_description)}"
            content_key = json.dumps(
                {"messages": messages, "output": output, "label": label},
                ensure_ascii=False,
                sort_keys=True,
            )
            state_hash = _digest(content_key)
            if state_hash in seen:
                continue
            seen.add(state_hash)
            rollout = str(row.get("rollout_id", "unknown"))
            states.append(
                AgentState(
                    state_id=f"mas-{state_hash}",
                    episode_id=f"mas-rollout-{rollout}",
                    group_id=group_id,
                    source="agentlens_mas_llama31",
                    split=_assign_split(group_id, seed, ratios),
                    step=int(row.get("step_number", index)),
                    messages=messages,
                    output=output,
                    harm_label=label,
                    original_id=str(row.get("task_id", "")),
                    provenance=str(path),
                )
            )
    return states


def _trajectory_messages(row: dict[str, Any], turn_index: int) -> tuple[Message, ...]:
    query = str(row.get("query", "")).strip()
    messages: list[Message] = []
    if query:
        messages.append({"role": "user", "content": query})
    turns = row.get("turns") or []
    for index, turn in enumerate(turns[: turn_index + 1]):
        turn_input = str(turn.get("input", "")).strip()
        if turn_input and (not messages or turn_input != messages[-1]["content"]):
            messages.append({"role": "user", "content": turn_input})
        if index < turn_index:
            messages.append({"role": "assistant", "content": str(turn.get("output", ""))})
    return _clean_messages(messages)


def _iter_trajectory_rows(trace_paths: Sequence[Path]):
    """Yield official AgentHazard JSONL records without extracting ZIP archives."""
    for path in trace_paths:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                members = sorted(
                    name for name in archive.namelist() if name.lower().endswith(".jsonl")
                )
                for member in members:
                    with archive.open(member) as raw_handle, io.TextIOWrapper(
                        raw_handle, encoding="utf-8-sig"
                    ) as handle:
                        for line_number, line in enumerate(handle, 1):
                            if line.strip():
                                yield f"{path}!{member}", line_number, json.loads(line)
        else:
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8-sig").splitlines(), 1
            ):
                if line.strip():
                    yield str(path), line_number, json.loads(line)


def load_agenthazard_states(
    dataset_path: Path,
    trace_globs: Sequence[str],
    categories: Sequence[str],
    seed: int,
    ratios: Sequence[float],
) -> list[AgentState]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"AgentHazard dataset not found: {dataset_path}")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    metadata = {str(row["id"]): row for row in dataset}
    trace_paths = sorted(
        {
            Path(name)
            for pattern in trace_globs
            for name in glob.glob(pattern, recursive=True)
        }
    )
    if not trace_paths:
        raise FileNotFoundError(
            "No AgentHazard trajectory JSONL matched: " + ", ".join(trace_globs)
        )
    allowed = set(categories)
    trace_rows: list[tuple[str, int, dict[str, Any], dict[str, Any], str, str]] = []
    for provenance, line_number, row in _iter_trajectory_rows(trace_paths):
        row_id = str(row["id"])
        base = metadata.get(row_id, row)
        category = str(row.get("category") or base.get("category"))
        if category not in allowed:
            continue
        original_id = str(
            row.get("original_id") or base.get("original_id") or row_id
        )
        trace_rows.append(
            (provenance, line_number, row, base, category, original_id)
        )

    split_by_group: dict[str, str] = {}
    for category in categories:
        group_ids = sorted(
            {
                f"{row_category}:{original_id}"
                for _provenance, _line, _row, _base, row_category, original_id
                in trace_rows
                if row_category == category
            }
        )
        random.Random(_digest(f"{seed}:{category}")).shuffle(group_ids)
        train_end = int(len(group_ids) * ratios[0])
        val_end = train_end + int(len(group_ids) * ratios[1])
        for index, group_id in enumerate(group_ids):
            split_by_group[group_id] = (
                "train" if index < train_end else "val" if index < val_end else "test"
            )

    states: list[AgentState] = []
    seen: set[str] = set()
    for provenance, line_number, row, base, category, original_id in trace_rows:
        row_id = str(row["id"])
        split_key = f"{category}:{original_id}"
        group_id = f"agenthazard-{category}-{original_id}"
        split = split_by_group[split_key]
        turns = row.get("turns") or []
        for turn_index, turn in enumerate(turns):
            messages = _trajectory_messages(row, turn_index)
            content_key = json.dumps(
                {"messages": messages, "category": category, "original_id": original_id},
                ensure_ascii=False,
                sort_keys=True,
            )
            state_hash = _digest(content_key)
            if state_hash in seen:
                continue
            seen.add(state_hash)
            states.append(
                AgentState(
                    state_id=f"agenthazard-{state_hash}",
                    episode_id=f"agenthazard-{row_id}",
                    group_id=group_id,
                    source="agenthazard_public_trace",
                    split=split,
                    step=int(turn.get("turn_idx", turn_index + 1)),
                    messages=messages,
                    output=str(turn.get("output", "")),
                    harm_label=None,
                    category=category,
                    attack_strategy=str(
                        row.get("jailbreak_method") or base.get("jailbreak_method") or ""
                    ),
                    original_id=original_id,
                    provenance=f"{provenance}:{line_number}",
                )
            )
    missing = sorted(allowed - {state.category for state in states})
    if missing:
        raise ValueError(f"AgentHazard traces contain no states for categories: {missing}")
    return states


def write_states(path: Path, states: Iterable[AgentState]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for state in states:
            handle.write(json.dumps(asdict(state), ensure_ascii=False) + "\n")


def load_states(path: Path) -> list[AgentState]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [AgentState.from_dict(json.loads(line)) for line in handle if line.strip()]


def prepare_manifests(config: DataConfig) -> dict[str, Any]:
    mas = load_agentlens_states(
        (config.agentlens_train, config.agentlens_test),
        config.seed,
        config.split_ratios,
    )
    hazard = load_agenthazard_states(
        config.agenthazard_dataset,
        config.agenthazard_trace_globs,
        config.categories,
        config.seed,
        config.split_ratios,
    )
    config.manifest_dir.mkdir(parents=True, exist_ok=True)
    write_states(config.manifest_dir / "agentlens_states.jsonl", mas)
    write_states(config.manifest_dir / "agenthazard_states.jsonl", hazard)
    summary = {
        "sources": {
            "agentlens": [
                {"path": str(path), "sha256": _file_sha256(path)}
                for path in (config.agentlens_train, config.agentlens_test)
            ],
            "agenthazard_dataset": {
                "path": str(config.agenthazard_dataset),
                "sha256": _file_sha256(config.agenthazard_dataset),
            },
            "agenthazard_traces": [
                {"path": str(path), "sha256": _file_sha256(path)}
                for path in sorted(
                    {
                        Path(name)
                        for pattern in config.agenthazard_trace_globs
                        for name in glob.glob(pattern, recursive=True)
                    }
                )
            ],
        },
        "agentlens": {
            "states": len(mas),
            "groups": len({state.group_id for state in mas}),
            "splits": Counter(state.split for state in mas),
            "labels": Counter(state.harm_label for state in mas),
        },
        "agenthazard": {
            "states": len(hazard),
            "groups": len({state.group_id for state in hazard}),
            "splits": Counter(state.split for state in hazard),
            "categories": Counter(state.category for state in hazard),
            "strategies": Counter(state.attack_strategy for state in hazard),
        },
    }
    serializable = json.loads(json.dumps(summary, default=dict))
    (config.manifest_dir / "summary.json").write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    validate_no_group_leakage((*mas, *hazard))
    return serializable


def validate_no_group_leakage(states: Sequence[AgentState]) -> None:
    group_splits: dict[str, set[str]] = {}
    for state in states:
        group_splits.setdefault(state.group_id, set()).add(state.split)
    leaking = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    if leaking:
        raise ValueError(f"Group leakage across splits: {list(leaking.items())[:5]}")


def deterministic_sample(states: Sequence[AgentState], count: int, seed: int) -> list[AgentState]:
    if len(states) < count:
        raise ValueError(f"Requested {count} states but only {len(states)} are eligible.")
    values = list(states)
    random.Random(seed).shuffle(values)
    return values[:count]
