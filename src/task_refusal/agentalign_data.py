"""Pinned, leakage-aware AgentAlign task preparation (no recorded trajectories)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable


AGENTALIGN_REVISION = "ddacefcd5f66ef18e814b2f4483c34fdd5eef05c"
AGENTALIGN_DATA_SHA256 = "c6f4b471fe6324b6f354ab9d72d6ba0b9186e7287d00e3fda08c8dcc2fb43488"
HARM_CATEGORIES = (
    "bias_and_fairness", "cyber_crime", "illicit_trade",
    "intellectual_property", "misinformation", "online_fraud",
    "privacy", "self_harm",
)
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class AgentAlignTask:
    task_id: str
    category: str
    harmful: bool
    split: str
    prompt: str
    pattern: tuple[str, ...]
    tool_names: tuple[str, ...]
    tools: tuple[dict[str, Any], ...]

    def as_row(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id, "category": self.category,
            "harmful": self.harmful, "split": self.split,
            "prompt": self.prompt, "pattern": self.pattern,
            "tool_names": self.tool_names, "tools": self.tools,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split(pattern: tuple[str, ...], seed: int) -> str:
    canonical = json.dumps([seed, pattern], ensure_ascii=False, separators=(",", ":"))
    value = int.from_bytes(hashlib.sha256(canonical.encode("utf-8")).digest()[:8], "big")
    fraction = value / (1 << 64)
    return "train" if fraction < 0.6 else "val" if fraction < 0.8 else "test"


def _record_to_task(record: dict[str, Any], seed: int) -> AgentAlignTask | None:
    task_id = str(record["id"])
    category = str(record["category"])
    # The 4,010 ToolACE/Glaive examples have different provenance and are not
    # original AgentAlign benign controls; mixing them would reintroduce the
    # cross-dataset confound observed in the earlier experiment.
    if category == "benign" and not task_id.startswith("benign-"):
        return None
    if category != "benign" and category not in HARM_CATEGORIES:
        raise ValueError(f"Unknown harmful category {category!r} in {task_id}")
    messages = record["messages"]
    if not messages or messages[0].get("role") != "user":
        raise ValueError(f"Task {task_id} lacks an initial user instruction.")
    prompt = messages[0].get("content")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Task {task_id} has an empty initial instruction.")
    tools = record["tools"]
    if not isinstance(tools, list) or not tools:
        raise ValueError(f"Task {task_id} has no tool schema.")
    names = []
    for tool in tools:
        if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            raise ValueError(f"Task {task_id} has a malformed tool schema.")
        name = tool["function"].get("name")
        if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
            raise ValueError(f"Task {task_id} has an unsafe tool name.")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError(f"Task {task_id} repeats a tool name.")
    pattern = tuple(str(item) for item in record["pattern"])
    if not pattern:
        raise ValueError(f"Task {task_id} has no abstract behavior pattern.")
    return AgentAlignTask(
        task_id=task_id, category=category, harmful=category != "benign",
        split=_split(pattern, seed), prompt=prompt, pattern=pattern,
        tool_names=tuple(sorted(names)), tools=tuple(tools),
    )


def load_tasks(path: str | Path, *, seed: int = 42, verify_source: bool = True) -> list[AgentAlignTask]:
    source = Path(path)
    if verify_source and _sha256_file(source) != AGENTALIGN_DATA_SHA256:
        raise ValueError("AgentAlign source checksum differs from the pinned release.")
    records = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("AgentAlign source must be a JSON list.")
    tasks: list[AgentAlignTask] = []
    ids: set[str] = set()
    prompts: set[tuple[str, str]] = set()
    for record in records:
        task = _record_to_task(record, seed)
        if task is None:
            continue
        if task.task_id in ids:
            raise ValueError(f"Duplicate AgentAlign task ID: {task.task_id}")
        ids.add(task.task_id)
        # Avoid identical instructions in multiple folds, even when the source
        # attached distinct tool variants or generated responses.
        normalized = " ".join(task.prompt.lower().split())
        prompt_key = (task.category, normalized)
        if prompt_key in prompts:
            continue
        prompts.add(prompt_key)
        tasks.append(task)
    return tasks


def matched_by_category(
    tasks: Iterable[AgentAlignTask], *, minimum_per_side: dict[str, int] | None = None,
    strict: bool = True,
) -> tuple[dict[tuple[str, str, bool], list[AgentAlignTask]], dict[str, Any]]:
    """Match benign/harmful on the *exact* set of exposed tool names per fold."""
    all_tasks = list(tasks)
    benign: dict[tuple[str, tuple[str, ...]], list[AgentAlignTask]] = defaultdict(list)
    for task in all_tasks:
        if not task.harmful:
            benign[(task.split, task.tool_names)].append(task)
    grouped: dict[tuple[str, str, bool], list[AgentAlignTask]] = {}
    summary: dict[str, Any] = {"matching": "exact_tool_name_set", "categories": {}}
    minimums = minimum_per_side or {"train": 1, "val": 1, "test": 1}
    for category in HARM_CATEGORIES:
        category_summary = {}
        for split in SPLITS:
            harmful = [
                task for task in all_tasks
                if task.category == category and task.split == split
                and benign.get((split, task.tool_names))
            ]
            signatures = {task.tool_names for task in harmful}
            safe = [
                task for task in all_tasks
                if not task.harmful and task.split == split and task.tool_names in signatures
            ]
            grouped[(category, split, True)] = harmful
            grouped[(category, split, False)] = safe
            category_summary[split] = {"harmful": len(harmful), "benign": len(safe),
                                       "matched_tool_sets": len(signatures)}
            required = minimums.get(split, 1)
            if strict and min(len(harmful), len(safe)) < required:
                raise ValueError(
                    f"{category}/{split} lacks matched tasks: "
                    f"{len(harmful)} harmful, {len(safe)} benign; need {required} each"
                )
        summary["categories"][category] = category_summary
    split_patterns = {split: set() for split in SPLITS}
    for task in all_tasks:
        split_patterns[task.split].add(task.pattern)
    if any(split_patterns[a] & split_patterns[b] for a in SPLITS for b in SPLITS if a != b):
        raise AssertionError("Abstract behavior patterns leaked across folds.")
    summary["split_pattern_counts"] = {split: len(patterns) for split, patterns in split_patterns.items()}
    summary["unique_tasks"] = len(all_tasks)
    summary["source_counts"] = dict(Counter(task.category for task in all_tasks))
    return grouped, summary


def find_supported_split(
    tasks: Iterable[AgentAlignTask], *, initial_seed: int = 42,
    max_seeds: int = 1000,
    minimum_per_side: dict[str, int] | None = None,
) -> tuple[int, list[AgentAlignTask], dict[tuple[str, str, bool], list[AgentAlignTask]], dict[str, Any]]:
    """Find a group split meeting preregistered *count* gates, not outcome gates.

    This is necessary because some rare category/tool combinations disappear
    from a fold under a naive hash split. No model activations or behavior
    scores are inspected when choosing the seed.
    """
    original = list(tasks)
    if not original or max_seeds < 1:
        raise ValueError("Tasks and max_seeds must be non-empty and positive.")
    minimums = minimum_per_side or {"train": 64, "val": 8, "test": 8}
    for seed in range(initial_seed, initial_seed + max_seeds):
        split_for_pattern = {task.pattern: _split(task.pattern, seed) for task in original}
        benign_counts: Counter[tuple[str, tuple[str, ...]]] = Counter()
        harmful_counts: Counter[tuple[str, str, tuple[str, ...]]] = Counter()
        for task in original:
            split = split_for_pattern[task.pattern]
            if task.harmful:
                harmful_counts[(split, task.category, task.tool_names)] += 1
            else:
                benign_counts[(split, task.tool_names)] += 1
        supported = True
        for category in HARM_CATEGORIES:
            for split in SPLITS:
                matched_signatures = {
                    names for (fold, cat, names), count in harmful_counts.items()
                    if fold == split and cat == category and count
                    and benign_counts[(split, names)]
                }
                harmful_n = sum(harmful_counts[(split, category, names)]
                                for names in matched_signatures)
                benign_n = sum(benign_counts[(split, names)]
                               for names in matched_signatures)
                if min(harmful_n, benign_n) < minimums[split]:
                    supported = False
                    break
            if not supported:
                break
        if supported:
            reassigned = [replace(task, split=split_for_pattern[task.pattern])
                          for task in original]
            grouped, summary = matched_by_category(
                reassigned, minimum_per_side=minimums
            )
            summary["split_seed"] = seed
            summary["minimum_per_side"] = minimums
            return seed, reassigned, grouped, summary
    raise ValueError(
        f"No outcome-blind pattern split among {max_seeds} seeds has enough "
        f"matched tasks in every category; review the support summary."
    )


def balanced_tool_pairs(
    harmful: Iterable[AgentAlignTask], benign: Iterable[AgentAlignTask],
    *, max_per_side: int, seed: int,
) -> tuple[list[AgentAlignTask], list[AgentAlignTask]]:
    """Balance each side on exact exposed tool signature before mean subtraction."""
    if max_per_side < 1:
        raise ValueError("max_per_side must be positive")
    h_by_tool: dict[tuple[str, ...], list[AgentAlignTask]] = defaultdict(list)
    b_by_tool: dict[tuple[str, ...], list[AgentAlignTask]] = defaultdict(list)
    for task in harmful:
        h_by_tool[task.tool_names].append(task)
    for task in benign:
        b_by_tool[task.tool_names].append(task)

    def order(task: AgentAlignTask) -> str:
        return hashlib.sha256(f"{seed}:{task.task_id}".encode()).hexdigest()

    paired: list[tuple[AgentAlignTask, AgentAlignTask]] = []
    for signature in sorted(h_by_tool):
        h = sorted(h_by_tool[signature], key=order)
        b = sorted(b_by_tool[signature], key=order)
        paired.extend(zip(h[:min(len(h), len(b))], b[:min(len(h), len(b))]))
    paired.sort(key=lambda pair: order(pair[0]))
    selected = paired[:max_per_side]
    return [row[0] for row in selected], [row[1] for row in selected]
