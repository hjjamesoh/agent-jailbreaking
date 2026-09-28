"""Pinned Agent-SafetyBench first-decision transfer set, never official full score.

Only benchmark JSON schemas are read; benchmark Python environments are not
imported or run. The resulting first assistant decision is an external safety
probe, not a completed Agent-SafetyBench episode.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from task_refusal.agentalign_data import AgentAlignTask


REVISION = "74feea8de601b3a1449a93fcf70017fe61556f73"
DATA_SHA256 = "61c147ca0a783121a4a5f865d635fdaaa8b8a5a3421a09c9fac13f4953fcf3fa"
RISK_MAP = {
    "Spread unsafe information / misinformation": "misinformation",
    "Leak sensitive data / information": "privacy",
}


def load_external_tasks(repo: str | Path) -> tuple[list[AgentAlignTask], dict]:
    root = Path(repo).resolve()
    head = subprocess.run(
        ["git", "-c", f"safe.directory={root}", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    source = root / "data" / "released_data.json"
    if head != REVISION or hashlib.sha256(source.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError("External benchmark source is not the audited release.")
    records = json.loads(source.read_text(encoding="utf-8"))
    tasks = []
    excluded = {}
    fulfillable = {}
    for record in records:
        risks = record["risks"]
        if len(risks) != 1 or risks[0] not in RISK_MAP:
            continue  # Do not invent a mapping for broad or unrelated categories.
        tools = []
        bad_reason = None
        for env in record["environments"]:
            name = env.get("name")
            if not name or not name.isidentifier():
                bad_reason = "no_agent_environment"
                break
            path = root / "environments" / f"{name}.json"
            if not path.is_file():
                bad_reason = "missing_tool_schema"
                break
            schema = json.loads(path.read_text(encoding="utf-8"))
            available = {row["name"]: row for row in schema if isinstance(row, dict)
                         and isinstance(row.get("name"), str)}
            for tool_name in env.get("tools", []):
                if tool_name not in available:
                    bad_reason = "missing_exposed_tool"
                    break
                tools.append({"type": "function", "function": available[tool_name]})
        names = [row["function"]["name"] for row in tools]
        if bad_reason is None and (not tools or len(names) != len(set(names))):
            bad_reason = "empty_or_ambiguous_tools"
        if bad_reason:
            excluded[bad_reason] = excluded.get(bad_reason, 0) + 1
            continue
        tasks.append(AgentAlignTask(
            task_id=f"asb-{record['id']}", category=RISK_MAP[risks[0]],
            harmful=True, split="external", prompt=record["instruction"],
            pattern=("Agent-SafetyBench", risks[0]), tool_names=tuple(sorted(names)),
            tools=tuple(tools),
        ))
        fulfillable[f"asb-{record['id']}"] = int(record["fulfillable"])
    return tasks, {"eligible": len(tasks), "excluded": excluded,
                   "mapping": RISK_MAP, "evaluation_scope": "first_decision_only",
                   "fulfillable_by_task_id": fulfillable}
