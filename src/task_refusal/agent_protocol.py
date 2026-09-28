"""Pure parser for Llama-style JSON tool calls (never executes parsed text)."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_tool_call(response: str) -> tuple[str, dict[str, Any]] | None:
    cleaned = response.strip()
    if cleaned.startswith("```json"):
        cleaned = re.sub(r"^```json\s*|\s*```$", "", cleaned).strip()
    header_name = None
    if cleaned.startswith("to="):
        match = re.match(r"^to=([A-Za-z_][A-Za-z_0-9]*)\s*(\{.*\})$",
                         cleaned, flags=re.DOTALL)
        if not match:
            return None
        header_name, cleaned = match.groups()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    if "tool_calls" in parsed and isinstance(parsed["tool_calls"], list):
        if len(parsed["tool_calls"]) != 1:
            return None
        parsed = parsed["tool_calls"][0]
    if "function" in parsed and isinstance(parsed["function"], dict):
        parsed = parsed["function"]
    if header_name is not None and not ("arguments" in parsed or "parameters" in parsed):
        return header_name, parsed
    name = parsed.get("name", header_name)
    if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
        return None
    if header_name is not None and name != header_name:
        return None
    arguments = parsed.get("arguments", parsed.get("parameters"))
    if arguments is None and header_name is not None:
        arguments = parsed
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    return name, arguments
