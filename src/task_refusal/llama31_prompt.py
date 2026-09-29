"""Deterministic rendering of Meta Llama 3.1 custom-tool conversations.

The pinned tokenizer's Jinja template is the specification, but recent
Transformers/Jinja combinations can return a non-string native value for some
tool schemas. Rendering the small supported subset explicitly keeps the prompt
format fixed and makes malformed conversation state fail closed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


HEADER = "<|start_header_id|>{role}<|end_header_id|>\n\n"
EOT = "<|eot_id|>"


def _json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=True, indent=indent, sort_keys=True)


def render_llama31_tool_chat(
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]],
    *,
    bos_token: str,
    add_generation_prompt: bool = True,
) -> str:
    """Render the Llama 3.1 custom-tool template used by this experiment."""
    if not isinstance(bos_token, str) or not bos_token:
        raise ValueError("Llama 3.1 tokenizer has no BOS token string.")
    if not messages:
        raise ValueError("An agent conversation cannot be empty.")
    rows = list(messages)
    system = ""
    if rows[0].get("role") == "system":
        system = rows.pop(0).get("content")
        if not isinstance(system, str):
            raise TypeError("System message content must be text.")
        system = system.strip()
    if not rows or rows[0].get("role") != "user":
        raise ValueError("Llama 3.1 tool chat requires an initial user message.")
    first_user = rows.pop(0).get("content")
    if not isinstance(first_user, str):
        raise TypeError("Initial user message content must be text.")

    parts = [
        bos_token,
        HEADER.format(role="system"),
        "Environment: ipython\n",
        "Cutting Knowledge Date: December 2023\n",
        "Today Date: 26 Jul 2024\n\n",
        system,
        EOT,
        HEADER.format(role="user"),
        "Given the following functions, please respond with a JSON for a function call ",
        "with its proper arguments that best answers the given prompt.\n\n",
        'Respond in the format {"name": function name, "parameters": dictionary of '
        "argument name and its value}.",
        "Do not use variables.\n\n",
    ]
    for tool in tools:
        if not isinstance(tool, dict):
            raise TypeError("Tool definitions must be JSON objects.")
        parts.extend((_json(tool, indent=4), "\n\n"))
    parts.extend((first_user.strip(), EOT))

    for message in rows:
        role = message.get("role")
        if "tool_calls" in message:
            calls = message["tool_calls"]
            if not isinstance(calls, list) or len(calls) != 1:
                raise ValueError("Llama 3.1 supports exactly one tool call per turn.")
            function = calls[0].get("function")
            if not isinstance(function, dict):
                raise TypeError("Assistant tool call lacks a function object.")
            name = function.get("name")
            arguments = function.get("arguments")
            if not isinstance(name, str) or not isinstance(arguments, dict):
                raise TypeError("Tool call name must be text and arguments must be an object.")
            parts.extend((
                HEADER.format(role="assistant"),
                '{"name": "', name, '", "parameters": ', _json(arguments), "}", EOT,
            ))
        elif role in ("tool", "ipython"):
            content = message.get("content")
            # This mirrors the pinned template's ``mapping or iterable`` test;
            # strings are therefore JSON-quoted in an ipython result message.
            rendered = _json(content) if isinstance(content, (Mapping, list, tuple, str)) \
                else str(content)
            parts.extend((HEADER.format(role="ipython"), rendered, EOT))
        else:
            content = message.get("content")
            if role not in ("user", "assistant") or not isinstance(content, str):
                raise TypeError(f"Unsupported Llama 3.1 message role/content: {role!r}")
            parts.extend((HEADER.format(role=role), content.strip(), EOT))
    if add_generation_prompt:
        parts.append(HEADER.format(role="assistant"))
    rendered = "".join(parts)
    if not isinstance(rendered, str):
        raise AssertionError("Llama prompt renderer did not produce text.")
    return rendered
