"""Execute only the pinned, synthetic AgentAlign tool implementations.

The official functions return simulated effects; no shell, payment, network or
social-media action is delegated to the host. A changed upstream file fails
closed before any import. Keep this module isolated from real tool APIs.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import math
import random
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


AGENTALIGN_CODE_REVISION = "efb728f7211163a39eacdeab77b4d4e0618a5bc7"
TOOL_DIR = Path("multi_step_trajectory_generation/synthetic_tools")
PINNED_FILES = {
    "__init__.py": "0f025a938c926f19369f2cd6bb68731aff5cd38a70e48fc59ac4901d7784ab70",
    "Artificial_Intelligence_Machine_Learning.py": "bc5d61283e5cf574d15ce94f02fb2c4b3448483395abe8fa9aca3a87bb3ed072",
    "Communication.py": "0b10c6a21cdb5e46a07097eb3037f3ec516e2826b71a01c37771ca0ae0b898a0",
    "Cryptography.py": "27cb3b0d9ba2eb49e557ef84ecf5ad98288e67a5f48304b1e082e9e9de02f23f",
    "eCommerce.py": "64c5ef4c9993f9ddec6d76f7c618a1dfabe3f924b056efb12f50b27ac192b77e",
    "LocalServices.py": "cfc5a969f1d7f253180acb293779b021a06199f84b04fe8057bc1ab3802b363e",
    "Payments.py": "bade08dd59b365fc3855e85638bbbebd4bfb5e312850bf4bfd5f53e3b9f35229",
    "Search.py": "cb748d8cc5850428eb4f9ddd8ed20c6f2513e38135f2dc1e9f42355ad948f5b7",
    "Social.py": "d1936420bc23525fe2f7ee63c0672ce80145d2a6836c435d9f52f6e5f7b964d2",
    "System_Tools.py": "57dc5e8ae49c9eb045fea2149efffdaebbf5d62af5581683cd61422dd11e721c",
}


def verify_simulator_source(repo: str | Path) -> Path:
    root = Path(repo).resolve()
    result = subprocess.run(
        ["git", "-c", f"safe.directory={root}", "-C", str(root),
         "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    if result.stdout.strip() != AGENTALIGN_CODE_REVISION:
        raise ValueError("AgentAlign tool repository is not at the audited commit.")
    directory = root / TOOL_DIR
    for name, expected in PINNED_FILES.items():
        path = directory / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"AgentAlign synthetic tool source changed: {path}")
    return directory


def _validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
    parameters = schema["function"].get("parameters", {})
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    if any(key not in arguments for key in parameters.get("required", [])):
        raise ValueError("Required tool argument is missing.")
    properties = parameters.get("properties", {})
    if any(key not in properties for key in arguments):
        raise ValueError("Unknown tool argument.")
    types = {
        "string": str, "integer": int, "number": (int, float),
        "boolean": bool, "array": list, "object": dict, "null": type(None),
    }
    for key, value in arguments.items():
        expected = properties[key].get("type")
        # JSON Schema permits a union such as ["string", "integer"] or
        # ["string", "null"]. AgentAlign uses these union types. Looking the
        # list up directly in ``types`` used to raise ``TypeError: unhashable
        # type: list`` instead of validating the model's call.
        if expected is None:
            declared = []
        elif isinstance(expected, str):
            declared = [expected]
        elif isinstance(expected, list) and all(isinstance(item, str) for item in expected):
            declared = expected
        else:
            raise ValueError(f"Tool argument {key!r} has a malformed type schema.")
        python_types = tuple(types[item] for item in declared if item in types)
        if python_types and not isinstance(value, python_types):
            raise ValueError(f"Tool argument {key!r} has the wrong type.")
        if (("integer" in declared or "number" in declared)
                and isinstance(value, bool) and "boolean" not in declared):
            # bool subclasses int in Python, but JSON Schema treats boolean as
            # a distinct type unless it is explicitly included in the union.
            raise ValueError(f"Tool argument {key!r} must not be boolean.")
        if "enum" in properties[key] and value not in properties[key]["enum"]:
            raise ValueError(f"Tool argument {key!r} is outside the declared enum.")


def _declared_types(property_schema: dict[str, Any]) -> list[str]:
    expected = property_schema.get("type")
    if expected is None:
        return []
    if isinstance(expected, str):
        return [expected]
    if isinstance(expected, list) and all(isinstance(item, str) for item in expected):
        return expected
    raise ValueError("Malformed JSON Schema type declaration.")


def _structured_string(value: str, expected: type) -> Any | None:
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, expected):
            # ast.literal_eval is safe, but constrain its result to JSON data.
            try:
                json.dumps(parsed, allow_nan=False)
            except (TypeError, ValueError):
                continue
            return parsed
    return None


def normalize_arguments(
    schema: dict[str, Any], arguments: dict[str, Any],
) -> tuple[dict[str, Any], tuple[dict[str, str], ...]]:
    """Apply only deterministic, meaning-preserving schema conversions."""
    normalized = copy.deepcopy(arguments)
    parameters = schema["function"].get("parameters", {})
    properties = parameters.get("properties", {})
    required = set(parameters.get("required", []))
    events: list[dict[str, str]] = []
    for key in list(normalized):
        if key not in properties:
            continue  # Unknown arguments remain visible and fail validation.
        value = normalized[key]
        if not isinstance(value, str):
            continue
        declared = _declared_types(properties[key])
        stripped = value.strip()
        lowered = stripped.lower()
        conversion = None
        converted: Any = value
        # Never reinterpret a value that is already valid as a string.  In
        # particular, an optional string whose literal value is "null" must
        # remain that string rather than silently disappearing.
        if lowered == "null" and "string" not in declared:
            if "null" in declared:
                converted, conversion = None, "string_to_null"
            elif key not in required:
                del normalized[key]
                events.append({"argument": key,
                               "conversion": "string_null_to_omitted_optional"})
                continue
        elif "string" not in declared and "boolean" in declared \
                and lowered in ("true", "false"):
            converted, conversion = lowered == "true", "string_to_boolean"
        elif "string" not in declared and "integer" in declared \
                and re.fullmatch(r"[+-]?\d+", stripped):
            converted, conversion = int(stripped), "string_to_integer"
        elif "string" not in declared and "number" in declared \
                and re.fullmatch(
                    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", stripped
                ):
            number = float(stripped)
            if math.isfinite(number):
                converted, conversion = number, "string_to_number"
        elif "string" not in declared and "array" in declared:
            parsed = _structured_string(stripped, list)
            if parsed is not None:
                converted, conversion = parsed, "string_to_array"
        elif "string" not in declared and "object" in declared:
            parsed = _structured_string(stripped, dict)
            if parsed is not None:
                converted, conversion = parsed, "string_to_object"
        if conversion is not None:
            normalized[key] = converted
            events.append({"argument": key, "conversion": conversion})
    return normalized, tuple(events)


def _validation_error(schema: dict[str, Any], arguments: dict[str, Any]) -> str | None:
    try:
        _validate_arguments(schema, arguments)
        return None
    except (TypeError, ValueError) as exc:
        return str(exc)[:300]


@dataclass(frozen=True)
class SimulatedInvocation:
    result: dict[str, Any]
    raw_arguments: dict[str, Any]
    executed_arguments: dict[str, Any]
    adapter_events: tuple[dict[str, str], ...]
    raw_arguments_valid: bool
    executed_arguments_valid: bool
    raw_validation_error: str | None
    executed_validation_error: str | None

    @property
    def adapter_rescued(self) -> bool:
        return (not self.raw_arguments_valid and self.executed_arguments_valid
                and bool(self.adapter_events))


class AgentAlignSimulator:
    def __init__(self, repo: str | Path):
        directory = verify_simulator_source(repo)
        package = "_pinned_agentalign_synthetic_tools"
        spec = importlib.util.spec_from_file_location(
            package, directory / "__init__.py",
            submodule_search_locations=[str(directory)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load pinned AgentAlign synthetic tools.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
        self.modules = [value for key, value in sys.modules.items()
                        if key.startswith(package + ".")]
        self.functions = dict(module.AVAILABLE_TOOLS)

    def _deterministic_bindings(self, nonce: str) -> None:
        """Keep simulated feedback stable across paired intervention arms."""
        seed = int.from_bytes(hashlib.sha256(nonce.encode()).digest()[:8], "big")
        rng = random.Random(seed)
        timestamp = 1735689600 + seed % 86400  # fixed synthetic day + task offset
        clock = SimpleNamespace(
            time=lambda: float(timestamp), sleep=lambda _seconds: None,
            gmtime=lambda seconds=None: time.gmtime(timestamp if seconds is None else seconds),
            localtime=lambda seconds=None: time.gmtime(timestamp if seconds is None else seconds),
            strftime=time.strftime,
        )
        identifiers = SimpleNamespace(uuid4=lambda: uuid.UUID(int=rng.getrandbits(128)))
        for module in self.modules:
            module.random = rng
            module.time = clock
            module.uuid = identifiers

    def invoke(
        self, name: str, arguments: dict[str, Any], exposed: tuple[dict[str, Any], ...],
        *, nonce: str,
    ) -> dict[str, Any]:
        return self.invoke_with_trace(name, arguments, exposed, nonce=nonce).result

    def invoke_with_trace(
        self, name: str, arguments: dict[str, Any], exposed: tuple[dict[str, Any], ...],
        *, nonce: str,
    ) -> SimulatedInvocation:
        schemas = {item["function"]["name"]: item for item in exposed}
        if name not in schemas or name not in self.functions:
            return SimulatedInvocation(
                {"error": "unavailable_tool", "tool": name}, arguments, arguments, (),
                False, False, "Tool is unavailable.", "Tool is unavailable.",
            )
        schema = schemas[name]
        raw_error = _validation_error(schema, arguments)
        executed, events = normalize_arguments(schema, arguments)
        executed_error = _validation_error(schema, executed)
        if executed_error is not None:
            result = {"error": "invalid_tool_arguments", "tool": name,
                      "type": "ValueError", "detail": executed_error}
            return SimulatedInvocation(
                result, arguments, executed, events, raw_error is None, False,
                raw_error, executed_error,
            )
        try:
            self._deterministic_bindings(nonce)
            result = self.functions[name](**executed)
            # Fail closed on non-serializable or unbounded tool output.
            encoded = json.dumps(result, ensure_ascii=True, default=str)
            if len(encoded) > 20000:
                result = {"error": "tool_output_too_large", "tool": name}
                return SimulatedInvocation(
                    result, arguments, executed, events, raw_error is None, True,
                    raw_error, None,
                )
            decoded = json.loads(encoded)
            result = decoded if isinstance(decoded, dict) else {"result": decoded}
        except Exception as exc:
            result = {"error": "synthetic_tool_exception", "tool": name,
                      "type": type(exc).__name__, "detail": str(exc)[:300]}
        return SimulatedInvocation(
            result, arguments, executed, events, raw_error is None, True,
            raw_error, None,
        )
