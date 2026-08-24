"""Small, dependency-free `/action_cmd` payload codec.

This module intentionally uses only the standard library so the motion/handheld
control path can start even when optional LLM, FastMCP, or cloud dependencies
are unavailable.
"""

from __future__ import annotations

import json
import uuid
from typing import Any


def build_action_cmd(
    tool_name: str,
    arguments: Any,
    *,
    request_id: str | None = None,
    source: str | None = None,
) -> str:
    """Build an action command, optionally with correlation metadata.

    Existing in-process callers can keep emitting the legacy two-field payload.
    Remote gateways add ``request_id`` and ``source`` so executors can publish a
    correlated status without changing the tool arguments seen by ROS nodes.
    """
    if isinstance(arguments, str):
        arguments = json.loads(arguments or "{}")
    if not isinstance(tool_name, str) or not tool_name or not isinstance(arguments, dict):
        raise ValueError("action_cmd requires a tool name and object arguments")
    command = {"name": tool_name, "arguments": arguments}
    if request_id is not None:
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be a non-empty string")
        command["request_id"] = request_id.strip()
    if source is not None:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")
        command["source"] = source.strip()
    return json.dumps(command, ensure_ascii=False, separators=(",", ":"))


def new_action_request_id() -> str:
    """Return an opaque identifier suitable for cross-process correlation."""
    return uuid.uuid4().hex


def parse_action_request(payload: Any) -> dict[str, Any] | None:
    """Decode a complete action envelope while preserving safe metadata."""
    try:
        command = json.loads(payload) if isinstance(payload, str) else payload
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(command, dict) or not isinstance(command.get("name"), str):
        return None
    name = command["name"].strip()
    if not name:
        return None
    arguments = command.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None

    decoded = {"name": name, "arguments": arguments}
    for field in ("request_id", "source"):
        value = command.get(field)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                return None
            decoded[field] = value.strip()
    return decoded


def parse_action_cmd(payload: Any) -> tuple[str, dict[str, Any]] | None:
    """Decode an action command; return ``None`` for malformed messages."""
    command = parse_action_request(payload)
    if command is None:
        return None
    return command["name"], command["arguments"]
