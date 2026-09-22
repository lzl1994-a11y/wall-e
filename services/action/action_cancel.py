"""Protocol for targeted cancellation of an accepted action request."""

from __future__ import annotations

import json
from typing import Any


ACTION_CANCEL_TOPIC = "/action_cancel"


def build_action_cancel(
    request_id: str,
    name: str,
    *,
    reason: str,
    replacement_request_id: str = "",
) -> str:
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must be a non-empty string")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    payload = {
        "request_id": request_id.strip(),
        "name": name.strip(),
        "reason": reason.strip(),
        "source": "action_coordinator",
    }
    if replacement_request_id:
        payload["replacement_request_id"] = replacement_request_id.strip()
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def parse_action_cancel(payload: Any) -> dict[str, str] | None:
    try:
        data = json.loads(payload) if isinstance(payload, str) else payload
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    values = {}
    for field in ("request_id", "name", "reason"):
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            return None
        values[field] = value.strip()
    values["source"] = str(data.get("source") or "action_coordinator")
    values["replacement_request_id"] = str(
        data.get("replacement_request_id") or ""
    )
    return values


__all__ = ["ACTION_CANCEL_TOPIC", "build_action_cancel", "parse_action_cancel"]
