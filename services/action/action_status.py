"""Structured status protocol for correlated robot action requests."""

from __future__ import annotations

import json
from typing import Any


ACTION_STATUS_TOPIC = "/action_status"
TERMINAL_ACTION_STATUSES = frozenset({"completed", "rejected", "failed", "interrupted"})
VALID_ACTION_STATUSES = frozenset({"accepted", "running", *TERMINAL_ACTION_STATUSES})


def build_action_status(
    request_id: str,
    name: str,
    status: str,
    *,
    source: str = "robot",
    detail: str = "",
) -> str:
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must be a non-empty string")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    if status not in VALID_ACTION_STATUSES:
        raise ValueError(f"invalid action status: {status}")
    payload = {
        "request_id": request_id.strip(),
        "name": name.strip(),
        "status": status,
        "source": str(source or "robot"),
    }
    if detail:
        payload["detail"] = str(detail)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def parse_action_status(payload: Any) -> dict[str, str] | None:
    try:
        data = json.loads(payload) if isinstance(payload, str) else payload
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    request_id = data.get("request_id")
    name = data.get("name")
    status = data.get("status")
    if (
        not isinstance(request_id, str)
        or not request_id.strip()
        or not isinstance(name, str)
        or not name.strip()
        or status not in VALID_ACTION_STATUSES
    ):
        return None
    return {
        "request_id": request_id.strip(),
        "name": name.strip(),
        "status": status,
        "source": str(data.get("source") or "robot"),
        "detail": str(data.get("detail") or ""),
    }
