"""ROS-independent protocol helpers for game-mode coordination."""

from __future__ import annotations

import json
import struct
from typing import Any


GAME_MODE_REQUEST_TOPIC = "/game_mode_request"
GAME_MODE_STATE_TOPIC = "/game_mode_state"
GAME_FRAME_TOPIC = "/game_frame"
GAME_SURFACE_READY = "game_surface_ready"
_FRAME_HEADER = struct.Struct("<HHI")


def encode_game_request(request: str, **fields: Any) -> str:
    return json.dumps(
        {"request": str(request), **fields},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def decode_game_message(raw: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def game_mode_from_message(raw: str) -> str | None:
    value = decode_game_message(raw)
    mode = value.get("mode") if value is not None else None
    return mode if isinstance(mode, str) else None


def game_is_active(raw: str) -> bool:
    mode = game_mode_from_message(raw)
    return mode is not None and mode != "robot"


def encode_game_frame(raw: bytes, width: int, height: int, pitch: int) -> bytes:
    return _FRAME_HEADER.pack(int(width), int(height), int(pitch)) + bytes(raw)


def decode_game_frame(packet: bytes) -> tuple[bytes, int, int, int] | None:
    data = bytes(packet)
    if len(data) < _FRAME_HEADER.size:
        return None
    width, height, pitch = _FRAME_HEADER.unpack_from(data)
    raw = data[_FRAME_HEADER.size:]
    if width <= 0 or height <= 0 or pitch < width * 4 or len(raw) < pitch * height:
        return None
    return raw, width, height, pitch


__all__ = [
    "GAME_FRAME_TOPIC",
    "GAME_MODE_REQUEST_TOPIC",
    "GAME_MODE_STATE_TOPIC",
    "GAME_SURFACE_READY",
    "decode_game_message",
    "decode_game_frame",
    "encode_game_frame",
    "encode_game_request",
    "game_is_active",
    "game_mode_from_message",
]
