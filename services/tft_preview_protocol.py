"""ROS-independent request/result codec for the TFT preview owner node."""

from __future__ import annotations

import base64
import json
import uuid
from typing import Any

from services.tft_preview_server import PreviewResult


TFT_PREVIEW_READY_TOPIC = "tft_preview_ready"
TFT_PREVIEW_REQUEST_TOPIC = "/tft_preview_request"
TFT_PREVIEW_RESULT_TOPIC = "/tft_preview_result"
CAMERA_PREVIEW = "camera_preview"


def encode_camera_preview_request(
    *,
    duration_ms: int,
    hold_ms: int,
    fps: int,
    request_id: str | None = None,
) -> tuple[str, str]:
    request_id = request_id or uuid.uuid4().hex
    payload = {
        "request_id": request_id,
        "kind": CAMERA_PREVIEW,
        "duration_ms": max(100, int(duration_ms)),
        "hold_ms": max(0, int(hold_ms)),
        "fps": min(30, max(1, int(fps))),
    }
    return request_id, json.dumps(payload, separators=(",", ":"))


def decode_preview_request(raw: Any) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    request_id = value.get("request_id")
    if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
        return None
    if value.get("kind") != CAMERA_PREVIEW:
        return None
    try:
        return {
            "request_id": request_id,
            "kind": CAMERA_PREVIEW,
            "duration_ms": max(100, int(value.get("duration_ms", 1500))),
            "hold_ms": max(0, int(value.get("hold_ms", 3000))),
            "fps": min(30, max(1, int(value.get("fps", 10)))),
        }
    except (TypeError, ValueError):
        return None


def encode_preview_result(request_id: str, result: PreviewResult) -> str:
    payload = {
        "request_id": str(request_id),
        "last_frame_base64": (
            base64.b64encode(result.last_frame).decode("ascii")
            if result.last_frame
            else ""
        ),
        "source_frames": result.source_frames,
        "encoded_frames": result.encoded_frames,
        "sent_frames": result.sent_frames,
        "no_new_frame_skips": result.no_new_frame_skips,
        "backpressure_drops": result.backpressure_drops,
        "encode_drops": result.encode_drops,
        "total_bytes": result.total_bytes,
        "elapsed_seconds": result.elapsed_seconds,
        "connected": result.connected,
        "busy": result.busy,
        "error": result.error,
    }
    return json.dumps(payload, separators=(",", ":"))


def decode_preview_result(raw: Any) -> tuple[str, PreviewResult] | None:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    request_id = value.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        return None
    encoded_frame = value.get("last_frame_base64", "")
    try:
        last_frame = base64.b64decode(encoded_frame, validate=True) if encoded_frame else None
    except (TypeError, ValueError):
        return None
    try:
        result = PreviewResult(
            last_frame=last_frame,
            source_frames=int(value.get("source_frames", 0)),
            encoded_frames=int(value.get("encoded_frames", 0)),
            sent_frames=int(value.get("sent_frames", 0)),
            no_new_frame_skips=int(value.get("no_new_frame_skips", 0)),
            backpressure_drops=int(value.get("backpressure_drops", 0)),
            encode_drops=int(value.get("encode_drops", 0)),
            total_bytes=int(value.get("total_bytes", 0)),
            elapsed_seconds=float(value.get("elapsed_seconds", 0.0)),
            connected=bool(value.get("connected", False)),
            busy=bool(value.get("busy", False)),
            error=str(value.get("error", "")),
        )
    except (TypeError, ValueError):
        return None
    return request_id, result


__all__ = [
    "CAMERA_PREVIEW",
    "TFT_PREVIEW_READY_TOPIC",
    "TFT_PREVIEW_REQUEST_TOPIC",
    "TFT_PREVIEW_RESULT_TOPIC",
    "decode_preview_request",
    "decode_preview_result",
    "encode_camera_preview_request",
    "encode_preview_result",
]
