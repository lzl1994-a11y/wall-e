"""Synchronous ROS client used by dialogue workers to request a TFT preview."""

from __future__ import annotations

import threading
import time
from typing import Any

try:
    from std_msgs.msg import String
except ImportError:  # pragma: no cover - ROS is unavailable in unit tests
    String = None

from services.tft_preview_protocol import (
    TFT_PREVIEW_REQUEST_TOPIC,
    TFT_PREVIEW_RESULT_TOPIC,
    decode_preview_result,
    encode_camera_preview_request,
)
from services.tft_preview_server import PreviewResult


class TftPreviewClient:
    """Publish one request and wait for its correlated result off the ROS thread."""

    def __init__(self, node: Any, *, logger: Any = None) -> None:
        self._condition = threading.Condition()
        self._results: dict[str, PreviewResult] = {}
        self._pending: set[str] = set()
        self._closed = False
        self._logger = logger
        self._publisher = None
        self._subscription = None
        if String is not None:
            self._publisher = node.create_publisher(String, TFT_PREVIEW_REQUEST_TOPIC, 10)
            self._subscription = node.create_subscription(
                String, TFT_PREVIEW_RESULT_TOPIC, self._on_result, 10
            )

    def _on_result(self, message: Any) -> None:
        decoded = decode_preview_result(getattr(message, "data", ""))
        if decoded is None:
            return
        request_id, result = decoded
        with self._condition:
            if request_id not in self._pending:
                return
            self._results[request_id] = result
            self._condition.notify_all()

    def send_camera_preview(
        self,
        *,
        duration_ms: int,
        hold_ms: int,
        fps: int,
        timeout: float | None = None,
    ) -> PreviewResult:
        if self._publisher is None or String is None:
            return PreviewResult(error="tft_preview_ros_unavailable")
        request_id, payload = encode_camera_preview_request(
            duration_ms=duration_ms,
            hold_ms=hold_ms,
            fps=fps,
        )
        wait_seconds = (
            max(1.0, float(timeout))
            if timeout is not None
            else 20.0 + (max(100, int(duration_ms)) + max(0, int(hold_ms))) / 1000.0
        )
        deadline = time.monotonic() + wait_seconds
        with self._condition:
            if self._closed:
                return PreviewResult(error="tft_preview_client_closed")
            self._pending.add(request_id)
        self._publisher.publish(String(data=payload))
        with self._condition:
            while not self._closed:
                result = self._results.pop(request_id, None)
                if result is not None:
                    self._pending.discard(request_id)
                    return result
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            self._pending.discard(request_id)
            self._results.pop(request_id, None)
        self._log("warning", f"TFT preview request timed out: {request_id}")
        return PreviewResult(error="tft_preview_request_timeout")

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def _log(self, level: str, message: str) -> None:
        callback = getattr(self._logger, level, None)
        if callback is not None:
            callback(message)


__all__ = ["TftPreviewClient"]
