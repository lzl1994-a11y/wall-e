"""Ephemeral ROS topic RPC between the config web process and serial owner.

Passwords appear only in the one request message sent over the local ROS graph;
they are never persisted, logged, or sent back in a response.  This avoids a
second process opening the ESP32 USB serial port.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from typing import Any

try:
    from services.esp32_netcfg import NetworkConfigError
except ImportError:  # Supports: python services/web_server.py
    from esp32_netcfg import NetworkConfigError

REQUEST_TOPIC = "esp32_netcfg_request"
RESPONSE_TOPIC = "esp32_netcfg_response"
WEB_RPC_TIMEOUT_SECONDS = 90.0
ROS_DISCOVERY_TIMEOUT_SECONDS = 5.0


class Esp32NetworkRpcClient:
    """Blocking, thread-safe client used by HTTP handlers in config_web."""

    def __init__(
        self,
        *,
        monotonic=time.monotonic,
        discovery_timeout_seconds: float = ROS_DISCOVERY_TIMEOUT_SECONDS,
    ) -> None:
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from std_msgs.msg import String
        except ImportError as exc:
            raise NetworkConfigError("ROS 串口服务不可用；请通过主程序启动配置网页") from exc
        self._rclpy = rclpy
        self._String = String
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy_context = True
        else:
            self._owns_rclpy_context = False
        self._node = Node("walle_netcfg_web_client")
        self._publisher = self._node.create_publisher(String, REQUEST_TOPIC, 10)
        self._response_subscription = self._node.create_subscription(
            String, RESPONSE_TOPIC, self._on_response, 10
        )
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._pending: dict[str, tuple[threading.Event, dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._monotonic = monotonic
        self._discovery_timeout_seconds = discovery_timeout_seconds
        self._thread = threading.Thread(target=self._spin, name="netcfg-web-rpc", daemon=True)
        self._thread.start()

    def _spin(self) -> None:
        while not self._closed.is_set() and self._rclpy.ok():
            try:
                self._executor.spin_once(timeout_sec=0.2)
            except Exception:
                # A malformed DDS message/callback must not permanently stop
                # the RPC receive loop and turn all later requests into timeouts.
                if self._closed.is_set():
                    return
                self._closed.wait(0.05)

    def _on_response(self, message: Any) -> None:
        try:
            body = json.loads(message.data)
            if not isinstance(body, dict):
                return
            request_id = body.get("request_id")
        except (AttributeError, TypeError, json.JSONDecodeError):
            return
        with self._lock:
            pending = self._pending.get(request_id)
        if pending is None:
            return
        event, result = pending
        result.update(body)
        event.set()

    def _wait_for_serial_owner(self, deadline: float) -> bool:
        """Wait for ROS discovery before publishing a volatile one-shot request."""
        discovery_deadline = min(deadline, self._monotonic() + self._discovery_timeout_seconds)
        while not self._closed.is_set() and self._monotonic() < discovery_deadline:
            try:
                # ROS 2 Humble subscriptions do not expose
                # ``get_publisher_count()``. Query the graph through Node APIs,
                # which are supported across the deployed Humble runtime.
                has_request_subscriber = (
                    self._node.count_subscribers(REQUEST_TOPIC) >= 1
                )
                has_response_publisher = (
                    self._node.count_publishers(RESPONSE_TOPIC) >= 1
                )
            except RuntimeError:
                return False
            if has_request_subscriber and has_response_publisher:
                return True
            self._closed.wait(min(0.05, max(0.0, discovery_deadline - self._monotonic())))
        return False

    def _call(self, operation: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        request_id = secrets.token_urlsafe(18)
        event = threading.Event()
        result: dict[str, Any] = {}
        deadline = self._monotonic() + WEB_RPC_TIMEOUT_SECONDS
        with self._lock:
            if self._closed.is_set():
                raise NetworkConfigError("ROS 串口服务已停止")
            self._pending[request_id] = (event, result)
        try:
            if not self._wait_for_serial_owner(deadline):
                raise NetworkConfigError(
                    "未发现 serial_ros_node 的 ESP32 NETCFG RPC 端点"
                )
            # Do not log this JSON: save_and_apply contains Wi-Fi passwords.
            body = {"request_id": request_id, "operation": operation}
            if payload is not None:
                body["payload"] = payload
            message = self._String()
            message.data = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
            self._publisher.publish(message)
            remaining = deadline - self._monotonic()
            if remaining <= 0 or not event.wait(remaining):
                raise NetworkConfigError("等待串口配置服务超时；请确认 serial_ros_node 正在运行")
            if not result.get("ok"):
                raise NetworkConfigError(str(result.get("error") or "设备网络配置失败"))
            data = result.get("data")
            return data if isinstance(data, dict) else {}
        finally:
            with self._lock:
                self._pending.pop(request_id, None)

    def save_and_apply(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._call("save_and_apply", payload)

    def query(self) -> dict[str, Any]:
        return self._call("query")

    def close(self) -> None:
        self._closed.set()
        self._thread.join(timeout=1)
        try:
            self._executor.remove_node(self._node)
            self._node.destroy_node()
        except Exception:
            pass
        if self._owns_rclpy_context:
            try:
                self._rclpy.shutdown()
            except Exception:
                pass
