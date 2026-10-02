"""ROS topic RPC shared by the web clients of the sole serial owner.

Protocol clients retain their own topics and codecs.  Discovery, request
correlation and executor lifetime are implemented once; no client opens USB.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from typing import Any

try:
    from services.hardware.esp32_netcfg import NetworkConfigError
except ImportError:
    from esp32_netcfg import NetworkConfigError

REQUEST_TOPIC = "esp32_netcfg_request"
RESPONSE_TOPIC = "esp32_netcfg_response"
WEB_RPC_TIMEOUT_SECONDS = 90.0
ROS_DISCOVERY_TIMEOUT_SECONDS = 5.0

_context_lock = threading.Lock()
_context_users = 0
_context_owned = False


class _SerialOwnerRpcClient:
    """Private transport used by the two serial-owner protocol clients."""

    _request_topic = REQUEST_TOPIC
    _response_topic = RESPONSE_TOPIC
    _node_name = "walle_netcfg_web_client"
    _error_type = NetworkConfigError

    def __init__(self, *, monotonic=time.monotonic,
                 discovery_timeout_seconds=ROS_DISCOVERY_TIMEOUT_SECONDS):
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from std_msgs.msg import String
        except ImportError as exc:
            raise self._error_type("ROS 串口服务不可用；请通过主程序启动配置网页") from exc
        self._rclpy = rclpy
        self._String = String
        self._closed = threading.Event()
        self._lock = threading.RLock()
        self._pending: dict[str, tuple[threading.Event, dict[str, Any]]] = {}
        self._failure = ""
        self._monotonic = monotonic
        self._discovery_timeout_seconds = max(0.1, float(discovery_timeout_seconds))
        self._context_registered = False
        global _context_users, _context_owned
        with _context_lock:
            if not rclpy.ok():
                rclpy.init(args=None)
                _context_owned = True
            _context_users += 1
            self._context_registered = True
        try:
            self._node = Node(self._node_name)
            self._publisher = self._node.create_publisher(String, self._request_topic, 10)
            self._response_subscription = self._node.create_subscription(
                String, self._response_topic, self._on_response, 10
            )
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            self._thread = threading.Thread(
                target=self._spin, name=self._node_name, daemon=True
            )
            self._thread.start()
        except Exception as exc:
            self.close()
            raise self._error_type("ROS 串口配置服务初始化失败") from exc

    def _fail_pending(self, message):
        with self._lock:
            self._failure = message
            for event, result in self._pending.values():
                result.update(ok=False, error=message)
                event.set()

    def _spin(self):
        try:
            while not self._closed.is_set() and self._rclpy.ok():
                self._executor.spin_once(timeout_sec=0.2)
            if not self._closed.is_set():
                self._fail_pending("ROS 串口服务上下文已停止")
        except Exception:
            if not self._closed.is_set():
                # Do not log message contents: NETCFG requests contain secrets.
                self._fail_pending("串口 RPC 接收线程异常，服务已停止")
                self._node.get_logger().error("串口 RPC 接收线程异常，服务已停止")

    def _on_response(self, message):
        try:
            body = json.loads(message.data)
        except (AttributeError, TypeError, json.JSONDecodeError):
            return
        if not isinstance(body, dict) or not isinstance(body.get("request_id"), str):
            return
        with self._lock:
            pending = self._pending.get(body["request_id"])
            if pending is not None:
                event, result = pending
                result.update(body)
                event.set()

    def _wait_for_serial_owner(self, deadline):
        discovery_deadline = min(deadline, self._monotonic() + self._discovery_timeout_seconds)
        while not self._closed.is_set() and self._monotonic() < discovery_deadline:
            if getattr(self, "_failure", ""):
                return False
            try:
                subscriber = self._node.count_subscribers(self._request_topic) >= 1
                publisher = self._node.count_publishers(self._response_topic) >= 1
            except RuntimeError:
                return False
            if subscriber and publisher:
                return True
            self._closed.wait(min(0.05, max(0, discovery_deadline - self._monotonic())))
        return False

    def _exchange(self, body, timeout):
        request_id = secrets.token_urlsafe(18)
        event = threading.Event()
        result = {}
        deadline = self._monotonic() + timeout
        with self._lock:
            if self._closed.is_set() or self._failure:
                raise self._error_type(self._failure or "ROS 串口服务已停止")
            self._pending[request_id] = (event, result)
        try:
            if not self._wait_for_serial_owner(deadline):
                raise self._error_type(self._failure or "未发现 serial_ros_node 的串口 RPC 端点")
            message = self._String()
            message.data = json.dumps(
                {**body, "request_id": request_id}, ensure_ascii=False, separators=(",", ":")
            )
            self._publisher.publish(message)
            remaining = deadline - self._monotonic()
            if remaining <= 0 or not event.wait(remaining):
                raise self._error_type("等待串口配置服务超时；请确认下位机已连接")
            return result
        finally:
            with self._lock:
                self._pending.pop(request_id, None)

    def close(self):
        if self._closed.is_set():
            return
        self._closed.set()
        self._fail_pending("ROS 串口服务已停止")
        thread = getattr(self, "_thread", None)
        if thread is not None:
            thread.join(timeout=1.0)
        executor = getattr(self, "_executor", None)
        node = getattr(self, "_node", None)
        try:
            if executor is not None:
                if node is not None:
                    executor.remove_node(node)
                executor.shutdown(timeout_sec=1.0)
            if node is not None:
                node.destroy_node()
        finally:
            global _context_users, _context_owned
            with _context_lock:
                if self._context_registered:
                    self._context_registered = False
                    _context_users -= 1
                    if _context_users == 0 and _context_owned:
                        _context_owned = False
                        if self._rclpy.ok():
                            self._rclpy.shutdown()


class Esp32NetworkRpcClient(_SerialOwnerRpcClient):
    """Existing NETCFG API, backed by the shared ROS request transport."""

    def _call(self, operation: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = {"operation": operation}
        if payload is not None:
            body["payload"] = payload
        result = self._exchange(body, WEB_RPC_TIMEOUT_SECONDS)
        if not result.get("ok"):
            raise NetworkConfigError(str(result.get("error") or "设备网络配置失败"))
        data = result.get("data")
        return data if isinstance(data, dict) else {}

    def save_and_apply(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._call("save_and_apply", payload)

    def query(self) -> dict[str, Any]:
        return self._call("query")
