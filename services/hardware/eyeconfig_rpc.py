"""ROS contract and codec for the live WALL-E eye configuration page.

The web process never opens the ESP32 port.  It sends one validated eye
command to ``serial_ros_node`` over ROS, and the serial owner returns the
firmware's ``EYE:STATE``, ``EYE:OK`` or ``EYE:ERR`` line over a correlated
response topic.
"""

from __future__ import annotations

import json
import math
import re
import secrets
import threading
import time
from typing import Any


EYE_REQUEST_TOPIC = "eyeconfig_request"
EYE_RESPONSE_TOPIC = "eyeconfig_response"
EYE_STATUS_TOPIC = "eye_status"
EYE_RPC_TIMEOUT_SECONDS = 8.0
EYE_DISCOVERY_TIMEOUT_SECONDS = 3.0

_COLOR_RE = r"[0-9A-Fa-f]{6}"
_SIGNED_INTEGER_RE = r"-?\d{1,3}"


class EyeConfigError(RuntimeError):
    """A live eye command could not be sent or was rejected by the device."""

    def __init__(
        self,
        message: str,
        *,
        event: dict[str, Any] | None = None,
        kind: str = "unavailable",
    ) -> None:
        super().__init__(message)
        self.event = event
        self.kind = kind


def _number_in_range(value: str, minimum: float, maximum: float) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and minimum <= number <= maximum


def validate_eye_command(command: Any) -> str:
    """Validate and normalize one command accepted by the eye firmware.

    The web endpoint deliberately accepts only the documented eye commands;
    arbitrary strings must not become a general-purpose serial injection API.
    """
    if not isinstance(command, str):
        raise EyeConfigError("眼睛命令必须是字符串", kind="invalid")
    command = command.strip()
    if not command or len(command) > 160 or "\n" in command or "\r" in command:
        raise EyeConfigError("眼睛命令格式无效", kind="invalid")
    if command == "eyeconfig:query":
        return command

    match = re.fullmatch(r"eyeconfig:(color|ringColor|dotColor)=([0-9A-Fa-f]{6})", command)
    if match:
        return f"eyeconfig:{match.group(1)}={match.group(2).upper()}"

    match = re.fullmatch(
        r"eyeconfig:(brightness|ringBrightness|dotBrightness)=([^\s]+)",
        command,
    )
    if match and _number_in_range(match.group(2), 0.0, 1.0):
        value = float(match.group(2))
        return f"eyeconfig:{match.group(1)}={value:g}"

    match = re.fullmatch(r"eyeconfig:(scale)=([^\s]+)", command)
    if match and _number_in_range(match.group(2), 0.1, 3.0):
        return f"eyeconfig:scale={float(match.group(2)):g}"

    integer_ranges = {
        "glow": (0, 120),
        "breathMs": (100, 60000),
        "blinkMs": (100, 60000),
        "dots": (0, 256),
    }
    for key, (minimum, maximum) in integer_ranges.items():
        match = re.fullmatch(rf"eyeconfig:{key}=(-?\d+)", command)
        if match and minimum <= int(match.group(1)) <= maximum:
            return f"eyeconfig:{key}={int(match.group(1))}"

    # The requested command list names the auto-blink control but omits its
    # setter spelling; keep the page's switch explicit and symmetric with the
    # existing ring flag.
    match = re.fullmatch(r"eyeconfig:(autoBlink|ring)=([01])", command)
    if match:
        return command

    match = re.fullmatch(r"eyeconfig:mood=(dot|flame|heart)", command)
    if match:
        return command

    match = re.fullmatch(
        rf"eyeaction:look:x=({_SIGNED_INTEGER_RE}),y=({_SIGNED_INTEGER_RE})",
        command,
    )
    if match and all(-100 <= int(value) <= 100 for value in match.groups()):
        return f"eyeaction:look:x={int(match.group(1))},y={int(match.group(2))}"

    if command in {
        "eyeaction:mood:dot",
        "eyeaction:mood:flame",
        "eyeaction:mood:heart",
        "eyeaction:blink",
        "eyeaction:zoom",
        "eyeaction:dot",
        "eyeaction:flame",
        "eyeaction:heart",
    }:
        return command

    raise EyeConfigError(f"不支持的眼睛命令: {command}", kind="invalid")


_STATE_KEY_ALIASES = {
    "color": "color",
    "centercolor": "color",
    "ringcolor": "ringColor",
    "dotcolor": "dotColor",
    "brightness": "brightness",
    "centerbrightness": "brightness",
    "ringbrightness": "ringBrightness",
    "dotbrightness": "dotBrightness",
    "scale": "scale",
    "centerscale": "scale",
    "glow": "glow",
    "glowsize": "glow",
    "breathms": "breathMs",
    "breath": "breathMs",
    "blinkms": "blinkMs",
    "blink": "blinkMs",
    "autoblink": "autoBlink",
    "autoblinkenabled": "autoBlink",
    "ring": "ring",
    "ringenabled": "ring",
    "dots": "dots",
    "dotcount": "dots",
    "mood": "mood",
    "x": "lookX",
    "lookx": "lookX",
    "pupilx": "lookX",
    "y": "lookY",
    "looky": "lookY",
    "pupily": "lookY",
}

_BOOLEAN_STATE_KEYS = {"autoBlink", "ring"}
_INTEGER_STATE_KEYS = {"glow", "breathMs", "blinkMs", "dots", "lookX", "lookY"}
_FLOAT_STATE_KEYS = {"brightness", "ringBrightness", "dotBrightness", "scale"}
_KNOWN_MOODS = {"dot", "flame", "heart"}


def _canonical_state_key(key: Any) -> str | None:
    normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
    return _STATE_KEY_ALIASES.get(normalized)


def _coerce_state_value(key: str, value: Any) -> Any:
    if key in _BOOLEAN_STATE_KEYS:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "on", "yes", "enabled"}:
            return True
        if text in {"0", "false", "off", "no", "disabled"}:
            return False
        return None
    if key in _INTEGER_STATE_KEYS:
        try:
            return int(float(value))
        except (TypeError, ValueError, OverflowError):
            return None
    if key in _FLOAT_STATE_KEYS:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None
    if key in {"color", "ringColor", "dotColor"}:
        text = str(value).strip().lstrip("#")
        return text.upper() if re.fullmatch(_COLOR_RE, text) else None
    if key == "mood":
        text = str(value).strip().lower()
        return text if text in _KNOWN_MOODS else None
    return str(value)


def normalize_eye_state(fields: Any) -> dict[str, Any]:
    """Return only known, canonical eye state fields from a device payload."""
    if not isinstance(fields, dict):
        return {}
    result: dict[str, Any] = {}
    for raw_key, raw_value in fields.items():
        key = _canonical_state_key(raw_key)
        if key is None:
            continue
        value = _coerce_state_value(key, raw_value)
        if value is not None:
            result[key] = value
    return result


def _parse_state_payload(payload: str) -> dict[str, Any]:
    payload = payload.strip().lstrip(":| ")
    if not payload:
        return {}
    try:
        decoded = json.loads(payload)
    except (TypeError, json.JSONDecodeError):
        decoded = None
    if isinstance(decoded, dict):
        if isinstance(decoded.get("state"), dict):
            decoded = decoded["state"]
        elif isinstance(decoded.get("config"), dict):
            decoded = decoded["config"]
        return normalize_eye_state(decoded)

    pairs: dict[str, str] = {}
    for match in re.finditer(
        r"([A-Za-z][A-Za-z0-9_-]*)\s*[=:]\s*([^\s,;|]+)",
        payload,
    ):
        pairs[match.group(1)] = match.group(2)
    return normalize_eye_state(pairs)


def parse_eye_response(raw: Any) -> dict[str, Any] | None:
    """Parse one firmware response line into a JSON-safe event."""
    if not isinstance(raw, str):
        return None
    line = raw.strip().strip("\x00")
    if not line:
        return None
    if line == "EYE:STATE" or line.startswith(("EYE:STATE:", "EYE:STATE ", "EYE:STATE|")):
        payload = line[len("EYE:STATE") :]
        return {
            "kind": "state",
            "raw": line,
            "fields": _parse_state_payload(payload),
            "message": "",
            "received_at": time.time(),
        }
    if line == "EYE:OK" or line.startswith(("EYE:OK:", "EYE:OK ", "EYE:OK|")):
        message = line[len("EYE:OK") :].lstrip(":| ")
        fields = _parse_state_payload(message)
        return {
            "kind": "ok",
            "raw": line,
            "fields": fields,
            "message": message,
            "received_at": time.time(),
        }
    if line == "EYE:ERR" or line.startswith(("EYE:ERR:", "EYE:ERR ", "EYE:ERR|")):
        message = line[len("EYE:ERR") :].lstrip(":| ") or "设备拒绝了眼睛配置"
        return {
            "kind": "err",
            "raw": line,
            "fields": {},
            "message": message,
            "received_at": time.time(),
        }
    return None


def encode_eye_request(request_id: str, command: str, expected: str) -> str:
    return json.dumps(
        {
            "request_id": str(request_id),
            "command": validate_eye_command(command),
            "expected": expected,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


class EyeConfigRpcClient:
    """Blocking ROS client used by the HTTP handlers in the config web app."""

    def __init__(
        self,
        *,
        monotonic=time.monotonic,
        discovery_timeout_seconds: float = EYE_DISCOVERY_TIMEOUT_SECONDS,
        rpc_timeout_seconds: float = EYE_RPC_TIMEOUT_SECONDS,
    ) -> None:
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from std_msgs.msg import String
        except ImportError as exc:
            raise EyeConfigError("ROS 串口服务不可用；请通过主程序启动配置网页") from exc

        self._rclpy = rclpy
        self._String = String
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy_context = True
        else:
            self._owns_rclpy_context = False
        self._node = Node("walle_eye_config_web_client")
        self._publisher = self._node.create_publisher(String, EYE_REQUEST_TOPIC, 10)
        self._response_subscription = self._node.create_subscription(
            String, EYE_RESPONSE_TOPIC, self._on_response, 10
        )
        self._status_subscription = self._node.create_subscription(
            String, EYE_STATUS_TOPIC, self._on_status, 10
        )
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._pending: dict[str, tuple[threading.Event, dict[str, Any]]] = {}
        self._lock = threading.RLock()
        self._status: dict[str, Any] | None = None
        self._closed = threading.Event()
        self._monotonic = monotonic
        self._discovery_timeout_seconds = max(0.1, float(discovery_timeout_seconds))
        self._rpc_timeout_seconds = max(0.1, float(rpc_timeout_seconds))
        self._thread = threading.Thread(
            target=self._spin,
            name="eye-config-web-rpc",
            daemon=True,
        )
        self._thread.start()

    def _spin(self) -> None:
        while not self._closed.is_set() and self._rclpy.ok():
            try:
                self._executor.spin_once(timeout_sec=0.2)
            except Exception:
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

    def _on_status(self, message: Any) -> None:
        try:
            body = json.loads(message.data)
        except (AttributeError, TypeError, json.JSONDecodeError):
            event = parse_eye_response(getattr(message, "data", ""))
            body = {"event": event} if event is not None else None
        if not isinstance(body, dict):
            return
        with self._lock:
            self._status = body

    def _wait_for_serial_owner(self, deadline: float) -> bool:
        discovery_deadline = min(
            deadline,
            self._monotonic() + self._discovery_timeout_seconds,
        )
        while not self._closed.is_set() and self._monotonic() < discovery_deadline:
            try:
                has_request_subscriber = self._node.count_subscribers(EYE_REQUEST_TOPIC) >= 1
                has_response_publisher = self._node.count_publishers(EYE_RESPONSE_TOPIC) >= 1
            except RuntimeError:
                return False
            if has_request_subscriber and has_response_publisher:
                return True
            self._closed.wait(
                min(0.05, max(0.0, discovery_deadline - self._monotonic()))
            )
        return False

    def _call(self, command: str, expected: str) -> dict[str, Any]:
        command = validate_eye_command(command)
        if expected not in {"state", "ack"}:
            raise EyeConfigError("眼睛响应类型无效", kind="invalid")
        request_id = secrets.token_urlsafe(18)
        event = threading.Event()
        result: dict[str, Any] = {}
        deadline = self._monotonic() + self._rpc_timeout_seconds
        with self._lock:
            if self._closed.is_set():
                raise EyeConfigError("眼睛串口服务已停止")
            self._pending[request_id] = (event, result)
        try:
            if not self._wait_for_serial_owner(deadline):
                raise EyeConfigError(
                    "未发现 serial_ros_node 的眼睛配置 RPC 端点",
                )
            message = self._String()
            message.data = encode_eye_request(request_id, command, expected)
            self._publisher.publish(message)
            remaining = deadline - self._monotonic()
            if remaining <= 0 or not event.wait(remaining):
                raise EyeConfigError("等待眼睛设备响应超时；请确认下位机已连接")
            if not result.get("ok"):
                event_payload = result.get("event")
                message_text = str(result.get("error") or "眼睛设备拒绝了命令")
                raise EyeConfigError(
                    message_text,
                    event=event_payload if isinstance(event_payload, dict) else None,
                    kind="device" if isinstance(event_payload, dict) and event_payload.get("kind") == "err" else "unavailable",
                )
            return result
        finally:
            with self._lock:
                self._pending.pop(request_id, None)

    def query(self) -> dict[str, Any]:
        return self._call("eyeconfig:query", "state")

    def command(self, command: str) -> dict[str, Any]:
        return self._call(command, "ack")

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status or {})

    def close(self) -> None:
        self._closed.set()
        self._thread.join(timeout=1.0)
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


__all__ = [
    "EYE_DISCOVERY_TIMEOUT_SECONDS",
    "EYE_REQUEST_TOPIC",
    "EYE_RESPONSE_TOPIC",
    "EYE_RPC_TIMEOUT_SECONDS",
    "EYE_STATUS_TOPIC",
    "EyeConfigError",
    "EyeConfigRpcClient",
    "encode_eye_request",
    "normalize_eye_state",
    "parse_eye_response",
    "validate_eye_command",
]
