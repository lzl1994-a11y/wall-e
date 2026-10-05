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
import time
from pathlib import Path
from typing import Any

import yaml

from services.hardware.esp32_netcfg_rpc import _SerialOwnerRpcClient
from services.hardware.usb_devices import DEFAULT_CONFIG_PATH


EYE_REQUEST_TOPIC = "eyeconfig_request"
EYE_RESPONSE_TOPIC = "eyeconfig_response"
EYE_STATUS_TOPIC = "eye_status"
EYE_RPC_TIMEOUT_SECONDS = 8.0
EYE_DISCOVERY_TIMEOUT_SECONDS = 3.0

# Firmware contract: wall-e-tft/src/domain/EyeRenderer.cpp::parseEyeSettings.
# Both the API validator and browser sliders consume these limits.
EYE_FIELD_LIMITS = {
    "brightness": {"min": 0, "max": 1, "integer": False},
    "ringBrightness": {"min": 0, "max": 1, "integer": False},
    "dotBrightness": {"min": 0, "max": 1, "integer": False},
    "scale": {"min": 0.4, "max": 1.5, "integer": False},
    "glow": {"min": 8, "max": 30, "integer": True},
    "breathMs": {"min": 500, "max": 10000, "integer": True, "allowZero": True},
    "blinkMs": {"min": 1000, "max": 15000, "integer": True, "allowZero": True},
    "dots": {"min": 0, "max": 64, "integer": True},
    "lookX": {"min": -26, "max": 26, "integer": True},
    "lookY": {"min": -26, "max": 26, "integer": True},
}

# The upper host is authoritative.  ``blinkMs`` is retained even when
# ``autoBlink`` is false so re-enabling the switch restores the user's period.
EYE_CONFIG_DEFAULTS = {
    "color": "00E5FF",
    "ringColor": "00CFE8",
    "dotColor": "00BCD0",
    "brightness": 0.85,
    "ringBrightness": 0.65,
    "dotBrightness": 0.45,
    "scale": 1.0,
    "glow": 22,
    "lookX": 0,
    "lookY": 0,
    "breathMs": 2400,
    "autoBlink": True,
    "blinkMs": 4500,
    "ring": True,
    "dots": 48,
    "mood": "dot",
}
EYE_CONFIG_KEYS = tuple(EYE_CONFIG_DEFAULTS)

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

    for key, limits in EYE_FIELD_LIMITS.items():
        if key in {"lookX", "lookY"}:
            continue
        pattern = r"-?\d+" if limits["integer"] else r"[^\s]+"
        match = re.fullmatch(rf"eyeconfig:{key}=({pattern})", command)
        if match and (
            _number_in_range(match.group(1), limits["min"], limits["max"])
            or (limits.get("allowZero") and float(match.group(1)) == 0)
        ):
            value = int(match.group(1)) if limits["integer"] else float(match.group(1))
            return f"eyeconfig:{key}={value:g}"

    match = re.fullmatch(r"eyeconfig:ring=([01])", command)
    if match:
        return command

    match = re.fullmatch(r"eyeconfig:mood=(dot|flame|heart)", command)
    if match:
        return command

    match = re.fullmatch(
        rf"eyeaction:look:x=({_SIGNED_INTEGER_RE}),y=({_SIGNED_INTEGER_RE})",
        command,
    )
    if match and all(
        EYE_FIELD_LIMITS[key]["min"] <= int(value) <= EYE_FIELD_LIMITS[key]["max"]
        for key, value in zip(("lookX", "lookY"), match.groups())
    ):
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
    "ringcolor": "ringColor",
    "dotcolor": "dotColor",
    "brightness": "brightness",
    "ringbrightness": "ringBrightness",
    "dotbrightness": "dotBrightness",
    "scale": "scale",
    "minscale": "minScale",
    "maxscale": "maxScale",
    "glow": "glow",
    "breathms": "breathMs",
    "blinkms": "blinkMs",
    "autoblink": "autoBlink",
    "ring": "ring",
    "dots": "dots",
    "mood": "mood",
    "x": "lookX",
    "lookx": "lookX",
    "y": "lookY",
    "looky": "lookY",
}

_BOOLEAN_STATE_KEYS = {"autoBlink", "ring"}
_INTEGER_STATE_KEYS = {"breathMs", "blinkMs", "dots"}
_FLOAT_STATE_KEYS = {"brightness", "ringBrightness", "dotBrightness", "scale", "minScale", "maxScale", "glow", "lookX", "lookY"}
_KNOWN_MOODS = {"dot", "flame", "heart"}


def _canonical_state_key(key: Any) -> str | None:
    normalized = str(key).lower()
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


def normalize_eye_config(config: Any) -> dict[str, Any]:
    """Validate the host-owned eye configuration and fill omitted defaults."""
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise EyeConfigError("eye 必须是配置对象", kind="invalid")
    unknown = sorted(str(key) for key in config if key not in EYE_CONFIG_KEYS)
    if unknown:
        raise EyeConfigError(f"eye 包含不支持的字段: {', '.join(map(str, unknown))}", kind="invalid")

    result = dict(EYE_CONFIG_DEFAULTS)
    for key in EYE_CONFIG_KEYS:
        if key not in config:
            continue
        raw = config[key]
        if key in {"color", "ringColor", "dotColor"}:
            value = str(raw).strip().lstrip("#").upper()
            valid = re.fullmatch(_COLOR_RE, value) is not None
        elif key in _BOOLEAN_STATE_KEYS:
            value = _coerce_state_value(key, raw)
            valid = value is not None
        elif key == "mood":
            value = str(raw).strip().lower()
            valid = value in _KNOWN_MOODS
        else:
            try:
                number = float(raw) if not isinstance(raw, bool) else float("nan")
            except (TypeError, ValueError, OverflowError):
                number = float("nan")
            limits = EYE_FIELD_LIMITS[key]
            valid = math.isfinite(number) and (
                limits["min"] <= number <= limits["max"]
                or (limits.get("allowZero") and number == 0)
            )
            if valid and limits["integer"] and number != int(number):
                valid = False
            value = int(number) if valid and limits["integer"] else number
        if not valid:
            raise EyeConfigError(f"eye.{key} 超出支持的配置范围", kind="invalid")
        result[key] = value

    if result["autoBlink"] and result["blinkMs"] == 0:
        raise EyeConfigError("eye.blinkMs 开启自动眨眼时不能为 0", kind="invalid")
    if not result["autoBlink"] and result["blinkMs"] == 0:
        result["blinkMs"] = EYE_CONFIG_DEFAULTS["blinkMs"]
    return result


def eye_config_commands(config: Any) -> list[str]:
    """Encode one complete host configuration for startup replay."""
    settings = normalize_eye_config(config)
    return [
        f"eyeconfig:color={settings['color']}",
        f"eyeconfig:ringColor={settings['ringColor']}",
        f"eyeconfig:dotColor={settings['dotColor']}",
        f"eyeconfig:brightness={settings['brightness']:g}",
        f"eyeconfig:ringBrightness={settings['ringBrightness']:g}",
        f"eyeconfig:dotBrightness={settings['dotBrightness']:g}",
        f"eyeconfig:scale={settings['scale']:g}",
        f"eyeconfig:glow={settings['glow']}",
        f"eyeconfig:breathMs={settings['breathMs']}",
        f"eyeconfig:blinkMs={settings['blinkMs'] if settings['autoBlink'] else 0}",
        f"eyeconfig:ring={1 if settings['ring'] else 0}",
        f"eyeconfig:dots={settings['dots']}",
        f"eyeconfig:mood={settings['mood']}",
        f"eyeaction:look:x={settings['lookX']},y={settings['lookY']}",
    ]


def load_saved_eye_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the upper-host eye config; missing legacy sections use defaults."""
    try:
        document = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return dict(EYE_CONFIG_DEFAULTS)
    except (OSError, yaml.YAMLError) as exc:
        raise EyeConfigError(f"读取上位机眼睛配置失败: {exc}", kind="invalid") from exc
    if not isinstance(document, dict):
        raise EyeConfigError("配置文件根节点必须是对象", kind="invalid")
    return normalize_eye_config(document.get("eye"))


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


class EyeConfigRpcClient(_SerialOwnerRpcClient):
    """Eye codec using the same ROS transport as the existing NETCFG client."""

    _request_topic = EYE_REQUEST_TOPIC
    _response_topic = EYE_RESPONSE_TOPIC
    _node_name = "walle_eye_config_web_client"
    _error_type = EyeConfigError

    def __init__(self, *, monotonic=time.monotonic,
                 discovery_timeout_seconds=EYE_DISCOVERY_TIMEOUT_SECONDS,
                 rpc_timeout_seconds=EYE_RPC_TIMEOUT_SECONDS):
        self._status = {}
        self._speaking = False
        self._rpc_timeout_seconds = max(0.1, float(rpc_timeout_seconds))
        super().__init__(
            monotonic=monotonic, discovery_timeout_seconds=discovery_timeout_seconds
        )
        try:
            self._status_subscription = self._node.create_subscription(
                self._String, EYE_STATUS_TOPIC, self._on_status, 10
            )
        except Exception as exc:
            self.close()
            raise EyeConfigError("ROS 眼睛状态订阅初始化失败") from exc

    def _on_status(self, message):
        try:
            body = json.loads(message.data)
        except (AttributeError, TypeError, json.JSONDecodeError):
            return
        if isinstance(body, dict):
            with self._lock:
                if isinstance(body.get("speaking"), bool):
                    self._speaking = body["speaking"]
                if isinstance(body.get("event"), dict):
                    self._status = body

    def _call(self, command, expected):
        result = self._exchange(
            {"command": validate_eye_command(command), "expected": expected},
            self._rpc_timeout_seconds,
        )
        if not result.get("ok"):
            event = result.get("event")
            raise EyeConfigError(
                str(result.get("error") or "眼睛设备拒绝了命令"),
                event=event if isinstance(event, dict) else None,
                kind="device" if isinstance(event, dict) and event.get("kind") == "err" else "unavailable",
            )
        if isinstance(result.get("speaking"), bool):
            with self._lock:
                self._speaking = result["speaking"]
        return result

    def query(self):
        return self._call("eyeconfig:query", "state")

    def command(self, command):
        command = validate_eye_command(command)
        return self._call(command, "state" if command == "eyeconfig:query" else "ack")

    def status(self):
        with self._lock:
            return {**self._status, "speaking": self._speaking}


__all__ = [
    "EYE_CONFIG_DEFAULTS",
    "EYE_CONFIG_KEYS",
    "EYE_DISCOVERY_TIMEOUT_SECONDS",
    "EYE_FIELD_LIMITS",
    "EYE_REQUEST_TOPIC",
    "EYE_RESPONSE_TOPIC",
    "EYE_RPC_TIMEOUT_SECONDS",
    "EYE_STATUS_TOPIC",
    "EyeConfigError",
    "EyeConfigRpcClient",
    "eye_config_commands",
    "encode_eye_request",
    "load_saved_eye_config",
    "normalize_eye_config",
    "normalize_eye_state",
    "parse_eye_response",
    "validate_eye_command",
]
