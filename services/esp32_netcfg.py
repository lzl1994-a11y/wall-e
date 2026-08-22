"""ESP32 ``NETCFG`` protocol encoder/parser used by the sole USB serial owner.

Production calls execute inside ``serial_ros_node`` through ``SerialBridge``;
the web service relays requests over ROS and never opens the ESP32 port itself.
The optional direct-open path exists only for diagnostics and focused tests.
"""

from __future__ import annotations

import base64
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

try:
    import serial
except ImportError:  # pragma: no cover - surfaced as a user-facing connection error
    serial = None  # type: ignore[assignment]

try:
    from services.usb_devices import DEFAULT_CONFIG_PATH, serial_ports_for_role
except ImportError:  # Supports: python services/esp32_netcfg.py
    from usb_devices import DEFAULT_CONFIG_PATH, serial_ports_for_role


PROTOCOL_VERSION = 1
MAX_COMMAND_BYTES = 512
SET_RESPONSE_TIMEOUT_SECONDS = 8.0
APPLY_ACCEPT_TIMEOUT_SECONDS = 8.0
# The firmware can take about 60 seconds. Never shorten this below 65 seconds.
APPLY_FINAL_TIMEOUT_SECONDS = 65.0
QUERY_RESPONSE_TIMEOUT_SECONDS = 8.0
_BASE64_URLSAFE = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")


class NetworkConfigError(ValueError):
    """Safe error suitable for returning to the browser (never includes a password)."""


@dataclass(frozen=True, repr=False)
class WifiCredential:
    """Validated credentials whose repr deliberately never exposes the password."""

    ssid: str
    password: str = field(repr=False)

    def __repr__(self) -> str:
        return f"WifiCredential(ssid={self.ssid!r}, password=<redacted>)"


@dataclass(frozen=True, repr=False)
class NetworkSettings:
    """Validated NETCFG input, safe to include in diagnostic object output."""

    wifi: tuple[WifiCredential, WifiCredential, WifiCredential]
    host: str
    port: int

    def __repr__(self) -> str:
        return f"NetworkSettings(wifi={self.wifi!r}, host={self.host!r}, port={self.port!r})"


RESULT_MESSAGES = {
    3: "设备拒绝了命令或参数",
    4: "设备没有可应用的候选网络配置",
    5: "设备已连通，但写入 NVS 失败",
    6: "Wi-Fi、TCP 图像服务器或 HELLO 验证失败，设备已回退到原配置",
}
DETAIL_MESSAGES = {
    1: "协议版本错误",
    2: "字段数量或命令格式错误",
    3: "SSID 编码或长度错误",
    4: "Wi-Fi 密码编码或长度错误",
    5: "图像服务器地址编码或长度错误",
    6: "图像服务器端口错误",
    7: "命令长度错误，或设备当前状态不允许操作",
    8: "没有配置任何有效 SSID",
}


def _utf8_bytes(value: Any, field: str, minimum: int, maximum: int) -> bytes:
    if not isinstance(value, str):
        raise NetworkConfigError(f"{field} 必须是文本")
    encoded = value.encode("utf-8")
    if not minimum <= len(encoded) <= maximum:
        if minimum:
            raise NetworkConfigError(f"{field} 的 UTF-8 长度必须为 {minimum}–{maximum} 字节")
        raise NetworkConfigError(f"{field} 的 UTF-8 长度不能超过 {maximum} 字节")
    return encoded


def encode_urlsafe_base64(value: str) -> str:
    """Encode UTF-8 text as unpadded URL-safe Base64."""
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")


def decode_urlsafe_base64(value: str, field: str) -> str:
    if not isinstance(value, str) or "=" in value or any(char not in _BASE64_URLSAFE for char in value):
        raise NetworkConfigError(f"设备返回的 {field} Base64 编码无效")
    try:
        decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)).decode("utf-8")
    except (UnicodeDecodeError, ValueError) as exc:
        raise NetworkConfigError(f"设备返回的 {field} Base64 编码无效") from exc
    return decoded


def validate_network_payload(payload: Any) -> NetworkSettings:
    """Validate browser input by UTF-8 *byte* count, and return three Wi-Fi pairs."""
    if not isinstance(payload, dict):
        raise NetworkConfigError("网络配置必须是对象")
    wifi = payload.get("wifi")
    if not isinstance(wifi, list) or len(wifi) != 3:
        raise NetworkConfigError("必须提供固定的 3 组 Wi-Fi 输入")
    pairs: list[WifiCredential] = []
    has_ssid = False
    for index, item in enumerate(wifi, start=1):
        if not isinstance(item, dict):
            raise NetworkConfigError(f"Wi-Fi {index} 必须是对象")
        ssid = item.get("ssid", "")
        password = item.get("password", "")
        _utf8_bytes(ssid, f"Wi-Fi {index} SSID", 0, 32)
        _utf8_bytes(password, f"Wi-Fi {index} 密码", 0, 64)
        if not ssid and password:
            raise NetworkConfigError(f"未使用的 Wi-Fi {index} 必须同时留空 SSID 和密码")
        has_ssid = has_ssid or bool(ssid)
        pairs.append(WifiCredential(ssid=ssid, password=password))
    if not has_ssid:
        raise NetworkConfigError("至少必须配置一个非空 SSID")
    host = payload.get("host")
    _utf8_bytes(host, "图像服务器地址", 1, 64)
    port = payload.get("port")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise NetworkConfigError("图像服务器端口必须是 1–65535 的整数")
    return NetworkSettings(wifi=(pairs[0], pairs[1], pairs[2]), host=host, port=port)


class Esp32NetworkConfigurator:
    """Execute SET/APPLY/QUERY transactions and correlate all responses by seq."""

    def __init__(
        self,
        config_path: Path | str = DEFAULT_CONFIG_PATH,
        *,
        serial_factory: Callable[..., Any] | None = None,
        port_resolver: Callable[[], str] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.config_path = Path(config_path)
        self._serial_factory = serial_factory
        self._port_resolver = port_resolver
        self._monotonic = monotonic
        self._lock = threading.Lock()

    def _next_seq(self) -> int:
        return secrets.randbits(32)

    def _next_distinct_seq(self, previous: int) -> int:
        """Generate a new sequence even if the random source happens to collide."""
        for _ in range(8):
            candidate = self._next_seq()
            if candidate != previous:
                return candidate
        return (previous + 1) & 0xFFFFFFFF

    def _resolve_port(self) -> str:
        if self._port_resolver:
            return self._port_resolver()
        ports, configured = serial_ports_for_role("screen_motion", self.config_path)
        if configured:
            if not ports:
                raise NetworkConfigError("已配置的屏幕/运动 USB 设备未连接或没有串口接口")
            return ports[0]
        try:
            config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise NetworkConfigError(f"读取串口配置失败: {exc}") from exc
        port = config.get("serial", {}).get("lower_board_port") if isinstance(config, dict) else None
        if not isinstance(port, str) or not port.strip():
            raise NetworkConfigError("请先在硬件页选择屏幕/运动 USB，或填写下位机串口")
        return port.strip()

    def _open(self) -> Any:
        if self._serial_factory is None:
            if serial is None:
                raise NetworkConfigError("未安装 pyserial，无法通过 USB 串口配置网络")
            factory = serial.Serial
        else:
            factory = self._serial_factory
        try:
            return factory(
                self._resolve_port(), baudrate=115200,
                bytesize=8, parity="N", stopbits=1, timeout=0.1, write_timeout=2,
            )
        except Exception as exc:
            raise NetworkConfigError("无法打开 ESP32 USB 串口；请确认设备已连接且串口未被屏幕控制节点占用") from exc

    @staticmethod
    def _command(text: str) -> bytes:
        raw = (text + "\r\n").encode("ascii")
        if len(raw) > MAX_COMMAND_BYTES:
            raise NetworkConfigError("网络配置命令超过 512 字节")
        return raw

    @staticmethod
    def _result_from_line(line: str) -> tuple[int, str, int, int] | None:
        if not line.startswith("NETCFG:RESULT:"):
            return None
        fields = line[len("NETCFG:RESULT:"):].split("|")
        if len(fields) != 4:
            return None
        try:
            seq, result, detail = int(fields[0]), int(fields[2]), int(fields[3])
        except ValueError:
            return None
        if not 0 <= seq <= 0xFFFFFFFF or fields[1] not in {"SET", "APPLY", "QUERY"}:
            return None
        return seq, fields[1], result, detail

    def _read_line(self, stream: Any, deadline: float) -> str | None:
        buffer = bytearray()
        while self._monotonic() < deadline:
            chunk = stream.read(1)
            if not chunk:
                continue
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8", "ignore")
            for byte in chunk:
                if byte in (10, 13):
                    if buffer:
                        return buffer.decode("utf-8", errors="replace")
                    continue
                # Bound unrelated firmware log lines as well as our memory use.
                if len(buffer) < 4096:
                    buffer.append(byte)
        return None

    def _wait_result(self, stream: Any, seq: int, operation: str, timeout: float, *, accepted: bool = False) -> tuple[int, int]:
        deadline = self._monotonic() + timeout
        while self._monotonic() < deadline:
            line = self._read_line(stream, deadline)
            if line is None:
                break
            parsed = self._result_from_line(line)
            if parsed is None:
                continue  # Ordinary logs and malformed NETCFG lines are not responses.
            response_seq, response_operation, result, detail = parsed
            if response_seq != seq or response_operation != operation:
                continue
            if accepted and (result, detail) != (1, 0):
                return result, detail
            return result, detail
        phase = "接受" if accepted else "最终"
        raise NetworkConfigError(f"等待设备 {operation} {phase}响应超时")

    @staticmethod
    def _result_error(operation: str, result: int, detail: int) -> NetworkConfigError:
        if result == 2 and detail == 0:
            return NetworkConfigError("内部状态错误")
        base = RESULT_MESSAGES.get(result, f"设备返回未知结果码 {result}")
        suffix = DETAIL_MESSAGES.get(detail, f"附加错误码 {detail}") if detail else ""
        return NetworkConfigError(f"{operation} 失败：{base}{('（' + suffix + '）') if suffix else ''}")

    def save_and_apply(self, payload: NetworkSettings | Any, *, stream: Any | None = None) -> dict[str, Any]:
        settings = payload if isinstance(payload, NetworkSettings) else validate_network_payload(payload)
        encoded = [
            encode_urlsafe_base64(value) if value else ""
            for credential in settings.wifi
            for value in (credential.ssid, credential.password)
        ]
        host_b64 = encode_urlsafe_base64(settings.host)
        # Command is constructed after validation; never log this because it contains passwords.
        with self._lock:
            owns_stream = stream is None
            stream = stream or self._open()
            try:
                set_seq = self._next_seq()
                set_command = self._command(
                    f"netcfg:set:{set_seq}|1|" + "|".join(encoded) + f"|{host_b64}|{settings.port}"
                )
                stream.write(set_command)
                if hasattr(stream, "flush"):
                    stream.flush()
                result, detail = self._wait_result(stream, set_seq, "SET", SET_RESPONSE_TIMEOUT_SECONDS)
                if (result, detail) != (0, 0):
                    raise self._result_error("SET", result, detail)

                apply_seq = self._next_distinct_seq(set_seq)
                stream.write(self._command(f"netcfg:apply:{apply_seq}|1"))
                if hasattr(stream, "flush"):
                    stream.flush()
                result, detail = self._wait_result(stream, apply_seq, "APPLY", APPLY_ACCEPT_TIMEOUT_SECONDS, accepted=True)
                if (result, detail) != (1, 0):
                    raise self._result_error("APPLY", result, detail)
                result, detail = self._wait_result(stream, apply_seq, "APPLY", APPLY_FINAL_TIMEOUT_SECONDS)
                if (result, detail) != (2, 0):
                    raise self._result_error("APPLY", result, detail)
                return {"set_seq": set_seq, "apply_seq": apply_seq, "result": "applied"}
            finally:
                if owns_stream:
                    try:
                        stream.close()
                    except Exception:
                        pass

    def query(self, *, stream: Any | None = None) -> dict[str, Any]:
        with self._lock:
            owns_stream = stream is None
            stream = stream or self._open()
            try:
                seq = self._next_seq()
                stream.write(self._command(f"netcfg:query:{seq}|1"))
                if hasattr(stream, "flush"):
                    stream.flush()
                deadline = self._monotonic() + QUERY_RESPONSE_TIMEOUT_SECONDS
                while self._monotonic() < deadline:
                    line = self._read_line(stream, deadline)
                    if line is None:
                        continue
                    result = self._result_from_line(line)
                    if result is not None:
                        response_seq, operation, result_code, detail = result
                        if response_seq == seq and operation == "QUERY":
                            raise self._result_error("QUERY", result_code, detail)
                        continue
                    if not line.startswith("NETCFG:STATUS:"):
                        continue
                    fields = line[len("NETCFG:STATUS:"):].split("|")
                    if len(fields) != 9:
                        continue
                    try:
                        response_seq, version, flags, selected, port = map(int, (fields[0], fields[1], fields[2], fields[3], fields[8]))
                    except ValueError:
                        continue
                    if response_seq != seq or version != PROTOCOL_VERSION:
                        continue
                    if not 0 <= flags <= 7 or selected not in {0, 1, 2, 255} or not 0 <= port <= 65535:
                        continue
                    ssids = [decode_urlsafe_base64(value, f"Wi-Fi {index} SSID") if value else "" for index, value in enumerate(fields[4:7], 1)]
                    host = decode_urlsafe_base64(fields[7], "图像服务器地址") if fields[7] else ""
                    return {
                        "seq": seq,
                        "active_from_nvs": bool(flags & 1),
                        "candidate_present": bool(flags & 2),
                        "apply_running": bool(flags & 4),
                        "selected": selected,
                        "wifi": [{"ssid": ssid} for ssid in ssids],
                        "host": host,
                        "port": port,
                    }
                raise NetworkConfigError("等待设备 QUERY 响应超时")
            finally:
                if owns_stream:
                    try:
                        stream.close()
                    except Exception:
                        pass
