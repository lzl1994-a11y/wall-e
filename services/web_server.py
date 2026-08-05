#!/usr/bin/env python3
"""Wali 本地配置网页服务。

只依赖 Python 标准库和项目已经使用的 PyYAML。服务默认仅监听
127.0.0.1；如果开放到局域网，必须配置访问令牌。
"""

from __future__ import annotations

import argparse
import copy
import hmac
import json
import os
import re
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "core" / "config.yaml"
DEFAULT_STATIC_DIR = ROOT / "web" / "config"
MAX_BODY_BYTES = 1024 * 1024
SECRET_NAMES = {"key", "api_key", "token", "secret", "password"}
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
BAIDU_CUID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
BAIDU_DEV_PIDS = {1537, 15372, 15376, 1737, 17372}


class ConfigError(ValueError):
    """配置读取、校验或写入失败。"""


def _is_secret_name(name: object) -> bool:
    return isinstance(name, str) and name.lower() in SECRET_NAMES


def _redact_secrets(value: Any, path: tuple[str, ...] = ()) -> tuple[Any, dict[str, bool]]:
    secret_fields: dict[str, bool] = {}
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            child_path = path + (str(key),)
            dotted = ".".join(child_path)
            if _is_secret_name(key):
                result[key] = ""
                secret_fields[dotted] = item not in (None, "")
            else:
                redacted, child_secrets = _redact_secrets(item, child_path)
                result[key] = redacted
                secret_fields.update(child_secrets)
        return result, secret_fields
    if isinstance(value, list):
        result = []
        for index, item in enumerate(value):
            redacted, child_secrets = _redact_secrets(item, path + (str(index),))
            result.append(redacted)
            secret_fields.update(child_secrets)
        return result, secret_fields
    return value, secret_fields


def _merge_preserving_secrets(current: Any, incoming: Any) -> Any:
    """深度合并配置；空白密钥表示保留旧值。"""
    if not isinstance(incoming, dict):
        return copy.deepcopy(incoming)

    current_map = current if isinstance(current, dict) else {}
    merged = copy.deepcopy(current_map)
    for key, item in incoming.items():
        old_item = current_map.get(key)
        if _is_secret_name(key) and item in (None, ""):
            merged[key] = copy.deepcopy(old_item)
        elif isinstance(item, dict):
            merged[key] = _merge_preserving_secrets(old_item, item)
        else:
            merged[key] = copy.deepcopy(item)
    return merged


def _require_mapping(config: dict[str, Any], key: str, errors: list[str]) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key} 必须是配置对象")
        return {}
    return value


def _check_string(
    mapping: dict[str, Any],
    key: str,
    path: str,
    errors: list[str],
    *,
    allow_empty: bool = False,
    max_length: int = 4096,
) -> None:
    value = mapping.get(key)
    if not isinstance(value, str):
        errors.append(f"{path} 必须是字符串")
    elif not allow_empty and not value.strip():
        errors.append(f"{path} 不能为空")
    elif len(value) > max_length:
        errors.append(f"{path} 不能超过 {max_length} 个字符")


def _check_bool(mapping: dict[str, Any], key: str, path: str, errors: list[str]) -> None:
    if not isinstance(mapping.get(key), bool):
        errors.append(f"{path} 必须是布尔值")


def _check_number(
    mapping: dict[str, Any],
    key: str,
    path: str,
    errors: list[str],
    minimum: float,
    maximum: float,
    *,
    integer: bool = False,
) -> None:
    value = mapping.get(key)
    expected = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, expected):
        errors.append(f"{path} 必须是{'整数' if integer else '数字'}")
        return
    if not minimum <= value <= maximum:
        errors.append(f"{path} 必须在 {minimum} 到 {maximum} 之间")


def _check_url(mapping: dict[str, Any], key: str, path: str, errors: list[str]) -> None:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.startswith(("http://", "https://")):
        errors.append(f"{path} 必须是 http:// 或 https:// 地址")


def _validate_asr(asr: dict[str, Any], errors: list[str]) -> None:
    provider = asr.get("provider")
    if provider not in {"aliyun", "zhipu", "baidu"}:
        errors.append("asr.provider 只能是 aliyun、zhipu 或 baidu")
        return

    nested = asr.get(provider)
    settings = nested if isinstance(nested, dict) else asr
    prefix = f"asr.{provider}" if isinstance(nested, dict) else "asr"

    if provider in {"aliyun", "zhipu"}:
        _check_string(settings, "model", f"{prefix}.model", errors)
        key_name = "api_key" if isinstance(nested, dict) else "key"
        effective_key = settings.get(key_name)
        if isinstance(nested, dict) and effective_key is None:
            effective_key = asr.get("key")
        if not isinstance(effective_key, str):
            errors.append(f"{prefix}.{key_name} 必须是字符串")
        elif len(effective_key) > 8192:
            errors.append(f"{prefix}.{key_name} 不能超过 8192 个字符")
        if provider == "zhipu":
            _check_url(settings, "url", f"{prefix}.url", errors)
        return

    _check_number(settings, "app_id", f"{prefix}.app_id", errors, 1, 9223372036854775807, integer=True)
    _check_string(settings, "api_key", f"{prefix}.api_key", errors, max_length=8192)
    dev_pid = settings.get("dev_pid")
    if isinstance(dev_pid, bool) or not isinstance(dev_pid, int) or dev_pid not in BAIDU_DEV_PIDS:
        errors.append(f"{prefix}.dev_pid 不是百度支持的模型 PID")
    cuid = settings.get("cuid")
    if not isinstance(cuid, str) or not BAIDU_CUID_PATTERN.fullmatch(cuid):
        errors.append(f"{prefix}.cuid 只能包含字母、数字、下划线和连字符，长度 1-128")
    url = settings.get("url")
    if not isinstance(url, str) or not url.startswith("wss://"):
        errors.append(f"{prefix}.url 必须是 wss:// 地址")

    if "lm_id" in settings and settings.get("lm_id") is not None:
        _check_number(settings, "lm_id", f"{prefix}.lm_id", errors, 1, 9223372036854775807, integer=True)
    if dev_pid == 15376:
        _check_string(settings, "user", f"{prefix}.user", errors, max_length=256)
    elif "user" in settings and settings.get("user") is not None:
        _check_string(settings, "user", f"{prefix}.user", errors, allow_empty=True, max_length=256)


def _validate_servos(servos: Any, errors: list[str]) -> None:
    if not isinstance(servos, list) or not servos:
        errors.append("servos 必须是非空列表")
        return

    ids: set[int] = set()
    names: set[str] = set()
    for index, servo in enumerate(servos):
        prefix = f"servos[{index}]"
        if not isinstance(servo, dict):
            errors.append(f"{prefix} 必须是配置对象")
            continue
        _check_number(servo, "id", f"{prefix}.id", errors, 0, 15, integer=True)
        _check_string(servo, "name", f"{prefix}.name", errors, max_length=64)
        for key in ("limit_1", "limit_2", "init"):
            _check_number(servo, key, f"{prefix}.{key}", errors, 0, 65535, integer=True)

        servo_id = servo.get("id")
        name = servo.get("name")
        if isinstance(servo_id, int) and not isinstance(servo_id, bool):
            if servo_id in ids:
                errors.append(f"舵机 ID {servo_id} 重复")
            ids.add(servo_id)
        if isinstance(name, str):
            if not NAME_PATTERN.fullmatch(name):
                errors.append(f"{prefix}.name 只能包含字母、数字、下划线和连字符")
            if name in names:
                errors.append(f"舵机名称 {name} 重复")
            names.add(name)

        limit_1 = servo.get("limit_1")
        limit_2 = servo.get("limit_2")
        initial = servo.get("init")
        if all(isinstance(item, int) and not isinstance(item, bool) for item in (limit_1, limit_2, initial)):
            low, high = sorted((limit_1, limit_2))
            if not low <= initial <= high:
                errors.append(f"{prefix}.init 必须位于两个物理限位之间")


def _validate_motors(motors: Any, errors: list[str]) -> None:
    if not isinstance(motors, list) or not motors:
        errors.append("motors 必须是非空列表")
        return

    ids: set[int] = set()
    names: set[str] = set()
    for index, motor in enumerate(motors):
        prefix = f"motors[{index}]"
        if not isinstance(motor, dict):
            errors.append(f"{prefix} 必须是配置对象")
            continue
        _check_number(motor, "id", f"{prefix}.id", errors, 0, 15, integer=True)
        _check_string(motor, "name", f"{prefix}.name", errors, max_length=64)
        _check_number(motor, "max_speed", f"{prefix}.max_speed", errors, 0, 100, integer=True)
        _check_number(motor, "neutral_speed", f"{prefix}.neutral_speed", errors, -100, 100, integer=True)
        _check_bool(motor, "invert_direction", f"{prefix}.invert_direction", errors)

        motor_id = motor.get("id")
        name = motor.get("name")
        if isinstance(motor_id, int) and not isinstance(motor_id, bool):
            if motor_id in ids:
                errors.append(f"电机 ID {motor_id} 重复")
            ids.add(motor_id)
        if isinstance(name, str):
            if not NAME_PATTERN.fullmatch(name):
                errors.append(f"{prefix}.name 只能包含字母、数字、下划线和连字符")
            if name in names:
                errors.append(f"电机名称 {name} 重复")
            names.add(name)


def validate_config(config: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(config, dict):
        return ["配置根节点必须是对象"]

    pipeline = _require_mapping(config, "pipeline", errors)
    if pipeline.get("mode") not in {"asr_llm", "multimodal"}:
        errors.append("pipeline.mode 只能是 asr_llm 或 multimodal")

    asr = _require_mapping(config, "asr", errors)
    _validate_asr(asr, errors)

    llm = _require_mapping(config, "llm", errors)
    for key in ("provider", "model"):
        _check_string(llm, key, f"llm.{key}", errors)
    _check_url(llm, "url", "llm.url", errors)
    _check_string(llm, "key", "llm.key", errors, allow_empty=True, max_length=8192)
    _check_number(llm, "temperature", "llm.temperature", errors, 0, 2)
    _check_number(llm, "max_tokens", "llm.max_tokens", errors, 1, 131072, integer=True)

    launch = _require_mapping(config, "launch", errors)
    for key in ("serial", "tracking"):
        _check_bool(launch, key, f"launch.{key}", errors)

    wake_word = _require_mapping(config, "wake_word", errors)
    _check_bool(wake_word, "enabled", "wake_word.enabled", errors)
    for key in ("keyword", "model_dir", "response_wav"):
        _check_string(wake_word, key, f"wake_word.{key}", errors)
    _check_number(wake_word, "threshold", "wake_word.threshold", errors, 0, 1)
    _check_number(wake_word, "awake_timeout", "wake_word.awake_timeout", errors, 1, 300)

    if not isinstance(config.get("system_prompt"), str) or not config.get("system_prompt", "").strip():
        errors.append("system_prompt 不能为空")
    elif len(config["system_prompt"]) > 100000:
        errors.append("system_prompt 不能超过 100000 个字符")

    tts = _require_mapping(config, "tts", errors)
    for key in ("engine", "voice", "output_device"):
        _check_string(tts, key, f"tts.{key}", errors)

    serial = _require_mapping(config, "serial", errors)
    for key in ("doa_port", "lower_board_port"):
        _check_string(serial, key, f"serial.{key}", errors)
    _check_number(serial, "baudrate", "serial.baudrate", errors, 1200, 4000000, integer=True)

    i2c = _require_mapping(config, "i2c", errors)
    _check_number(i2c, "bus", "i2c.bus", errors, 0, 32, integer=True)
    _check_number(i2c, "pca9685_address", "i2c.pca9685_address", errors, 0, 127, integer=True)
    _check_number(i2c, "pwm_frequency", "i2c.pwm_frequency", errors, 1, 1600, integer=True)

    vision = _require_mapping(config, "vision", errors)
    _check_number(vision, "camera_index", "vision.camera_index", errors, 0, 32, integer=True)
    _check_string(vision, "model_path", "vision.model_path", errors)
    _check_bool(vision, "enabled_on_start", "vision.enabled_on_start", errors)
    pid = vision.get("pid")
    if not isinstance(pid, dict):
        errors.append("vision.pid 必须是配置对象")
    else:
        for key in ("kp", "ki", "kd"):
            _check_number(pid, key, f"vision.pid.{key}", errors, -100, 100)

    _validate_servos(config.get("servos"), errors)
    _validate_motors(config.get("motors"), errors)
    return errors


class ConfigStore:
    def __init__(self, config_path: Path | str):
        self.path = Path(config_path).expanduser().resolve()
        self._lock = threading.RLock()

    def load(self) -> dict[str, Any]:
        with self._lock:
            try:
                data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            except FileNotFoundError as exc:
                raise ConfigError(f"配置文件不存在: {self.path}") from exc
            except (OSError, yaml.YAMLError) as exc:
                raise ConfigError(f"读取配置失败: {exc}") from exc
            if not isinstance(data, dict):
                raise ConfigError("配置根节点必须是对象")
            return data

    def public_snapshot(self) -> dict[str, Any]:
        config = self.load()
        redacted, secret_fields = _redact_secrets(config)
        stat = self.path.stat()
        return {
            "config": redacted,
            "secret_fields": secret_fields,
            "config_path": str(self.path),
            "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        }

    def _save_merged(self, current: dict[str, Any], incoming: Any) -> dict[str, Any]:
        merged = _merge_preserving_secrets(current, incoming)
        errors = validate_config(merged)
        if errors:
            raise ConfigError("\n".join(errors))

        rendered = yaml.safe_dump(
            merged,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
            width=120,
        )
        temp_path = self.path.with_name(self.path.name + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with temp_path.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(rendered)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, self.path)
        except OSError as exc:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise ConfigError(f"写入配置失败: {exc}") from exc
        return self.public_snapshot()

    def save(self, incoming: Any) -> dict[str, Any]:
        with self._lock:
            return self._save_merged(self.load(), incoming)

    def save_patch(self, patch: Any) -> dict[str, Any]:
        if not isinstance(patch, dict) or not patch:
            raise ConfigError("配置补丁必须是非空对象")
        with self._lock:
            return self._save_merged(self.load(), patch)


class ConfigWebServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        store: ConfigStore,
        static_dir: Path,
        token: str | None,
    ):
        super().__init__(server_address, handler_class)
        self.store = store
        self.static_dir = static_dir.resolve()
        self.access_token = token or ""


class ConfigRequestHandler(BaseHTTPRequestHandler):
    server: ConfigWebServer

    STATIC_ROUTES = {
        "/": ("index.html", "text/html; charset=utf-8"),
        "/index.html": ("index.html", "text/html; charset=utf-8"),
        "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    }

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[ConfigWeb] {self.client_address[0]} {fmt % args}")

    def _authorized(self) -> bool:
        expected = self.server.access_token
        if not expected:
            return True
        supplied = self.headers.get("X-Wali-Token", "")
        try:
            return hmac.compare_digest(supplied.encode("ascii"), expected.encode("ascii"))
        except UnicodeEncodeError:
            # HTTP header values are not a reliable transport for arbitrary Unicode.
            return False

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _require_api_auth(self) -> bool:
        if self._authorized():
            return True
        self._send_json(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "访问令牌无效"})
        return False

    def _serve_static(self, route: str) -> None:
        static = self.STATIC_ROUTES.get(route)
        if static is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        filename, content_type = static
        path = self.server.static_dir / filename
        try:
            body = path.read_bytes()
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        route = urlsplit(self.path).path
        if route in self.STATIC_ROUTES:
            self._serve_static(route)
            return
        if route == "/api/health":
            if not self._require_api_auth():
                return
            self._send_json(HTTPStatus.OK, {"ok": True, "service": "wali-config-web", "restart_required_after_save": True})
            return
        if route == "/api/config":
            if not self._require_api_auth():
                return
            try:
                snapshot = self.server.store.public_snapshot()
            except ConfigError as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})
                return
            self._send_json(HTTPStatus.OK, {"ok": True, **snapshot})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        route = urlsplit(self.path).path
        if route != "/api/config":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not self._require_api_auth():
            return

        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            self._send_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"ok": False, "error": "只接受 application/json"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0 or content_length > MAX_BODY_BYTES:
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"ok": False, "error": "请求体为空或超过 1MB"})
            return

        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "JSON 格式错误"})
            return
        is_patch = isinstance(payload, dict) and "patch" in payload
        incoming = payload.get("patch" if is_patch else "config") if isinstance(payload, dict) else None
        try:
            snapshot = self.server.store.save_patch(incoming) if is_patch else self.server.store.save(incoming)
        except ConfigError as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "配置校验失败", "details": str(exc).splitlines()},
            )
            return
        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "message": ("模块已保存，重启主脑后生效" if is_patch else "配置已保存，重启主脑后生效"),
                **snapshot,
            },
        )


def _is_loopback_host(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


def create_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    static_dir: Path | str = DEFAULT_STATIC_DIR,
    token: str | None = None,
) -> ConfigWebServer:
    if not _is_loopback_host(host) and not token:
        raise ConfigError("监听局域网地址时必须通过 --token 或 WALI_CONFIG_TOKEN 设置访问令牌")
    if token:
        try:
            token.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ConfigError("访问令牌只能使用 ASCII 字母、数字和符号，不能包含中文") from exc
    static_path = Path(static_dir).expanduser().resolve()
    for filename in ("index.html", "app.js", "styles.css"):
        if not (static_path / filename).is_file():
            raise ConfigError(f"网页资源缺失: {static_path / filename}")
    return ConfigWebServer(
        (host, port),
        ConfigRequestHandler,
        store=ConfigStore(config_path),
        static_dir=static_path,
        token=token,
    )


def run_web_server(bus: object | None = None) -> None:
    """兼容旧 main.py 的阻塞式入口；bus 参数保留但不再使用。"""
    del bus
    host = os.environ.get("WALI_CONFIG_HOST", "127.0.0.1")
    port = int(os.environ.get("WALI_CONFIG_PORT", "8080"))
    token = os.environ.get("WALI_CONFIG_TOKEN")
    server = create_server(host=host, port=port, token=token)
    print(f"[ConfigWeb] 配置页面已启动: http://{host}:{server.server_port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Wali config.yaml 配置网页")
    parser.add_argument("--host", default=os.environ.get("WALI_CONFIG_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("WALI_CONFIG_PORT", "8080")))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="config.yaml 路径")
    parser.add_argument("--token", default=os.environ.get("WALI_CONFIG_TOKEN"), help="局域网访问令牌")
    args = parser.parse_args()

    try:
        server = create_server(host=args.host, port=args.port, config_path=args.config, token=args.token)
    except ConfigError as exc:
        parser.error(str(exc))
        return

    display_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    print(f"[ConfigWeb] 配置文件: {server.store.path}")
    print(f"[ConfigWeb] 打开页面: http://{display_host}:{server.server_port}")
    if server.access_token:
        print("[ConfigWeb] API 已启用访问令牌保护")
    print("[ConfigWeb] Ctrl+C 停止服务")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[ConfigWeb] 正在停止...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
