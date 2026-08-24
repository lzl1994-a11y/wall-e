"""TCP server and wire protocol for the WALL-E chest TFT camera preview."""

from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml


MAGIC = b"WTFT"
PROTOCOL_VERSION = 1
HEADER = struct.Struct("!4sBBHII")
STREAM_START = struct.Struct("!IIHH")

HELLO = 0x01
PING = 0x02
PONG = 0x03
STREAM_START_MESSAGE = 0x10
JPEG_FRAME = 0x11
STREAM_END = 0x12

EXPECTED_DEVICE_ID = "WALL_E_TFT"
MAX_INCOMING_PAYLOAD = 64 * 1024
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "core" / "config.yaml"


class ProtocolError(ValueError):
    """The peer sent a malformed or unsupported WTFT message."""


def encode_message(message_type: int, sequence: int, payload: bytes = b"") -> bytes:
    """Encode one complete WTFT message using network byte order."""
    body = bytes(payload)
    return HEADER.pack(
        MAGIC,
        PROTOCOL_VERSION,
        int(message_type) & 0xFF,
        0,
        int(sequence) & 0xFFFFFFFF,
        len(body),
    ) + body


def decode_header(data: bytes) -> tuple[int, int, int, int]:
    """Decode a 16-byte header into type, flags, sequence and payload length."""
    if len(data) != HEADER.size:
        raise ProtocolError(f"WTFT header must be {HEADER.size} bytes")
    magic, version, message_type, flags, sequence, payload_length = HEADER.unpack(data)
    if magic != MAGIC:
        raise ProtocolError("invalid WTFT magic")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported WTFT version: {version}")
    return message_type, flags, sequence, payload_length


def encode_stream_start(duration_ms: int, hold_ms: int, fps: int) -> bytes:
    return STREAM_START.pack(int(duration_ms), int(hold_ms), int(fps), 0)


@dataclass(frozen=True)
class TftPreviewSettings:
    bind_address: str = "0.0.0.0"
    port: int = 9000
    frame_provider: str = "ros_camera_frame"
    fps: int = 10
    recognition_duration_ms: int = 1500
    photo_duration_ms: int = 3000
    hold_ms: int = 3000
    jpeg_quality: int = 70
    max_frame_bytes: int = 256 * 1024
    photo_directory: str = "~/.wali/photos"

    @classmethod
    def from_mapping(cls, value: Any) -> "TftPreviewSettings":
        config = value if isinstance(value, dict) else {}
        return cls(
            bind_address=str(config.get("bind_address", cls.bind_address)),
            port=int(config.get("port", cls.port)),
            frame_provider=str(config.get("frame_provider", cls.frame_provider)),
            fps=min(30, max(1, int(config.get("fps", cls.fps)))),
            recognition_duration_ms=max(
                100, int(config.get("recognition_duration_ms", cls.recognition_duration_ms))
            ),
            photo_duration_ms=max(
                100, int(config.get("photo_duration_ms", cls.photo_duration_ms))
            ),
            hold_ms=max(0, int(config.get("hold_ms", cls.hold_ms))),
            jpeg_quality=min(100, max(1, int(config.get("jpeg_quality", cls.jpeg_quality)))),
            max_frame_bytes=min(
                256 * 1024,
                max(1024, int(config.get("max_frame_bytes", cls.max_frame_bytes))),
            ),
            photo_directory=str(config.get("photo_directory", cls.photo_directory)),
        )


def load_tft_preview_settings(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> TftPreviewSettings:
    try:
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        config = {}
    return TftPreviewSettings.from_mapping(config.get("tft_preview"))


@dataclass
class PreviewResult:
    last_frame: bytes | None = None
    source_frames: int = 0
    encoded_frames: int = 0
    sent_frames: int = 0
    no_new_frame_skips: int = 0
    backpressure_drops: int = 0
    encode_drops: int = 0
    total_bytes: int = 0
    elapsed_seconds: float = 0.0
    connected: bool = False
    busy: bool = False
    error: str = ""

    @property
    def actual_fps(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.sent_frames / self.elapsed_seconds

    @property
    def average_fps(self) -> float:
        """Backward-compatible name for actual transmitted FPS."""
        return self.actual_fps

    @property
    def dropped_frames(self) -> int:
        """Backward-compatible aggregate for older status consumers."""
        return self.encode_drops + self.backpressure_drops


def prepare_tft_jpeg(jpeg: bytes, *, quality: int = 70) -> bytes | None:
    """Fit a complete image within 240x240 without cropping or padding."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    raw = bytes(jpeg or b"")
    if not raw.startswith(b"\xff\xd8") or not raw.endswith(b"\xff\xd9"):
        return None
    try:
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None
    if image is None or image.size == 0:
        return None
    height, width = image.shape[:2]
    scale = min(240.0 / width, 240.0 / height)
    target_width = min(240, max(1, int(width * scale + 0.5)))
    target_height = min(240, max(1, int(height * scale + 0.5)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image,
        (target_width, target_height),
        interpolation=interpolation,
    )
    options = [int(cv2.IMWRITE_JPEG_QUALITY), min(100, max(1, int(quality)))]
    if hasattr(cv2, "IMWRITE_JPEG_PROGRESSIVE"):
        options.extend([int(cv2.IMWRITE_JPEG_PROGRESSIVE), 0])
    ok, encoded = cv2.imencode(".jpg", resized, options)
    if not ok:
        return None
    result = encoded.tobytes()
    if not result.startswith(b"\xff\xd8") or not result.endswith(b"\xff\xd9"):
        return None
    return result


class TftPreviewServer:
    """Background TCP service with one verified ESP32 client and one preview stream."""

    HEARTBEAT_SECONDS = 5.0

    def __init__(
        self,
        settings: TftPreviewSettings | None = None,
        *,
        logger: Any = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings or TftPreviewSettings()
        self._logger = logger
        self._clock = clock
        self._stop_event = threading.Event()
        self._server_socket: socket.socket | None = None
        self._client_socket: socket.socket | None = None
        self._client_address: tuple[str, int] | None = None
        self._device_id = ""
        self._client_lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._stream_lock = threading.Lock()
        self._sequence_lock = threading.Lock()
        self._control_sequence = 0
        self._stream_sequence = 0
        self._threads: list[threading.Thread] = []

    @property
    def port(self) -> int:
        server = self._server_socket
        if server is not None:
            return int(server.getsockname()[1])
        return self.settings.port

    @property
    def connected(self) -> bool:
        with self._client_lock:
            return self._client_socket is not None

    @property
    def device_connected(self) -> bool:
        with self._client_lock:
            return self._client_socket is not None and self._device_id == EXPECTED_DEVICE_ID

    def start(self) -> None:
        if self._server_socket is not None:
            return
        self._stop_event.clear()
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.settimeout(0.5)
        try:
            server.bind((self.settings.bind_address, self.settings.port))
            server.listen(2)
        except Exception:
            server.close()
            raise
        self._server_socket = server
        self._threads = [
            threading.Thread(target=self._accept_loop, name="tft-preview-accept", daemon=True),
            threading.Thread(target=self._heartbeat_loop, name="tft-preview-heartbeat", daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        self._log("info", f"TFT 预览服务监听 {self.settings.bind_address}:{self.port}")

    def stop(self) -> None:
        self._stop_event.set()
        server = self._server_socket
        self._server_socket = None
        if server is not None:
            try:
                server.close()
            except OSError:
                pass
        self._disconnect_client(reason="服务停止")
        for thread in self._threads:
            if thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=1.0)
        self._threads = []

    def send_camera_preview(
        self,
        frame_provider: Any,
        duration_ms: int = 3000,
        hold_ms: int = 3000,
        fps: int | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> PreviewResult:
        """Capture a preview, send it when a TFT is connected, and return the last source frame."""
        result = PreviewResult()
        if not self._stream_lock.acquire(blocking=False):
            result.busy = True
            result.error = "preview_busy"
            self._log("warn", "已有 TFT 预览流正在发送，忽略重复请求")
            return result

        requested_fps = self.settings.fps if fps is None else fps
        target_fps = min(30, max(1, int(requested_fps)))
        duration_ms = max(100, int(duration_ms))
        hold_ms = max(0, int(hold_ms))
        operation_started_at = self._clock()
        stream_started_at: float | None = None
        stream_sequence = self._next_stream_sequence()
        slot_condition = threading.Condition()
        latest_slot: tuple[int, bytes] | None = None
        capture_done = False
        network_failed = threading.Event()
        synthetic_source_sequence = 0
        last_source_sequence: int | None = None

        def record_no_new_frame() -> None:
            with slot_condition:
                result.no_new_frame_skips += 1

        def queue_source_frame(source_jpeg: bytes, source_sequence: int | None = None) -> None:
            nonlocal latest_slot, synthetic_source_sequence, last_source_sequence, stream_started_at
            if stream_started_at is None:
                stream_started_at = self._clock()
            if source_sequence is None:
                synthetic_source_sequence += 1
                source_sequence = synthetic_source_sequence
            if source_sequence == last_source_sequence:
                record_no_new_frame()
                return
            last_source_sequence = source_sequence
            source = bytes(source_jpeg or b"")
            if not source:
                result.encode_drops += 1
                return
            result.source_frames += 1
            result.last_frame = source
            # Keep the original frame for photo/cloud consumers, but avoid a
            # pointless decode-resize-encode cycle without a verified TFT.
            if not self.device_connected:
                return
            preview_jpeg = prepare_tft_jpeg(
                source,
                quality=self.settings.jpeg_quality,
            )
            if preview_jpeg is None or len(preview_jpeg) > self.settings.max_frame_bytes:
                result.encode_drops += 1
                return
            result.encoded_frames += 1
            with slot_condition:
                if latest_slot is not None:
                    result.backpressure_drops += 1
                latest_slot = (source_sequence, preview_jpeg)
                slot_condition.notify()

        def send_latest_frames() -> None:
            nonlocal latest_slot
            client: socket.socket | None = None
            network_active = False
            frame_index = 0
            while True:
                with slot_condition:
                    while latest_slot is None and not capture_done and not network_failed.is_set():
                        slot_condition.wait(timeout=0.2)
                    if network_failed.is_set():
                        return
                    if latest_slot is None and capture_done:
                        break
                    slot = latest_slot
                    latest_slot = None
                if slot is None:
                    continue
                _source_sequence, preview_jpeg = slot
                if client is None:
                    client = self._verified_client()
                    result.connected = client is not None
                if client is None:
                    continue
                try:
                    if not network_active:
                        self._send_packet(
                            client,
                            STREAM_START_MESSAGE,
                            stream_sequence,
                            encode_stream_start(duration_ms, hold_ms, target_fps),
                        )
                        network_active = True
                        self._log(
                            "info",
                            f"TFT 预览开始: duration={duration_ms}ms "
                            f"hold={hold_ms}ms fps={target_fps}",
                        )
                    sequence = ((stream_sequence & 0xFFFF) << 16) | (frame_index & 0xFFFF)
                    self._send_packet(client, JPEG_FRAME, sequence, preview_jpeg)
                    result.sent_frames += 1
                    result.total_bytes += len(preview_jpeg)
                    frame_index += 1
                except (ConnectionError, OSError) as exc:
                    result.error = str(exc)
                    network_failed.set()
                    with slot_condition:
                        latest_slot = None
                        slot_condition.notify_all()
                    self._disconnect_client(client, reason="预览发送失败")
                    return
            if network_active and client is not None:
                try:
                    self._send_packet(client, STREAM_END, stream_sequence, b"")
                except (ConnectionError, OSError) as exc:
                    result.error = result.error or str(exc)
                    network_failed.set()
                    self._disconnect_client(client, reason="结束消息发送失败")

        sender = threading.Thread(
            target=send_latest_frames,
            name="tft-preview-latest-frame-sender",
            daemon=True,
        )
        sender.start()

        try:
            capture = getattr(frame_provider, "capture_stream", None)
            if not callable(capture):
                raise TypeError("frame_provider must provide capture_stream()")
            result.last_frame = capture(
                duration_ms=duration_ms,
                fps=target_fps,
                on_frame=queue_source_frame,
                on_source_frame=queue_source_frame,
                on_no_new_frame=record_no_new_frame,
                should_stop=lambda: network_failed.is_set()
                or (should_stop is not None and should_stop()),
                timeout=10.0,
                request_timeout=15.0,
            ) or result.last_frame
        except Exception as exc:
            result.error = str(exc)
            self._log("error", f"TFT 预览摄像头错误: {exc}")
        finally:
            with slot_condition:
                capture_done = True
                slot_condition.notify_all()
            sender.join(timeout=2.0)
            if sender.is_alive():
                result.error = result.error or "preview_sender_timeout"
                network_failed.set()
                with slot_condition:
                    latest_slot = None
                    slot_condition.notify_all()
                client = self._verified_client()
                if client is not None:
                    self._disconnect_client(client, reason="预览发送超时")
                sender.join(timeout=1.0)
            if not result.connected:
                self._log(
                    "warn",
                    "拍照/识别已完成，但 WALL_E_TFT 未连接；本地摄像头流程未受影响",
                )
            elapsed_from = stream_started_at or operation_started_at
            result.elapsed_seconds = max(0.0, self._clock() - elapsed_from)
            # Release first: logging must never leave the preview permanently
            # busy, especially on ROS distributions whose logger can reject a
            # severity change from the same Python call site.
            self._stream_lock.release()
            self._log(
                "info",
                "TFT 预览结束: "
                f"source={result.source_frames} encoded={result.encoded_frames} "
                f"sent={result.sent_frames} no_new={result.no_new_frame_skips} "
                f"backpressure_drops={result.backpressure_drops} "
                f"encode_drops={result.encode_drops} bytes={result.total_bytes} "
                f"elapsed={result.elapsed_seconds:.3f}s actual_fps={result.actual_fps:.2f} "
                f"error={result.error or '-'}",
            )
        return result

    def _accept_loop(self) -> None:
        while not self._stop_event.is_set():
            server = self._server_socket
            if server is None:
                return
            try:
                client, address = server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client.settimeout(0.5)
            self._disconnect_client(reason="新连接替换旧连接")
            with self._client_lock:
                self._client_socket = client
                self._client_address = address
                self._device_id = ""
            self._log("info", f"ESP32 已连接: {address[0]}:{address[1]}")
            thread = threading.Thread(
                target=self._client_loop,
                args=(client, address),
                name=f"tft-preview-client-{address[0]}:{address[1]}",
                daemon=True,
            )
            thread.start()

    def _client_loop(self, client: socket.socket, address: tuple[str, int]) -> None:
        reason = "连接关闭"
        try:
            while not self._stop_event.is_set():
                header = self._recv_exact(client, HEADER.size)
                message_type, _flags, sequence, payload_length = decode_header(header)
                if payload_length > MAX_INCOMING_PAYLOAD:
                    raise ProtocolError(
                        f"incoming payload too large: {payload_length} > {MAX_INCOMING_PAYLOAD}"
                    )
                payload = self._recv_exact(client, payload_length) if payload_length else b""
                if message_type == HELLO:
                    try:
                        device_id = payload.decode("ascii")
                    except UnicodeDecodeError:
                        device_id = ""
                    with self._client_lock:
                        if self._client_socket is client:
                            self._device_id = device_id
                    self._log("info", f"ESP32 HELLO device_id={device_id or '<invalid>'}")
                    if device_id != EXPECTED_DEVICE_ID:
                        self._log("warn", f"未识别的 TFT 设备 ID: {device_id or '<invalid>'}")
                elif message_type == PING:
                    self._send_packet(client, PONG, sequence, b"")
                elif message_type == PONG:
                    continue
        except (ConnectionError, OSError, ProtocolError) as exc:
            reason = str(exc)
        finally:
            self._disconnect_client(client, reason=reason, address=address)

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.HEARTBEAT_SECONDS):
            client = self._verified_client()
            if client is None:
                continue
            try:
                self._send_packet(client, PING, self._next_control_sequence(), b"")
            except (ConnectionError, OSError) as exc:
                self._log("warn", f"TFT 心跳发送失败: {exc}")
                self._disconnect_client(client, reason="心跳失败")

    def _verified_client(self) -> socket.socket | None:
        with self._client_lock:
            if self._client_socket is None or self._device_id != EXPECTED_DEVICE_ID:
                return None
            return self._client_socket

    def _send_packet(
        self,
        client: socket.socket | None,
        message_type: int,
        sequence: int,
        payload: bytes,
    ) -> None:
        if client is None:
            raise ConnectionError("WALL_E_TFT is not connected")
        packet = encode_message(message_type, sequence, payload)
        with self._write_lock:
            client.sendall(packet)

    def _recv_exact(self, client: socket.socket, size: int) -> bytes:
        data = bytearray()
        while len(data) < size and not self._stop_event.is_set():
            try:
                chunk = client.recv(size - len(data))
            except socket.timeout:
                continue
            if not chunk:
                raise ConnectionError("peer disconnected")
            data.extend(chunk)
        if len(data) != size:
            raise ConnectionError("connection stopped during message")
        return bytes(data)

    def _disconnect_client(
        self,
        client: socket.socket | None = None,
        *,
        reason: str,
        address: tuple[str, int] | None = None,
    ) -> None:
        with self._client_lock:
            current = self._client_socket
            if client is not None and current is not client:
                return
            self._client_socket = None
            current_address = self._client_address
            self._client_address = None
            self._device_id = ""
        if current is None:
            return
        try:
            current.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            current.close()
        except OSError:
            pass
        peer = address or current_address
        peer_text = f"{peer[0]}:{peer[1]}" if peer else "unknown"
        self._log("info", f"ESP32 已断开: {peer_text}, reason={reason}")

    def _next_control_sequence(self) -> int:
        with self._sequence_lock:
            self._control_sequence = (self._control_sequence + 1) & 0xFFFFFFFF
            return self._control_sequence

    def _next_stream_sequence(self) -> int:
        with self._sequence_lock:
            self._stream_sequence += 1
            if self._stream_sequence > 0xFFFF:
                self._stream_sequence = 1
            return self._stream_sequence

    def _log(self, level: str, message: str) -> None:
        logger = self._logger
        if logger is None:
            print(f"[TFT Preview] {message}")
            return
        # rclpy identifies a log caller by its source location and rejects
        # changing severity at that same location.  Dynamic ``method(message)``
        # dispatch put every level on one line and triggered
        # "Logger severity cannot be changed between calls" on the RDK image.
        # Keep one stable source line per severity.
        try:
            if level == "error":
                logger.error(message)
            elif level in {"warn", "warning"}:
                method = getattr(logger, "warning", None) or getattr(logger, "warn", None)
                if callable(method):
                    method(message)
            elif level == "debug":
                method = getattr(logger, "debug", None)
                if callable(method):
                    method(message)
            else:
                logger.info(message)
        except Exception as exc:
            # Logging is diagnostic only; a platform logger bug must not abort
            # capture or leave the preview lock held.
            print(f"[TFT Preview] {message} (logger failed: {exc})")


__all__ = [
    "EXPECTED_DEVICE_ID",
    "HEADER",
    "HELLO",
    "JPEG_FRAME",
    "MAGIC",
    "PING",
    "PONG",
    "PROTOCOL_VERSION",
    "PreviewResult",
    "STREAM_END",
    "STREAM_START",
    "STREAM_START_MESSAGE",
    "TftPreviewServer",
    "TftPreviewSettings",
    "decode_header",
    "encode_message",
    "encode_stream_start",
    "load_tft_preview_settings",
    "prepare_tft_jpeg",
]
