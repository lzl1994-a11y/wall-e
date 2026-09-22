"""Baidu realtime WebSocket ASR adapter."""

from __future__ import annotations

import json
import threading
import time
import uuid
import wave
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .base import AbstractASR


class BaiduASR(AbstractASR):
    """Baidu realtime ASR with streaming capture and batch fallback."""

    DEFAULT_URL = "wss://vop.baidu.com/realtime_asr"
    CHUNK_BYTES = 5120  # 160 ms of 16 kHz, mono, 16-bit PCM
    STANDBY_TTL_SEC = 8.0
    WARMUP_WAIT_SEC = 1.0
    supports_streaming = True

    def __init__(
        self,
        app_id: int,
        api_key: str,
        dev_pid: int = 15372,
        cuid: str = "wali-x3",
        url: str = DEFAULT_URL,
        lm_id: int | None = None,
        user: str | None = None,
        timeout: float = 20.0,
    ):
        self.app_id = app_id
        self.api_key = api_key
        self.dev_pid = dev_pid
        self.cuid = cuid
        self.url = url or self.DEFAULT_URL
        self.lm_id = lm_id
        self.user = user
        self.timeout = timeout
        self._stream_lock = threading.Lock()
        self._stream_connection = None
        self._stream_buffer = bytearray()
        self._lifecycle_lock = threading.Lock()
        self._standby_connection = None
        self._standby_created_at = 0.0
        self._warmup_thread = None
        self._closed = False

    def _connection_url(self) -> str:
        parts = urlsplit(self.url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query["sn"] = str(uuid.uuid4())
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))

    def _start_message(self, sample_rate: int) -> str:
        data = {
            "appid": self.app_id,
            "appkey": self.api_key,
            "dev_pid": self.dev_pid,
            "cuid": self.cuid,
            "format": "pcm",
            "sample": sample_rate,
        }
        if self.lm_id is not None:
            data["lm_id"] = self.lm_id
        if self.user:
            data["user"] = self.user
        return json.dumps({"type": "START", "data": data}, ensure_ascii=False)

    def _open_connection(self):
        # Lazy import keeps Aliyun/Zhipu usable when websocket-client is absent.
        import websocket

        return websocket.create_connection(self._connection_url(), timeout=self.timeout)

    @staticmethod
    def _connection_is_open(connection) -> bool:
        if connection is None:
            return False
        connected = getattr(connection, "connected", True)
        try:
            return bool(connected)
        except Exception:
            return True

    @staticmethod
    def _close_connection(connection) -> None:
        if connection is None:
            return
        try:
            connection.close()
        except Exception:
            pass

    @staticmethod
    def _receive_final(connection) -> str:
        while True:
            raw_message = connection.recv()
            if not raw_message:
                raise RuntimeError("Baidu ASR closed before returning a final result")
            if isinstance(raw_message, bytes):
                raw_message = raw_message.decode("utf-8")
            message = json.loads(raw_message)
            error_number = message.get("err_no", 0)
            if error_number != 0:
                raise RuntimeError(
                    f"Baidu ASR error {error_number}: {message.get('err_msg', 'unknown error')}"
                )
            if message.get("type") == "FIN_TEXT":
                result = message.get("result", "")
                return result.strip() if isinstance(result, str) else ""

    @staticmethod
    def _read_pcm(wav_path: str, sample_rate: int) -> bytes:
        with wave.open(str(Path(wav_path)), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            wav_rate = wav_file.getframerate()
            if channels != 1 or sample_width != 2 or wav_rate != 16000 or sample_rate != 16000:
                raise ValueError(
                    "Baidu ASR requires mono, 16-bit, 16 kHz PCM WAV audio "
                    f"(got channels={channels}, width={sample_width}, rate={wav_rate})"
                )
            return wav_file.readframes(wav_file.getnframes())

    def recognize(self, wav_path: str, sample_rate: int = 16000) -> str:
        connection = None
        try:
            pcm_data = self._read_pcm(wav_path, sample_rate)
            connection = self._open_connection()
            connection.send(self._start_message(sample_rate))

            for offset in range(0, len(pcm_data), self.CHUNK_BYTES):
                chunk = pcm_data[offset : offset + self.CHUNK_BYTES]
                connection.send_binary(chunk)
                time.sleep(len(chunk) / (sample_rate * 2))

            connection.send(json.dumps({"type": "FINISH"}))
            return self._receive_final(connection)
        except Exception as exc:
            print(f"[BaiduASR] 识别失败: {exc}")
            return ""
        finally:
            self._close_connection(connection)

    def start_stream(self, sample_rate: int = 16000) -> None:
        if sample_rate != 16000:
            raise ValueError(f"Baidu streaming ASR requires 16 kHz PCM (got {sample_rate})")

        self.cancel_stream()
        connection = None
        last_error = None
        for _attempt in range(2):
            connection = self._take_standby_connection()
            try:
                connection.send(self._start_message(sample_rate))
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                self._close_connection(connection)
                connection = None
        if last_error is not None or connection is None:
            raise RuntimeError(f"Baidu ASR connection failed: {last_error}") from last_error

        with self._stream_lock:
            self._stream_connection = connection
            self._stream_buffer.clear()

    def accept_audio(self, pcm_data: bytes) -> None:
        if not pcm_data:
            return
        if len(pcm_data) % 2:
            raise ValueError("Baidu streaming ASR requires aligned 16-bit PCM bytes")

        with self._stream_lock:
            connection = self._stream_connection
            if connection is None:
                raise RuntimeError("Baidu streaming ASR session is not active")
            self._stream_buffer.extend(pcm_data)
            chunks = []
            while len(self._stream_buffer) >= self.CHUNK_BYTES:
                chunks.append(bytes(self._stream_buffer[: self.CHUNK_BYTES]))
                del self._stream_buffer[: self.CHUNK_BYTES]

        try:
            for chunk in chunks:
                connection.send_binary(chunk)
        except Exception:
            self.cancel_stream()
            raise

    def finish_stream(self) -> str:
        with self._stream_lock:
            connection = self._stream_connection
            if connection is None:
                raise RuntimeError("Baidu streaming ASR session is not active")
            trailing = bytes(self._stream_buffer)
            self._stream_buffer.clear()

        try:
            if trailing:
                connection.send_binary(trailing)
            connection.send(json.dumps({"type": "FINISH"}))
            return self._receive_final(connection)
        finally:
            with self._stream_lock:
                if self._stream_connection is connection:
                    self._stream_connection = None
                    self._stream_buffer.clear()
            self._close_connection(connection)

    def cancel_stream(self) -> None:
        with self._stream_lock:
            connection = self._stream_connection
            self._stream_connection = None
            self._stream_buffer.clear()
        self._close_connection(connection)

    def warmup(self) -> None:
        stale_connection = None
        thread = None
        now = time.monotonic()
        with self._lifecycle_lock:
            if self._closed:
                return
            if (
                self._connection_is_open(self._standby_connection)
                and now - self._standby_created_at < self.STANDBY_TTL_SEC
            ):
                return
            stale_connection = self._standby_connection
            self._standby_connection = None
            self._standby_created_at = 0.0
            if self._warmup_thread is None or not self._warmup_thread.is_alive():
                thread = threading.Thread(
                    target=self._warmup_worker,
                    name="baidu-asr-warmup",
                    daemon=True,
                )
                self._warmup_thread = thread

        self._close_connection(stale_connection)
        if thread is not None:
            thread.start()

    def _warmup_worker(self) -> None:
        connection = None
        try:
            connection = self._open_connection()
        except Exception as exc:
            print(f"[BaiduASR] 待命连接建立失败: {exc}")

        keep_connection = False
        with self._lifecycle_lock:
            self._warmup_thread = None
            if not self._closed and connection is not None and self._standby_connection is None:
                self._standby_connection = connection
                self._standby_created_at = time.monotonic()
                keep_connection = True
        if keep_connection:
            print("[BaiduASR] 待命连接已就绪")
        else:
            self._close_connection(connection)

    def _take_standby_connection(self):
        stale_connection = None
        warmup_thread = None
        now = time.monotonic()
        with self._lifecycle_lock:
            connection = self._standby_connection
            if (
                self._connection_is_open(connection)
                and now - self._standby_created_at < self.STANDBY_TTL_SEC
            ):
                self._standby_connection = None
                self._standby_created_at = 0.0
                return connection
            stale_connection = connection
            self._standby_connection = None
            self._standby_created_at = 0.0
            warmup_thread = self._warmup_thread

        self._close_connection(stale_connection)
        if warmup_thread is not None and warmup_thread.is_alive():
            warmup_thread.join(timeout=min(self.WARMUP_WAIT_SEC, self.timeout))
        with self._lifecycle_lock:
            connection = self._standby_connection
            if self._connection_is_open(connection):
                self._standby_connection = None
                self._standby_created_at = 0.0
                return connection
        return self._open_connection()

    def close(self) -> None:
        with self._lifecycle_lock:
            self._closed = True
            standby_connection = self._standby_connection
            self._standby_connection = None
            self._standby_created_at = 0.0
        self.cancel_stream()
        self._close_connection(standby_connection)
