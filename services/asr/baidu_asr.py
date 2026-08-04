"""Baidu realtime WebSocket ASR adapter."""

from __future__ import annotations

import json
import time
import uuid
import wave
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .base import AbstractASR


class BaiduASR(AbstractASR):
    """Recognize a complete WAV file with Baidu's realtime ASR protocol."""

    DEFAULT_URL = "wss://vop.baidu.com/realtime_asr"
    CHUNK_BYTES = 5120  # 160 ms of 16 kHz, mono, 16-bit PCM

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
            # Lazy import keeps Aliyun/Zhipu usable when websocket-client is absent.
            import websocket

            pcm_data = self._read_pcm(wav_path, sample_rate)
            connection = websocket.create_connection(self._connection_url(), timeout=self.timeout)
            connection.send(self._start_message(sample_rate))

            for offset in range(0, len(pcm_data), self.CHUNK_BYTES):
                chunk = pcm_data[offset : offset + self.CHUNK_BYTES]
                connection.send_binary(chunk)
                time.sleep(len(chunk) / (sample_rate * 2))

            connection.send(json.dumps({"type": "FINISH"}))
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
        except Exception as exc:
            print(f"[BaiduASR] 识别失败: {exc}")
            return ""
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass
