# services/tts_service.py
"""TTS 合成服务：文本 → edge-tts → PCM int16 numpy array。

纯合成，不负责播放。播放由 playback_service 统一管理。
"""

import asyncio
from concurrent.futures import CancelledError
import io
import queue
import subprocess
import threading
import time

import edge_tts
import numpy as np

from services.audio_output import (
    OUTPUT_CHANNELS,
    OUTPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_WIDTH,
)


class TTSService:
    """Edge-TTS 合成器：文本 → PCM int16 48kHz mono。"""

    def __init__(
        self,
        voice="zh-CN-YunxiaNeural",
        rate="+20%",
        pitch="+5Hz",
        sample_rate=OUTPUT_SAMPLE_RATE,
    ):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        self.sample_rate = sample_rate

        # 后台 event loop（edge-tts 需要异步调用）
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        print(
            f"[TTS Service] 合成器就绪 "
            f"(voice={voice}, rate={rate}, pitch={pitch}, sr={sample_rate})"
        )

    def synthesize(self, text: str) -> np.ndarray:
        """同步接口：文本 → PCM int16 数组（48kHz mono）。"""
        if not text or not text.strip():
            raise ValueError("text is empty")

        future = asyncio.run_coroutine_threadsafe(
            self._download(text), self._loop
        )
        return future.result()

    def synthesize_stream(
        self,
        text: str,
        chunk_ms: int = 100,
        idle_timeout_sec: float = 2.0,
    ):
        """Yield 48 kHz PCM while Edge TTS is still producing MP3 data."""
        if not text or not text.strip():
            raise ValueError("text is empty")

        chunk_ms = max(20, int(chunk_ms))
        idle_timeout_sec = max(0.1, float(idle_timeout_sec))
        chunk_bytes = max(2, self.sample_rate * OUTPUT_SAMPLE_WIDTH * chunk_ms // 1000)
        mp3_queue = queue.Queue()
        mp3_sentinel = object()
        pcm_queue = queue.Queue()
        pcm_sentinel = object()
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "mp3",
                "-i",
                "pipe:0",
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-f",
                "s16le",
                "-ar",
                str(self.sample_rate),
                "-ac",
                str(OUTPUT_CHANNELS),
                "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            process.kill()
            raise RuntimeError("ffmpeg streaming pipes could not be created")

        producer = asyncio.run_coroutine_threadsafe(
            self._download_to_queue(text, mp3_queue, mp3_sentinel),
            self._loop,
        )
        writer_errors = []
        reader_errors = []
        last_activity = [time.monotonic()]

        def feed_decoder():
            try:
                while True:
                    data = mp3_queue.get()
                    if data is mp3_sentinel:
                        break
                    last_activity[0] = time.monotonic()
                    process.stdin.write(data)
            except Exception as exc:
                writer_errors.append(exc)
            finally:
                try:
                    process.stdin.close()
                except Exception:
                    pass

        def read_decoder():
            try:
                while True:
                    data = process.stdout.read(chunk_bytes)
                    if not data:
                        break
                    last_activity[0] = time.monotonic()
                    pcm_queue.put(data)
            except Exception as exc:
                reader_errors.append(exc)
            finally:
                pcm_queue.put(pcm_sentinel)

        writer = threading.Thread(
            target=feed_decoder,
            name="edge-tts-ffmpeg-writer",
            daemon=True,
        )
        reader = threading.Thread(
            target=read_decoder,
            name="edge-tts-ffmpeg-reader",
            daemon=True,
        )
        writer.start()
        reader.start()

        yielded_audio = False
        received_pcm = False
        pending = bytearray()
        completed = False
        idle_timed_out = False
        flush_deadline = None
        try:
            while True:
                if flush_deadline is not None:
                    wait_timeout = max(0.0, flush_deadline - time.monotonic())
                    if wait_timeout == 0.0:
                        raise RuntimeError("ffmpeg did not flush after TTS idle timeout")
                elif received_pcm:
                    idle_remaining = idle_timeout_sec - (
                        time.monotonic() - last_activity[0]
                    )
                    wait_timeout = max(0.01, idle_remaining)
                else:
                    wait_timeout = None

                try:
                    data = pcm_queue.get(timeout=wait_timeout)
                except queue.Empty:
                    if flush_deadline is not None:
                        raise RuntimeError("ffmpeg did not flush after TTS idle timeout")
                    idle_for = time.monotonic() - last_activity[0]
                    if idle_for < idle_timeout_sec:
                        continue
                    idle_timed_out = True
                    flush_deadline = time.monotonic() + 5.0
                    print(
                        f"[TTS Service] 流式音频连续 {idle_for:.2f}s 无新数据，"
                        "主动关闭输入并冲刷尾帧"
                    )
                    producer.cancel()
                    mp3_queue.put_nowait(mp3_sentinel)
                    continue

                if data is pcm_sentinel:
                    break
                received_pcm = True
                pending.extend(data)
                while len(pending) >= chunk_bytes:
                    yielded_audio = True
                    samples = np.frombuffer(
                        bytes(pending[:chunk_bytes]), dtype=np.int16
                    ).copy()
                    del pending[:chunk_bytes]
                    yield samples

            usable = len(pending) - (len(pending) % OUTPUT_SAMPLE_WIDTH)
            if usable:
                yielded_audio = True
                yield np.frombuffer(bytes(pending[:usable]), dtype=np.int16).copy()

            writer.join(timeout=5.0)
            if writer.is_alive():
                raise RuntimeError("ffmpeg input writer did not finish")
            reader.join(timeout=1.0)
            if reader.is_alive():
                raise RuntimeError("ffmpeg output reader did not finish")
            try:
                producer.result(timeout=5.0)
            except CancelledError:
                if not idle_timed_out:
                    raise
            return_code = process.wait(timeout=5.0)
            stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
            if writer_errors:
                raise RuntimeError(f"ffmpeg input failed: {writer_errors[0]}")
            if reader_errors:
                raise RuntimeError(f"ffmpeg output failed: {reader_errors[0]}")
            if return_code != 0:
                raise RuntimeError(stderr or f"ffmpeg exited with code {return_code}")
            if not yielded_audio:
                raise RuntimeError("edge-tts streaming returned empty audio")
            completed = True
        finally:
            if not completed:
                producer.cancel()
                try:
                    mp3_queue.put_nowait(mp3_sentinel)
                except Exception:
                    pass
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                writer.join(timeout=1.0)
                reader.join(timeout=1.0)

    async def _download_to_queue(self, text, output_queue, sentinel):
        communicate = edge_tts.Communicate(
            text, self.voice, rate=self.rate, pitch=self.pitch
        )
        received_audio = False
        try:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio" and chunk.get("data"):
                    received_audio = True
                    output_queue.put_nowait(chunk["data"])
            if not received_audio:
                raise RuntimeError("edge-tts 返回空音频")
        finally:
            output_queue.put_nowait(sentinel)

    async def _download(self, text: str) -> np.ndarray:
        """异步下载 MP3 → pydub 解码 → PCM int16 48kHz mono。"""
        communicate = edge_tts.Communicate(
            text, self.voice, rate=self.rate, pitch=self.pitch
        )
        mp3_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_data += chunk["data"]

        if not mp3_data:
            raise RuntimeError("edge-tts 返回空音频")

        from pydub import AudioSegment
        seg = AudioSegment.from_mp3(io.BytesIO(mp3_data))
        pcm = (
            seg.set_frame_rate(self.sample_rate)
            .set_channels(OUTPUT_CHANNELS)
            .set_sample_width(OUTPUT_SAMPLE_WIDTH)
        )
        return np.array(pcm.get_array_of_samples(), dtype=np.int16)

    def shutdown(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2)
        if not self._loop.is_running() and not self._loop.is_closed():
            self._loop.close()
