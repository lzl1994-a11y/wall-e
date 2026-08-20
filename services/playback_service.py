# services/playback_service.py
"""音频播放服务：PCM int16 → sounddevice 播放 → USB / I2S 切换。

不关心音频来源（TTS / WAV / 其他），只管播放。
"""

import threading
import queue

import numpy as np
import sounddevice as sd

from services.audio_output import OUTPUT_SAMPLE_RATE
from services.usb_devices import DEFAULT_CONFIG_PATH, resolve_audio_device


class PlaybackService:
    """音频播放器：后台线程顺序播放，支持 USB / 板载切换。"""

    _TURN_END = object()
    TURN_END_SILENCE_SEC = 0.1
    IDLE_SILENCE_SEC = 0.02

    def __init__(
        self,
        mode="default",
        sample_rate=OUTPUT_SAMPLE_RATE,
        config_path=DEFAULT_CONFIG_PATH,
        on_turn_complete=None,
    ):
        self.mode = mode
        self.sample_rate = sample_rate
        self.config_path = config_path
        self.on_turn_complete = on_turn_complete
        self._device = None
        self._device_identity = ""
        self._stream = None
        self._refresh_device()

        self._queue = queue.Queue()
        self._worker = threading.Thread(target=self._play_worker, daemon=True)
        self._worker.start()

        print(f"[Playback Service] 播放器就绪 (mode={mode}, device={self._device}, sr={sample_rate})")

    def _select_device(self):
        """根据 mode 选择 sounddevice 输出设备 ID。

        ESP32-S3 UAC 设备同时提供输入（麦克风）和输出（喇叭），
        共用同一个 PortAudio 设备索引。VoiceChatService 的 InputStream
        占用该设备后，播放也必须用精确索引，不能用 -1（None）。
        """
        devices = sd.query_devices()
        # 优先找同时有输入和输出的设备（UAC）
        for idx, dev in enumerate(devices):
            if dev["max_input_channels"] > 0 and dev["max_output_channels"] > 0:
                print(f"[Playback Service] UAC 音频设备: [{idx}] {dev['name']}")
                return idx

        # 兜底：用系统默认输入设备索引（VoiceChatService 占的那个）
        try:
            default_dev = sd.query_devices(kind="input")
            for idx, dev in enumerate(devices):
                if dev["name"] == default_dev["name"]:
                    print(f"[Playback Service] 默认输入设备: [{idx}] {dev['name']}")
                    return idx
        except Exception:
            pass

        print("[Playback Service] 未找到音频设备，回退到 None")
        return None

    def _refresh_device(self):
        resolution = resolve_audio_device(
            "output", self.config_path, sounddevice_module=sd
        )
        if resolution.configured:
            self._device = resolution.index if resolution.available else None
            self._device_identity = resolution.identity
            return resolution.available
        self._device = self._select_device()
        self._device_identity = f"legacy:{self._device}"
        return True

    def play(self, samples: np.ndarray):
        """入队播放 PCM int16 数组（48kHz mono）。"""
        if samples is None or len(samples) == 0:
            return
        self._queue.put(samples)

    def mark_turn_end(self):
        """在此前入队的音频全部播放后触发本轮完成回调。"""
        self._queue.put(self._TURN_END)

    def _play_item(self, item):
        if item is self._TURN_END:
            try:
                self._write_silence(self.TURN_END_SILENCE_SEC)
                self._close_stream(drain=True)
            finally:
                if self.on_turn_complete:
                    self.on_turn_complete()
            return

        if not self._ensure_stream():
            return
        audio = item.astype(np.float32) / 32768.0
        underflowed = self._stream.write(audio.reshape(-1, 1))
        if underflowed is True:
            print("[Playback Service] 输出缓冲欠载，音频数据到达速度低于播放速度")

    def _write_silence(self, duration_sec):
        """Make the last UAC packet digital silence before stopping the stream."""
        if self._stream is None:
            return
        frame_count = max(1, int(round(self.sample_rate * float(duration_sec))))
        self._stream.write(np.zeros((frame_count, 1), dtype=np.float32))

    def _ensure_stream(self):
        if self._stream is not None:
            return True
        if not self._refresh_device():
            print("[Playback Service] configured voice USB is offline; audio skipped")
            return False
        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                device=self._device,
                channels=1,
                dtype="float32",
                blocksize=0,
            )
            self._stream.start()
            return True
        except Exception as exc:
            self._stream = None
            print(f"[Playback Service] 打开音频流失败: {exc}")
            return False

    def _close_stream(self, drain=False):
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        try:
            if drain:
                stream.stop()
            else:
                stream.abort()
        finally:
            stream.close()

    def _play_worker(self):
        """后台线程：顺序播放；流已打开但暂时断粮时持续输出数字静音。"""
        while True:
            self._play_next_item()

    def _play_next_item(self):
        """播放一个队列项，或在等待超时时向已打开的流补一小段静音。"""
        try:
            item = self._queue.get(timeout=self.IDLE_SILENCE_SEC)
        except queue.Empty:
            try:
                self._write_silence(self.IDLE_SILENCE_SEC)
            except Exception as exc:
                print(f"[Playback Service] 静音填充失败: {exc}")
                self._close_stream(drain=False)
            return

        try:
            self._play_item(item)
        except Exception as e:
            print(f"[Playback Service] 播放失败: {e}")
            self._close_stream(drain=False)
        finally:
            self._queue.task_done()
