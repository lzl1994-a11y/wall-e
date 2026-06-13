# services/playback_service.py
"""音频播放服务：PCM int16 → sounddevice 播放 → USB / I2S 切换。

不关心音频来源（TTS / WAV / 其他），只管播放。
"""

import threading
import queue

import numpy as np
import sounddevice as sd


class PlaybackService:
    """音频播放器：后台线程顺序播放，支持 USB / 板载切换。"""

    def __init__(self, mode="default", sample_rate=16000):
        self.mode = mode
        self.sample_rate = sample_rate
        self._device = self._select_device()

        self._queue = queue.Queue()
        self._worker = threading.Thread(target=self._play_worker, daemon=True)
        self._worker.start()

        print(f"[Playback Service] 播放器就绪 (mode={mode}, device={self._device}, sr={sample_rate})")

    def _select_device(self):
        """根据 mode 选择 sounddevice 输出设备 ID。"""
        if self.mode != "usb":
            return None  # 系统默认

        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if dev["max_input_channels"] > 0 and dev["max_output_channels"] > 0:
                print(f"[Playback Service] USB 音频设备: [{idx}] {dev['name']}")
                return idx

        print("[Playback Service] 未找到 USB 音频设备，回退到 default")
        return None

    def play(self, samples: np.ndarray):
        """入队播放 PCM int16 数组（16kHz mono）。"""
        if samples is None or len(samples) == 0:
            return
        self._queue.put(samples)

    def _play_worker(self):
        """后台线程：阻塞播放音频队列。"""
        while True:
            samples = self._queue.get()
            try:
                audio = samples.astype(np.float32) / 32768.0
                sd.play(audio, samplerate=self.sample_rate, device=self._device)
                sd.wait()
            except Exception as e:
                print(f"[Playback Service] 播放失败: {e}")
