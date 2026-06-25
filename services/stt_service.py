# services/stt_service.py
"""
[ZH] 语音识别服务
     委托 AudioPipeline 完成音频采集+唤醒词守门+VAD断句，
     本层仅负责 PCM→WAV→ASR适配器→文本回调。
[EN] STT service: delegates audio capture/VAD to AudioPipeline,
     only handles WAV encoding → ASR adapter → text callback.
"""
import os
import tempfile
import wave
import numpy as np

from .asr import create_asr
from .audio_pipeline import AudioPipeline


class STTService:
    """
    外部接口保持兼容:
        stt = STTService(config_path, on_sentence_received=callback)
        stt.on_wake_word = lambda: play_wav()
        stt.start()   # 开始监听
        stt.stop()    # 停止监听
        stt.pause()   # 机器人说话时暂停（防回声）
        stt.resume()  # 恢复监听
    """

    SAMPLE_RATE = AudioPipeline.SAMPLE_RATE

    def __init__(self, config_path="core/config.yaml", on_sentence_received=None):
        self.on_sentence_received = on_sentence_received
        self.asr_adapter = create_asr(config_path)

        self._pipe = AudioPipeline(config_path)
        self._pipe.on_sentence = self._on_sentence

        # 透传唤醒词回调
        self.on_wake_word = None
        self._pipe.on_wake_word = self._on_wake_word

    def _on_wake_word(self):
        if self.on_wake_word:
            self.on_wake_word()

    # ===================================================================
    # Public API
    # ===================================================================
    def start(self):
        self._pipe.start()
        print(f"[STT] 语音监听已启动 (适配器: {type(self.asr_adapter).__name__})")
        return True

    def stop(self):
        self._pipe.stop()
        print("[STT] 语音监听已停止")

    def pause(self):
        self._pipe.pause()
        print("[STT] 麦克风已暂停 (防回声)")

    def resume(self):
        self._pipe.resume()
        print("[STT] 麦克风已恢复")

    # ===================================================================
    # PCM → WAV → ASR
    # ===================================================================
    def _on_sentence(self, pcm_data: bytes):
        """AudioPipeline 断句回调：PCM bytes → WAV → ASR → 文本回调。"""
        duration_ms = len(pcm_data) // 2 * 1000 // self.SAMPLE_RATE

        wav_path = None
        try:
            fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="stt_")
            os.close(fd)
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.SAMPLE_RATE)
                wf.writeframes(pcm_data)

            # 调试副本
            import shutil
            debug_path = os.path.expanduser("~/.wali_debug/stt_debug_last.wav")
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            shutil.copy2(wav_path, debug_path)
            print(f"[STT] 调试音频: {debug_path} ({duration_ms}ms)")

            print(f"[STT] 上传语音 {duration_ms}ms 至云端...")
            text = self.asr_adapter.recognize(wav_path, self.SAMPLE_RATE)

            if text and self.on_sentence_received:
                print(f"[STT] {text}")
                self.on_sentence_received(text)
            else:
                print("[STT] 云端未识别出文字")

        except Exception as e:
            print(f"[STT] 云端识别失败: {e}")
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
