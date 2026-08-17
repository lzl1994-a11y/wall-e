# services/stt_service.py
"""
[ZH] 语音识别服务
     委托 AudioPipeline 完成音频采集+唤醒词守门+VAD断句，
     本层仅负责 PCM→WAV→ASR适配器→文本回调。
     唤醒词守门：未唤醒时静默丢弃断句，不消耗云端 ASR 配额。
[EN] STT service: delegates audio capture/VAD to AudioPipeline,
     only handles WAV encoding → ASR adapter → text callback.
"""
import os
import tempfile
import threading
import time
import wave
import yaml

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

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        ww_cfg = config.get("wake_word", {})
        self._awake_timeout = ww_cfg.get("awake_timeout", 8.0)

        self._pipe = AudioPipeline(config_path)
        self._pipe.on_sentence = self._on_sentence
        self._pipe.on_speech_start = self._on_speech_start
        self._pipe.on_speech_audio = self._on_speech_audio
        self._pipe.on_speech_cancel = self._on_speech_cancel
        self._pipe.on_wake_word = self._on_wake_word

        # 透传唤醒词回调
        self.on_wake_word = None

        # 唤醒词守门
        self._awake = False
        self._awake_timer = None
        self._awake_lock = threading.Lock()
        self._awake_timer_generation = 0
        self._streaming_active = False
        self._streaming_lock = threading.Lock()

    def _on_wake_word(self):
        """唤醒词触发：进入监听状态。"""
        self._cancel_streaming()
        self._warmup_asr()
        with self._awake_lock:
            self._awake = True
            self._reset_awake_timer()
        print(f"[STT] 唤醒词触发，进入监听 (超时 {self._awake_timeout:.0f}s)")
        if self.on_wake_word:
            self.on_wake_word()

    # ===================================================================
    # Public API
    # ===================================================================
    def start(self):
        self._warmup_asr()
        self._pipe.start()
        print(f"[STT] 语音监听已启动 (适配器: {type(self.asr_adapter).__name__}), 等待唤醒词")

    def stop(self):
        self._cancel_streaming()
        self._pipe.stop()
        self._close_asr()
        with self._awake_lock:
            self._awake = False
            self._cancel_awake_timer_locked()
        print("[STT] 语音监听已停止")

    def pause(self):
        self._cancel_streaming()
        with self._awake_lock:
            self._cancel_awake_timer_locked()
        self._pipe.pause()
        self._warmup_asr()
        print("[STT] 麦克风已暂停，唤醒超时计时已挂起")

    def resume(self):
        self._pipe.resume()
        self._warmup_asr()
        with self._awake_lock:
            if self._awake:
                self._reset_awake_timer()
        print("[STT] 麦克风已恢复，重新开始唤醒超时计时")

    def set_awake(self, value: bool):
        """外部控制唤醒状态。"""
        with self._awake_lock:
            self._awake = value
            if value:
                self._reset_awake_timer()
            else:
                self._cancel_awake_timer_locked()

    # ===================================================================
    # 超时管理
    # ===================================================================
    def _cancel_awake_timer_locked(self):
        self._awake_timer_generation += 1
        if self._awake_timer:
            self._awake_timer.cancel()
        self._awake_timer = None

    def _reset_awake_timer(self):
        self._cancel_awake_timer_locked()
        generation = self._awake_timer_generation
        self._awake_timer = threading.Timer(
            self._awake_timeout,
            self._on_awake_timeout,
            args=(generation,),
        )
        self._awake_timer.daemon = True
        self._awake_timer.start()

    def _on_awake_timeout(self, generation=None):
        with self._awake_lock:
            if generation is not None and generation != self._awake_timer_generation:
                return
            if not self._awake:
                return
            self._awake = False
            self._awake_timer = None
            self._awake_timer_generation += 1
        self._pipe.set_awake(False)
        self._cancel_streaming()
        print(f"[STT] {self._awake_timeout:.0f}s 无语音，退出监听，等待唤醒词")

    # ===================================================================
    # 可选实时 ASR 传输
    # ===================================================================
    def _on_speech_start(self, initial_pcm: bytes):
        if not getattr(self.asr_adapter, "supports_streaming", False):
            return
        with self._awake_lock:
            if not self._awake:
                return
            self._cancel_awake_timer_locked()

        try:
            self.asr_adapter.start_stream(self.SAMPLE_RATE)
            with self._streaming_lock:
                self._streaming_active = True
            self.asr_adapter.accept_audio(initial_pcm)
            print(f"[STT] {type(self.asr_adapter).__name__} 实时传输已开始")
        except Exception as exc:
            print(f"[STT] ASR 实时连接失败，将回退整句识别: {exc}")
            self._cancel_streaming()

    def _on_speech_audio(self, pcm_frame: bytes):
        with self._streaming_lock:
            active = self._streaming_active
        if not active:
            return
        try:
            self.asr_adapter.accept_audio(pcm_frame)
        except Exception as exc:
            print(f"[STT] ASR 实时传输中断，将回退整句识别: {exc}")
            self._cancel_streaming()

    def _on_speech_cancel(self):
        self._cancel_streaming()
        with self._awake_lock:
            if self._awake:
                self._reset_awake_timer()

    def _cancel_streaming(self):
        streaming_lock = getattr(self, "_streaming_lock", None)
        adapter = getattr(self, "asr_adapter", None)
        if streaming_lock is None or adapter is None:
            return
        with streaming_lock:
            active = self._streaming_active
            self._streaming_active = False
        if active or getattr(adapter, "supports_streaming", False):
            try:
                adapter.cancel_stream()
            except Exception:
                pass

    def _warmup_asr(self):
        adapter = getattr(self, "asr_adapter", None)
        if adapter is None:
            return
        try:
            adapter.warmup()
        except Exception as exc:
            print(f"[STT] ASR 预热失败，将在识别时重试: {exc}")

    def _close_asr(self):
        adapter = getattr(self, "asr_adapter", None)
        if adapter is None:
            return
        try:
            adapter.close()
        except Exception as exc:
            print(f"[STT] ASR 关闭异常: {exc}")

    # ===================================================================
    # PCM → WAV/debug → ASR final result
    # ===================================================================
    def _on_sentence(self, pcm_data: bytes):
        """VAD 断句回调：实时会话取最终文本，其他引擎识别完整 WAV。"""
        with self._awake_lock:
            awake = self._awake
        if not awake:
            return  # 未唤醒，静默丢弃

        # 一旦开始识别完整句子，挂起超时；成功时由播放完成事件重新计时。
        with self._awake_lock:
            self._cancel_awake_timer_locked()

        duration_ms = len(pcm_data) // 2 * 1000 // self.SAMPLE_RATE

        wav_path = None
        delivered = False
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

            with self._streaming_lock:
                streaming_active = self._streaming_active

            started = time.monotonic()
            if streaming_active:
                print(f"[STT] 实时音频已发送 {duration_ms}ms，等待 ASR 最终结果...")
                try:
                    text = self.asr_adapter.finish_stream()
                except Exception as exc:
                    print(f"[STT] ASR 最终结果失败，回退整句识别: {exc}")
                    text = self.asr_adapter.recognize(wav_path, self.SAMPLE_RATE)
                finally:
                    with self._streaming_lock:
                        self._streaming_active = False
            else:
                print(f"[STT] 提交语音 {duration_ms}ms 至 ASR...")
                text = self.asr_adapter.recognize(wav_path, self.SAMPLE_RATE)
            print(f"[STT] ASR 耗时 {time.monotonic() - started:.2f}s")

            if text and self.on_sentence_received:
                print(f"[STT] {text}")
                self.on_sentence_received(text)
                delivered = True
            else:
                print("[STT] ASR 未识别出文字")

        except Exception as e:
            print(f"[STT] ASR 识别失败: {e}")
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
            if not delivered:
                with self._awake_lock:
                    if self._awake:
                        self._reset_awake_timer()
