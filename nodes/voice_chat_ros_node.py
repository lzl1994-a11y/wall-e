#!/usr/bin/env python3
"""语音直聊 ROS2 节点：唤醒词 → 语音应答 → VAD → Qwen-Omni → TTS

状态机由 VoiceChatService 驱动，本节点负责 ROS 侧回调：
  on_wake_word   → 播放预合成 WAV + TFT 切聊天页
  on_llm_chunk   → 流式文本块，2 标点攒一句 → tts_text
  on_llm_reply   → 最终完整回复 → screen_dialog
  on_tool_call   → /action_cmd
  on_llm_timeout → TFT 切待机页 + 日志
"""

import base64
import json
import os
import random
import re
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, UInt8MultiArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.voice_chat_service import VoiceChatService
from services.camera_frame import CameraFrameProvider, save_camera_photo
from services.game_protocol import (
    GAME_FRAME_TOPIC,
    GAME_MODE_REQUEST_TOPIC,
    GAME_MODE_STATE_TOPIC,
    GAME_SURFACE_READY,
    decode_game_frame,
    encode_game_request,
    game_mode_from_message,
)
from services.game_tft_stream import GameTftStreamServer, prepare_game_bgr
from services.audio_output import (
    OUTPUT_CHANNELS,
    OUTPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_WIDTH,
)
from services.tool_dispatcher import build_action_cmd
from services.tft_preview_server import load_tft_preview_settings
from services.tracking_tft_preview import TrackingTftPreview
from services.tts_protocol import encode_turn_end
from services.dialog_motion_protocol import (
    DIALOG_MOTION_VAD_TOPIC,
    VAD_SPEECH_ENDED,
    VAD_SPEECH_STARTED,
)
from services.usb_devices import resolve_audio_device
from services.vision_pipeline_protocol import (
    VISION_PIPELINE_COMMAND_TOPIC,
    decode_vision_pipeline_command,
)

# 去掉 TTS 不需要的符号（保留中文标点和空格）
TTS_CLEAN_RE = re.compile(r'[*#_~`>\[\]\(\)\{\}]')
OUTPUT_ECHO_GUARD_SECONDS = 0.35

class VoiceChatNode(Node):
    def __init__(self):
        super().__init__("voice_chat_node")

        self.tts_pub = self.create_publisher(String, "tts_text", 10)
        self.dialog_pub = self.create_publisher(String, "screen_dialog", 10)
        self.action_pub = self.create_publisher(String, "action_cmd", 10)
        self.game_busy_pub = self.create_publisher(String, "llm_busy", 10)
        self.dialog_motion_pub = self.create_publisher(
            String, DIALOG_MOTION_VAD_TOPIC, 10
        )
        self.game_request_pub = self.create_publisher(String, GAME_MODE_REQUEST_TOPIC, 10)
        self.create_subscription(String, "llm_busy", self._on_playback_state, 10)

        self.camera_frames = CameraFrameProvider(self)
        self.tft_preview_ready_pub = self.create_publisher(
            String, "tft_preview_ready", 10
        )
        self.tft_preview_settings = load_tft_preview_settings()
        self.tft_preview = GameTftStreamServer(
            self.tft_preview_settings,
            logger=self.get_logger(),
        )
        try:
            self.tft_preview.start()
            self._publish_tft_preview_ready()
            self.tft_preview_ready_timer = self.create_timer(
                1.0, self._publish_tft_preview_ready
            )
        except Exception as exc:
            # Image analysis still works without a connected chest screen.
            self.get_logger().error(f"TFT preview service failed to start: {exc}")
        self.tracking_tft_preview = TrackingTftPreview(
            self.tft_preview,
            self.camera_frames,
            fps=self.tft_preview_settings.fps,
            logger=self.get_logger(),
        )
        self.create_subscription(
            String,
            VISION_PIPELINE_COMMAND_TOPIC,
            self._on_vision_pipeline_command,
            10,
        )
        self._game_mode = "robot"
        self._game_stream = None
        self._game_frame_adapter = None
        self._game_frame_lock = threading.Lock()
        self._latest_game_frame = None
        self._next_game_commentary = None
        self._game_commentary_running = False
        self._tracking_was_enabled = False
        self.create_subscription(String, GAME_MODE_STATE_TOPIC, self._on_game_state, 10)
        self.create_subscription(UInt8MultiArray, GAME_FRAME_TOPIC, self._on_game_frame, 1)
        self.create_timer(1.0, self._game_commentary_tick)

        self.get_logger().info("正在预热唤醒词 + Qwen-Omni 引擎...")

        self.vc = VoiceChatService()
        self.vc.on_wake_word = self._on_wake_word
        self.vc.on_speech_start = self._on_vad_speech_start
        self.vc.on_speech_end = self._on_vad_speech_end
        self.vc.on_llm_chunk = self._on_llm_chunk
        self.vc.on_llm_reply = self._on_llm_reply
        self.vc.on_tool_call = self._on_tool_call
        self.vc.on_photo_request = self._process_camera_photo
        self.vc.on_inspection_request = self._process_heard_camera_inspection
        self.vc.on_llm_done = self._on_llm_done
        self.vc.on_llm_timeout = self._on_llm_timeout

        # 流式 TTS 状态
        self._sentence_buffer = ""     # 当前攒的句子
        self._punc_count = 0           # 标点计数
        self._correction_done = False  # 第一行纠错已提取
        self._active_turn_id = None
        self._output_state_lock = threading.Lock()
        self._awaiting_tts_playback = False
        self._wake_response_active = False
        self._resume_timer = None
        self.punctuations = {"。", "？", ".", "?", "！", "!"}

        # 唤醒应答 WAV 路径
        root = Path(__file__).resolve().parent.parent
        self._wake_wav = str(root / "assets" / "wake_response.wav")
        self._wake_play_lock = threading.Lock()

        self.vc.start()
        self.get_logger().info("语音直聊节点已上线")

    def _publish_tft_preview_ready(self):
        ready = String()
        ready.data = "ready"
        self.tft_preview_ready_pub.publish(ready)

    def _on_vision_pipeline_command(self, message):
        command = decode_vision_pipeline_command(message.data)
        if command is not None:
            self.tracking_tft_preview.set_command(command)

    def _on_game_state(self, message):
        mode = game_mode_from_message(message.data)
        if mode is None:
            return
        previous = self._game_mode
        self._game_mode = mode
        if mode != "robot":
            if previous == "robot":
                self.vc.pause()
                self._tracking_was_enabled = self.tracking_tft_preview.pause()
            self._ensure_game_stream()
            if mode == "playing" and previous != "playing":
                self._schedule_next_game_commentary()
            return
        if previous == "robot":
            return
        self._close_game_stream()
        with self._game_frame_lock:
            self._latest_game_frame = None
        self._next_game_commentary = None
        if self._tracking_was_enabled:
            self.tracking_tft_preview.resume()
        self._tracking_was_enabled = False
        self.vc.resume()

    def _ensure_game_stream(self):
        if self._game_frame_adapter is not None:
            return
        from services.game_frame_adapter import GameFrameAdapter

        stream = self.tft_preview.open_jpeg_stream(fps=10)
        if stream is None:
            return
        self._game_stream = stream
        self._game_frame_adapter = GameFrameAdapter(stream, fps=10)
        self.game_request_pub.publish(String(data=encode_game_request(GAME_SURFACE_READY)))

    def _close_game_stream(self):
        adapter = self._game_frame_adapter
        self._game_frame_adapter = None
        if adapter is not None:
            adapter.close()
        stream = self._game_stream
        self._game_stream = None
        if stream is not None:
            stream.close()

    def _on_game_frame(self, message):
        if self._game_mode == "robot":
            return
        frame = decode_game_frame(bytes(message.data))
        if frame is None:
            return
        raw, width, height, pitch = frame
        with self._game_frame_lock:
            self._latest_game_frame = frame
        adapter = self._game_frame_adapter
        if adapter is not None:
            adapter.submit_frame(raw, width, height, pitch)

    def _schedule_next_game_commentary(self):
        self._next_game_commentary = time.monotonic() + random.uniform(50.0, 120.0)

    def _game_commentary_tick(self):
        if self._game_mode != "playing" or self._game_commentary_running:
            return
        if self._next_game_commentary is None:
            self._schedule_next_game_commentary()
            return
        if time.monotonic() < self._next_game_commentary:
            return
        with self._game_frame_lock:
            frame = self._latest_game_frame
        self._schedule_next_game_commentary()
        if frame is None:
            return
        import numpy as np

        raw, width, height, pitch = frame
        image = np.frombuffer(raw, dtype=np.uint8).reshape(height, pitch // 4, 4)
        jpeg = prepare_game_bgr(image[:, :width, :3], quality=75)
        if not jpeg:
            return
        self._game_commentary_running = True
        threading.Thread(
            target=self._run_game_commentary,
            args=(jpeg,),
            name="game-vision-commentary",
            daemon=True,
        ).start()

    def _run_game_commentary(self, jpeg):
        self.game_busy_pub.publish(String(data="busy"))
        try:
            answer = self.vc.analyze_image(
                "观察当前 FC 游戏画面，以瓦力的口吻说一句简短自然的中文评论。"
                "可以提醒危险、鼓励玩家或描述关键局面；看不清时不要猜。",
                base64.b64encode(jpeg).decode("ascii"),
            )
            answer = TTS_CLEAN_RE.sub("", str(answer or "")).strip()
            if answer and self._game_mode == "playing":
                self.tts_pub.publish(String(data=answer))
                self._active_turn_id = "game-" + uuid.uuid4().hex[:8]
                self._on_llm_done()
            else:
                self.game_busy_pub.publish(String(data="idle"))
        except Exception as exc:
            self.get_logger().error(f"游戏画面识别失败: {exc}")
            self.game_busy_pub.publish(String(data="idle"))
        finally:
            self._game_commentary_running = False

    def _run_camera_preview(self, *, duration_ms):
        tracking_preview = getattr(self, "tracking_tft_preview", None)
        was_tracking = tracking_preview.pause() if tracking_preview is not None else False
        try:
            return self.tft_preview.send_camera_preview(
                self.camera_frames,
                duration_ms=duration_ms,
                hold_ms=self.tft_preview_settings.hold_ms,
                fps=self.tft_preview_settings.fps,
            )
        finally:
            if was_tracking:
                tracking_preview.resume()

    # ── 唤醒词回调 ──
    def _on_wake_word(self):
        """唤醒词触发：播放预合成语音 + 切 TFT 到聊天页。"""
        self.get_logger().info("唤醒词触发")

        # The wake response uses the same speaker as TTS. Mute capture before
        # starting it so the response itself cannot become the user's sentence.
        with self._output_state_lock:
            self._wake_response_active = True
            self._awaiting_tts_playback = False
            if self._resume_timer is not None:
                self._resume_timer.cancel()
                self._resume_timer = None
        self.vc.begin_output_playback()

        # 切 TFT 到聊天页面
        try:
            screen_msg = String()
            screen_msg.data = json.dumps(
                {"page": "chat", "text": "正在听...", "source": "wake_word"},
                ensure_ascii=False,
            )
            self.dialog_pub.publish(screen_msg)
        except Exception:
            pass

        # 播放预合成应答 WAV（后台线程，不阻塞主循环）
        threading.Thread(target=self._play_wake_response, daemon=True).start()

    def _on_vad_speech_start(self):
        if self._game_mode == "robot":
            self.dialog_motion_pub.publish(String(data=VAD_SPEECH_STARTED))

    def _on_vad_speech_end(self):
        if self._game_mode == "robot":
            self.dialog_motion_pub.publish(String(data=VAD_SPEECH_ENDED))

    def _play_wake_response(self):
        """播放 assets/wake_response.wav。"""
        with self._wake_play_lock:
            try:
                if not os.path.exists(self._wake_wav):
                    self.get_logger().warn(f"唤醒应答文件不存在: {self._wake_wav}")
                    self.get_logger().warn("请先运行 generate_wake_response.py 生成语音文件")
                    return

                import sounddevice as sd
                import numpy as np
                from pydub import AudioSegment

                audio = (
                    AudioSegment.from_wav(self._wake_wav)
                    .set_frame_rate(OUTPUT_SAMPLE_RATE)
                    .set_channels(OUTPUT_CHANNELS)
                    .set_sample_width(OUTPUT_SAMPLE_WIDTH)
                )
                if not len(audio):
                    return
                samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
                samples = samples.astype(np.float32) / 32768.0
                resolution = resolve_audio_device("output", sounddevice_module=sd)
                if resolution.configured and not resolution.available:
                    self.get_logger().warn("voice USB offline; wake response skipped")
                    return
                sd.play(
                    samples,
                    samplerate=OUTPUT_SAMPLE_RATE,
                    device=resolution.index,
                )
                sd.wait()
                self.get_logger().info(
                    f"唤醒应答播放完毕 (sr={OUTPUT_SAMPLE_RATE})"
                )
            except ImportError:
                self.get_logger().error("缺少音频播放依赖，无法播放唤醒应答")
            except Exception as e:
                self.get_logger().error(f"播放唤醒应答失败: {e}")
            finally:
                with self._output_state_lock:
                    self._wake_response_active = False
                self._schedule_capture_resume()

    def _on_playback_state(self, msg):
        """Resume multimodal capture only after the queued TTS turn is done."""
        if msg.data != "idle":
            return
        with self._output_state_lock:
            if not self._awaiting_tts_playback:
                return
            self._awaiting_tts_playback = False
            if self._wake_response_active:
                return
        self._schedule_capture_resume()

    def _schedule_capture_resume(self):
        """Discard the speaker's acoustic tail before reopening capture."""
        if getattr(self, "_game_mode", "robot") != "robot":
            return
        with self._output_state_lock:
            if self._resume_timer is not None:
                self._resume_timer.cancel()
            self._resume_timer = threading.Timer(
                OUTPUT_ECHO_GUARD_SECONDS,
                self._resume_capture_after_output,
            )
            self._resume_timer.daemon = True
            self._resume_timer.start()

    def _resume_capture_after_output(self):
        if getattr(self, "_game_mode", "robot") != "robot":
            return
        with self._output_state_lock:
            self._resume_timer = None
            if self._wake_response_active or self._awaiting_tts_playback:
                return
        if self.vc.complete_output_playback():
            self.get_logger().info("扬声器尾音已清除，恢复多模态录音")

    # ── LLM 回调 ──
    def _on_tool_call(self, name, arguments):
        if name == "inspect_camera":
            return self._process_camera_inspection(arguments)
        payload = build_action_cmd(name, arguments)
        msg = String()
        msg.data = payload
        self.action_pub.publish(msg)
        self.get_logger().info(f"Tool: {name}({arguments})")
        return None

    def _process_camera_inspection(self, arguments):
        """Voice-selected camera inspection: preview, capture, then analyze."""
        question = "看看当前画面"
        if isinstance(arguments, dict):
            value = arguments.get("question")
            if isinstance(value, str) and value.strip():
                question = value.strip()

        self.tts_pub.publish(String(data="好的，我看一下。"))
        preview = self._run_camera_preview(
            duration_ms=self.tft_preview_settings.recognition_duration_ms,
        )
        if preview.busy:
            return "我正在处理上一张画面，等一下再看。"
        if not preview.last_frame:
            self.get_logger().error(
                f"Camera inspection failed: {preview.error or 'camera_frame_unavailable'}"
            )
            return "我现在看不到画面，检查一下摄像头连接。"

        try:
            image_base64 = base64.b64encode(preview.last_frame).decode("ascii")
            return self.vc.analyze_image(question, image_base64)
        except Exception as exc:
            self.get_logger().error(
                f"Vision request failed: {exc}\n{traceback.format_exc()}"
            )
            return "这张图我没分析出来，你换个角度再让我看看。"

    def _process_heard_camera_inspection(self, heard_text):
        """Run inspection selected from the structured audio transcript."""
        return self._process_camera_inspection({"question": heard_text})

    def _process_camera_photo(self):
        """Voice-selected photo request: capture and save without vision LLM."""
        self.tts_pub.publish(String(data="好的，准备拍照。"))
        preview = self._run_camera_preview(
            duration_ms=self.tft_preview_settings.photo_duration_ms,
        )
        if preview.busy:
            return "我正在拍上一张，等一下再试。"
        if not preview.last_frame:
            self.get_logger().error(
                f"Camera photo failed: {preview.error or 'camera_frame_unavailable'}"
            )
            return "我现在拍不到照片，检查一下摄像头连接。"
        try:
            saved = save_camera_photo(
                preview.last_frame,
                self.tft_preview_settings.photo_directory,
            )
            self.get_logger().info(f"Camera photo saved: {saved}")
            return "拍好了，照片已经保存。"
        except Exception as exc:
            self.get_logger().error(f"Camera photo save failed: {exc}")
            return "照片拍到了，但保存失败了。"

    def _on_llm_chunk(self, text):
        """流式文本块：跳过纠错首行，2 标点攒一句 → tts_text。"""
        if not text:
            return

        self._ensure_turn_id()
        self._sentence_buffer += text

        # 跳过第一行（纠错文本前缀）
        if not self._correction_done:
            if "\n" in self._sentence_buffer:
                parts = self._sentence_buffer.split("\n", 1)
                self._sentence_buffer = parts[1] if len(parts) > 1 else ""
                self._correction_done = True
                # 检查新 buffer 里是否已有标点
                self._punc_count = sum(
                    1 for c in self._sentence_buffer if c in self.punctuations
                )
            elif len(self._sentence_buffer) > 60:
                self._correction_done = True
                self._punc_count = sum(
                    1 for c in self._sentence_buffer if c in self.punctuations
                )
            else:
                return

        # 按标点累积
        for char in text:
            if char in self.punctuations:
                self._punc_count += 1

        if self._punc_count >= 2:
            clean = self._sentence_buffer.strip()
            tts_safe = TTS_CLEAN_RE.sub("", clean)
            if tts_safe.strip():
                msg = String()
                msg.data = tts_safe.strip()
                self.tts_pub.publish(msg)
                self.get_logger().info(f"TTS: {tts_safe.strip()[:80]}")
            self._sentence_buffer = ""
            self._punc_count = 0

    def _on_llm_reply(self, text):
        """最终完整回复 → 解析 you/ai → screen_dialog。"""
        text = text.strip()
        if not text:
            return

        turn_id = self._ensure_turn_id()

        # 解析 you: / ai: 格式
        corrected_text = ""
        ai_text = text
        if text.startswith("you:"):
            lines = text.split("\n", 1)
            corrected_text = lines[0][4:].strip()
            ai_text = lines[1].strip() if len(lines) > 1 else ""
            if ai_text.startswith("ai:"):
                ai_text = ai_text[3:].strip()

        # 终端输出：让用户看到自己说了什么
        if corrected_text:
            self.get_logger().info(f"[识别] {corrected_text}")
        self.get_logger().info(f"[回复] {ai_text[:80]}")

        # flush 残留 TTS 文本
        if self._sentence_buffer.strip():
            tts_safe = TTS_CLEAN_RE.sub("", self._sentence_buffer.strip())
            if tts_safe.strip():
                msg = String()
                msg.data = tts_safe.strip()
                self.tts_pub.publish(msg)
                self.get_logger().info(f"TTS tail: {tts_safe.strip()[:80]}")

        # 屏幕对话框（对齐 llm_ros_node 格式）
        dialog = String()
        dialog.data = json.dumps({
            "turn_id": turn_id,
            "corrected_text": corrected_text,
            "ai_text": ai_text,
            "actions": [],
            "source": "voice_chat",
        }, ensure_ascii=False)
        self.dialog_pub.publish(dialog)
        self.get_logger().info(f"Screen: {ai_text[:60]}")

        # 重置流式状态
        self._sentence_buffer = ""
        self._punc_count = 0
        self._correction_done = False

    def _ensure_turn_id(self):
        if self._active_turn_id is None:
            self._active_turn_id = uuid.uuid4().hex[:12]
        return self._active_turn_id

    def _on_llm_done(self):
        """关闭本轮 TTS；播放节点会在音频真正播完后结束回合。"""
        turn_id = self._ensure_turn_id()
        with self._output_state_lock:
            self._awaiting_tts_playback = True
        self.tts_pub.publish(String(data=encode_turn_end(turn_id)))
        self.get_logger().info(f"TTS turn queued: {turn_id}")
        self._sentence_buffer = ""
        self._punc_count = 0
        self._correction_done = False
        self._active_turn_id = None

    # ── 超时回调 ──
    def _on_llm_timeout(self):
        """40s 无 LLM 回复，TFT 切回待机。"""
        self.get_logger().info("LLM 超时，切回待机")
        try:
            screen_msg = String()
            screen_msg.data = json.dumps(
                {"page": "idle", "text": "说「瓦力瓦力」唤醒我", "source": "timeout"},
                ensure_ascii=False,
            )
            self.dialog_pub.publish(screen_msg)
        except Exception:
            pass

    def destroy_node(self):
        self.get_logger().info("正在关闭语音直聊节点...")
        with self._output_state_lock:
            if self._resume_timer is not None:
                self._resume_timer.cancel()
                self._resume_timer = None
        if hasattr(self, "vc"):
            self.vc.stop()
        self._close_game_stream()
        if getattr(self, "tracking_tft_preview", None) is not None:
            self.tracking_tft_preview.stop()
        if getattr(self, "tft_preview", None) is not None:
            self.tft_preview.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VoiceChatNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
