#!/usr/bin/env python3
"""语音直聊 ROS2 节点：唤醒词 → 语音应答 → VAD → Qwen-Omni → TTS

状态机由 VoiceChatService 驱动，本节点负责 ROS 侧回调：
  on_wake_word   → 播放预合成 WAV + TFT 切聊天页
  on_llm_chunk   → 流式文本块，2 标点攒一句 → tts_text
  on_llm_reply   → 最终完整回复 → screen_dialog
  on_tool_call   → /action_request → action coordinator
  on_llm_timeout → TFT 切待机页 + 日志
"""

import base64
import json
import os
import random
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
from services.action_execution import CorrelatedActionExecutor
from services.action_command import ACTION_REQUEST_TOPIC
from services.action_intent_guard import validate_action_arguments
from services.action_status import ACTION_STATUS_TOPIC
from services.behavior_tree_execution import CorrelatedPlanExecutor
from services.ros2_action_execution import (
    Ros2ActionPlanExecutor,
    create_wali_task_action_client,
)
from services.behavior_tree_protocol import (
    BEHAVIOR_TREE_CANCEL_TOPIC,
    BEHAVIOR_TREE_EXECUTE_TOPIC,
    BEHAVIOR_TREE_STATUS_TOPIC,
)
from services.behavior_tree_workflow import NativeBehaviorTreeWorkflow
from services.camera_frame import save_camera_photo
from services.conditional_task import CONDITIONAL_TASK_TOOL_NAME
from services.dialog_workflow import ConditionalTaskWorkflow
from services.dialog_output import DialogOutputController
from services.dialog_turn import DialogTurnController, TTS_CLEAN_RE
from services.visual_search import (
    VISUAL_SEARCH_REQUEST_TOPIC,
    VISUAL_SEARCH_STATUS_TOPIC,
    VISUAL_SEARCH_TOOL_NAME,
    compile_visual_search_plan,
    encode_visual_search_status,
    parse_visual_search_request,
)
from services.game_protocol import (
    GAME_FRAME_TOPIC,
    GAME_MODE_STATE_TOPIC,
    decode_game_frame,
    game_mode_from_message,
)
from services.game_tft_stream import prepare_game_bgr
from services.audio_output import (
    OUTPUT_CHANNELS,
    OUTPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_WIDTH,
)
from services.tft_preview_client import TftPreviewClient
from services.tft_preview_server import load_tft_preview_settings
from services.tts_protocol import encode_turn_end
from services.dialog_motion_protocol import (
    DIALOG_MOTION_VAD_TOPIC,
    VAD_SPEECH_ENDED,
    VAD_SPEECH_STARTED,
)
from services.dialog_expression_protocol import (
    DIALOG_EXPRESSION_TOPIC,
    encode_dialog_expression,
)
from services.wake_audio_protocol import (
    WAKE_AUDIO_DONE_TOPIC,
    WAKE_AUDIO_TOPIC,
    encode_wake_audio,
)

OUTPUT_ECHO_GUARD_SECONDS = 0.35

class VoiceChatNode(Node):
    def __init__(self):
        super().__init__("voice_chat_node")

        self.tts_pub = self.create_publisher(String, "tts_text", 10)
        self.wake_audio_pub = self.create_publisher(String, WAKE_AUDIO_TOPIC, 10)
        self.create_subscription(
            String, WAKE_AUDIO_DONE_TOPIC, self._on_wake_audio_done, 10
        )
        self.dialog_pub = self.create_publisher(String, "screen_dialog", 10)
        self.action_pub = self.create_publisher(String, ACTION_REQUEST_TOPIC, 10)
        self._action_executor = CorrelatedActionExecutor()
        self._behavior_tree_executor = CorrelatedPlanExecutor()
        self._ros2_task_executor = None
        self.behavior_tree_pub = self.create_publisher(
            String, BEHAVIOR_TREE_EXECUTE_TOPIC, 10
        )
        self.behavior_tree_cancel_pub = self.create_publisher(
            String, BEHAVIOR_TREE_CANCEL_TOPIC, 10
        )
        self.create_subscription(
            String,
            BEHAVIOR_TREE_STATUS_TOPIC,
            self._on_behavior_tree_status,
            10,
        )
        try:
            action_client, goal_type = create_wali_task_action_client(
                self, Path(__file__).resolve().parent.parent
            )
            self._ros2_task_executor = Ros2ActionPlanExecutor(action_client, goal_type)
            self.get_logger().info("BehaviorTree ROS2 Action client is ready")
        except Exception as exc:
            self.get_logger().warning(
                f"BehaviorTree ROS2 Action client unavailable; legacy fallback remains active: {exc}"
            )
        self.visual_search_status_pub = self.create_publisher(
            String, VISUAL_SEARCH_STATUS_TOPIC, 10
        )
        self.create_subscription(
            String,
            VISUAL_SEARCH_REQUEST_TOPIC,
            self._on_visual_search_request,
            10,
        )
        self._visual_search_lock = threading.Lock()
        self._conditional_task_workflow = None
        self.create_subscription(String, ACTION_STATUS_TOPIC, self._on_action_status, 10)
        self.game_busy_pub = self.create_publisher(String, "llm_busy", 10)
        self.dialog_motion_pub = self.create_publisher(
            String, DIALOG_MOTION_VAD_TOPIC, 10
        )
        self.dialog_expression_pub = self.create_publisher(
            String, DIALOG_EXPRESSION_TOPIC, 10
        )
        self.create_subscription(String, "llm_busy", self._on_playback_state, 10)

        self.tft_preview_settings = load_tft_preview_settings()
        self.tft_preview = TftPreviewClient(self, logger=self.get_logger())
        self._game_mode = "robot"
        self._game_frame_lock = threading.Lock()
        self._latest_game_frame = None
        self._next_game_commentary = None
        self._game_commentary_running = False
        self.create_subscription(String, GAME_MODE_STATE_TOPIC, self._on_game_state, 10)
        self.create_subscription(UInt8MultiArray, GAME_FRAME_TOPIC, self._on_game_frame, 1)
        self.create_timer(1.0, self._game_commentary_tick)

        self.get_logger().info("正在预热唤醒词 + Qwen-Omni 引擎...")

        self.vc = VoiceChatService()
        self.vc.on_wake_word = self._on_wake_word
        self.vc.on_speech_start = self._on_vad_speech_start
        self.vc.on_speech_end = self._on_vad_speech_end
        self.vc.on_llm_chunk = self._on_llm_chunk
        self.vc.on_expression = self._on_expression
        self.vc.on_llm_reply = self._on_llm_reply
        self.vc.on_tool_call = self._on_tool_call
        self.vc.on_action_plan = self._execute_behavior_tree_plan
        self.vc.on_photo_request = self._process_camera_photo
        self.vc.on_inspection_request = self._process_heard_camera_inspection
        self.vc.on_llm_done = self._on_llm_done
        self.vc.on_llm_timeout = self._on_llm_timeout

        self._turn_controller = DialogTurnController()
        self._output_controller = DialogOutputController()
        self._timer_lock = threading.Lock()
        self._wake_watchdog = None
        self._resume_timer = None
        # 唤醒应答 WAV 路径
        root = Path(__file__).resolve().parent.parent
        self._wake_wav = str(root / "assets" / "wake_response.wav")
        self._wake_play_lock = threading.Lock()

        self.vc.start()
        self.get_logger().info("语音直聊节点已上线")

    def _on_game_state(self, message):
        mode = game_mode_from_message(message.data)
        if mode is None:
            return
        previous = self._game_mode
        self._game_mode = mode
        if mode != "robot":
            if previous == "robot":
                self.vc.pause()
            if mode == "playing" and previous != "playing":
                self._schedule_next_game_commentary()
            return
        if previous == "robot":
            return
        with self._game_frame_lock:
            self._latest_game_frame = None
        self._next_game_commentary = None
        self.vc.resume()

    def _on_game_frame(self, message):
        if self._game_mode == "robot":
            return
        frame = decode_game_frame(bytes(message.data))
        if frame is None:
            return
        raw, width, height, pitch = frame
        with self._game_frame_lock:
            self._latest_game_frame = frame

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
                self._turn_controller.set_turn_id("game-" + uuid.uuid4().hex[:8])
                self._on_llm_done()
            else:
                self.game_busy_pub.publish(String(data="idle"))
        except Exception as exc:
            self.get_logger().error(f"游戏画面识别失败: {exc}")
            self.game_busy_pub.publish(String(data="idle"))
        finally:
            self._game_commentary_running = False

    def _run_camera_preview(self, *, duration_ms):
        return self.tft_preview.send_camera_preview(
            duration_ms=duration_ms,
            hold_ms=self.tft_preview_settings.hold_ms,
            fps=self.tft_preview_settings.fps,
        )

    # ── 唤醒词回调 ──
    def _on_wake_word(self):
        """唤醒词触发：播放预合成语音 + 切 TFT 到聊天页。"""
        self.get_logger().info("唤醒词触发")

        # The shared playback node owns the speaker. Keep capture muted until
        # its matching wake acknowledgement and any queued TTS have finished.
        request_id = uuid.uuid4().hex
        if not self._output().start_wake(request_id).accepted:
            return
        with self._timer_lock:
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
        threading.Thread(
            target=self._play_wake_response, args=(request_id,), daemon=True
        ).start()

    def _on_vad_speech_start(self):
        if self._game_mode == "robot":
            self.dialog_motion_pub.publish(String(data=VAD_SPEECH_STARTED))

    def _on_vad_speech_end(self):
        if self._game_mode == "robot":
            self.dialog_motion_pub.publish(String(data=VAD_SPEECH_ENDED))

    def _play_wake_response(self, request_id):
        """Send wake PCM to the shared mixer and await its actual completion."""
        submitted = False
        with self._wake_play_lock:
            try:
                if self.wake_audio_pub.get_subscription_count() == 0:
                    self.get_logger().warn("音频播放节点未连接，跳过唤醒应答")
                    return
                if not os.path.exists(self._wake_wav):
                    self.get_logger().warn(f"唤醒应答文件不存在: {self._wake_wav}")
                    self.get_logger().warn("请先运行 generate_wake_response.py 生成语音文件")
                    return

                from pydub import AudioSegment

                audio = (
                    AudioSegment.from_wav(self._wake_wav)
                    .set_frame_rate(OUTPUT_SAMPLE_RATE)
                    .set_channels(OUTPUT_CHANNELS)
                    .set_sample_width(OUTPUT_SAMPLE_WIDTH)
                )
                if not len(audio):
                    return
                payload = encode_wake_audio(request_id, audio.raw_data)
                self.wake_audio_pub.publish(String(data=payload))
                submitted = True
                self._schedule_wake_watchdog(request_id)
            except ImportError:
                self.get_logger().error("缺少音频播放依赖，无法播放唤醒应答")
            except Exception as e:
                self.get_logger().error(f"播放唤醒应答失败: {e}")
            finally:
                if not submitted:
                    self._finish_wake_response(request_id)

    def _on_wake_audio_done(self, message):
        """Only a matching wake completion may release the wake capture guard."""
        self._finish_wake_response(message.data)

    def _finish_wake_response(self, request_id):
        decision = self._output().finish_wake(request_id)
        if not decision.accepted:
            return
        with self._timer_lock:
            if self._wake_watchdog is not None:
                self._wake_watchdog.cancel()
                self._wake_watchdog = None
        if decision.schedule_capture_resume:
            self._schedule_capture_resume()

    def _schedule_wake_watchdog(self, request_id):
        if not self._output().wake_is_active(request_id):
            return
        with self._timer_lock:
            self._wake_watchdog = threading.Timer(
                1.0, self._check_wake_playback_connection, args=(request_id,)
            )
            self._wake_watchdog.daemon = True
            self._wake_watchdog.start()

    def _check_wake_playback_connection(self, request_id):
        # A duration timeout could reopen capture while earlier TTS is still
        # playing. Recover only if the speaker owner has disappeared instead.
        if not self._output().wake_is_active(request_id):
            return
        with self._timer_lock:
            self._wake_watchdog = None
        if self.wake_audio_pub.get_subscription_count() == 0:
            self.get_logger().warn("音频播放节点已断开，结束唤醒应答等待")
            self._finish_wake_response(request_id)
        else:
            self._schedule_wake_watchdog(request_id)

    def _on_playback_state(self, msg):
        """Resume multimodal capture only after the queued TTS turn is done."""
        if msg.data != "idle":
            return
        decision = self._output().finish_tts_playback()
        if decision.schedule_capture_resume:
            self._schedule_capture_resume()

    def _schedule_capture_resume(self):
        """Discard the speaker's acoustic tail before reopening capture."""
        if (
            getattr(self, "_game_mode", "robot") != "robot"
            or not self._output().can_resume_capture()
        ):
            return
        with self._timer_lock:
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
        with self._timer_lock:
            self._resume_timer = None
        if not self._output().can_resume_capture():
            return
        if self.vc.complete_output_playback():
            self.get_logger().info("扬声器尾音已清除，恢复多模态录音")

    # ── LLM 回调 ──
    def _on_tool_call(self, name, arguments):
        if name == "inspect_camera":
            return self._process_camera_inspection(arguments)
        if name == CONDITIONAL_TASK_TOOL_NAME:
            return self._process_conditional_task(arguments)
        if name == VISUAL_SEARCH_TOOL_NAME:
            return self._process_visual_search(arguments)
        allowed, reason = validate_action_arguments(name, arguments)
        if not allowed:
            return {"status": "rejected", "action": name, "reason": reason}
        result = self._action_executor.execute(
            name,
            arguments,
            publish=lambda payload: self.action_pub.publish(String(data=payload)),
            owner_available=lambda: self.action_pub.get_subscription_count() > 0,
            timeout=20.0,
            source="voice_dialog",
            cancelled=self.vc._cancel_llm.is_set,
        )
        self.get_logger().info(f"Tool: {name}({arguments}) -> {result['status']}")
        return result

    def _on_action_status(self, message):
        executor = getattr(self, "_action_executor", None)
        if executor is not None:
            executor.accept_status(message.data)

    def _on_behavior_tree_status(self, message):
        executor = getattr(self, "_behavior_tree_executor", None)
        if executor is not None:
            executor.accept_status(message.data)

    def _on_visual_search_request(self, message):
        request = parse_visual_search_request(message.data)
        if request is None:
            self.get_logger().warning("Ignored malformed visual-search request")
            return
        threading.Thread(
            target=self._serve_visual_search_request,
            args=(request,),
            daemon=True,
        ).start()

    def _serve_visual_search_request(self, request):
        with self._visual_search_lock:
            try:
                preview = self._run_camera_preview(
                    duration_ms=self.tft_preview_settings.recognition_duration_ms,
                )
                if preview.busy or not preview.last_frame:
                    result = {
                        "status": "uncertain",
                        "evidence": preview.error or "camera_frame_unavailable",
                        "response": "这次没有取得可判断的画面。",
                    }
                else:
                    result = self.vc.evaluate_visual_search(
                        request["target"],
                        request["question"],
                        base64.b64encode(preview.last_frame).decode("ascii"),
                    )
            except Exception as exc:
                self.get_logger().error(f"Visual search view failed: {exc}")
                result = {
                    "status": "uncertain",
                    "evidence": "visual_search_view_failed",
                    "response": "这次画面没有分析成功。",
                }
            self.visual_search_status_pub.publish(String(data=encode_visual_search_status(
                request, result
            )))

    def _process_visual_search(self, arguments):
        try:
            plan = compile_visual_search_plan(
                turn_id=self._ensure_turn_id(), arguments=arguments
            )
        except (TypeError, ValueError) as exc:
            return {
                "status": "rejected",
                "action": VISUAL_SEARCH_TOOL_NAME,
                "reason": str(exc),
            }
        for step in plan.steps[1:]:
            allowed, reason = validate_action_arguments(step.name, step.arguments)
            if not allowed:
                return {
                    "status": "rejected",
                    "action": VISUAL_SEARCH_TOOL_NAME,
                    "reason": f"completion_action_invalid:{step.step_id}:{reason}",
                }
        self.tts_pub.publish(String(data="我找一下。"))
        self.get_logger().info(
            "Visual search submitted: "
            f"target={plan.target}, max_views={plan.max_views}, "
            f"motion={plan.steps[0].arguments}, "
            f"on_found_actions={len(plan.steps) - 1}"
        )
        result = self._try_execute_native_plan(
            plan,
            timeout=plan.max_views * 50.0 + 10.0,
            cancelled=self.vc._cancel_llm.is_set,
        )
        if result is None:
            return {
                "status": "failed",
                "action": VISUAL_SEARCH_TOOL_NAME,
                "reason": "native_behavior_tree_unavailable",
            }
        search_results = [
            item for item in result.get("results", [])
            if item.get("action") == VISUAL_SEARCH_TOOL_NAME
        ]
        if result.get("status") == "success" and search_results:
            final = dict(search_results[-1])
            final["status"] = "completed"
            self.get_logger().info(
                "Visual search completed: "
                f"found={final.get('found')}, attempts={final.get('attempts')}"
            )
            return final
        return {
            "status": "failed",
            "action": VISUAL_SEARCH_TOOL_NAME,
            "reason": result.get("error") or result.get("status") or "search_failed",
        }

    def _execute_behavior_tree_plan(self, heard_text, actions):
        """Submit ordinary actions to the native tree; return None for fallback."""
        if any(
            action.get("name") in {
                "inspect_camera", CONDITIONAL_TASK_TOOL_NAME, VISUAL_SEARCH_TOOL_NAME
            }
            for action in actions
            if isinstance(action, dict)
        ):
            return None
        workflow = NativeBehaviorTreeWorkflow(
            authorize=lambda _prompt, name, arguments: validate_action_arguments(
                name, arguments
            ),
            execute_plan=lambda plan: self._try_execute_native_plan(
                plan, cancelled=self.vc._cancel_llm.is_set
            ),
        )
        return workflow.invoke(
            turn_id=self._ensure_turn_id(),
            user_prompt=heard_text,
            actions=actions,
        )

    def _try_execute_native_plan(self, plan, *, timeout=None, cancelled=None):
        """Prefer standard ExecuteTree Action, then retain the legacy fallback."""
        ros2_executor = getattr(self, "_ros2_task_executor", None)
        if ros2_executor is not None:
            result = ros2_executor.try_execute(
                plan, timeout=timeout, cancelled=cancelled
            )
            if result is not None:
                return result
        executor = getattr(self, "_behavior_tree_executor", None)
        publisher = getattr(self, "behavior_tree_pub", None)
        cancel_publisher = getattr(self, "behavior_tree_cancel_pub", None)
        if executor is None or publisher is None or cancel_publisher is None:
            return None
        return executor.try_execute(
            plan,
            publish=lambda payload: publisher.publish(String(data=payload)),
            cancel_publish=lambda payload: cancel_publisher.publish(String(data=payload)),
            owner_available=lambda: publisher.get_subscription_count() > 0,
            timeout=timeout,
            cancelled=cancelled,
        )

    def _process_conditional_task(self, plan):
        """Run one audio-selected observe-condition-action graph."""
        workflow = getattr(self, "_conditional_task_workflow", None)
        if workflow is None:
            workflow = ConditionalTaskWorkflow(
                capture=lambda: self._run_camera_preview(
                    duration_ms=self.tft_preview_settings.recognition_duration_ms,
                ),
                evaluate=self._evaluate_camera_condition,
                authorize=lambda _prompt, name, arguments: validate_action_arguments(
                    name, arguments
                ),
                execute=self._execute_workflow_action,
            )
            self._conditional_task_workflow = workflow
        try:
            result = workflow.invoke(
                turn_id=self._ensure_turn_id(),
                user_prompt="audio_conditional_task",
                plan=plan,
            )
            if result.get("error"):
                self.get_logger().warning(
                    f"Conditional task completed with error: {result['error']}"
                )
            action_result = result.get("action_result")
            action_status = (
                action_result.get("status")
                if isinstance(action_result, dict)
                else "not_run"
            )
            self.get_logger().info(
                "Conditional task outcome: "
                f"decision={result.get('decision') or 'unavailable'}, "
                f"evidence={result.get('evidence') or '-'}, "
                f"action_status={action_status}"
            )
            return result.get("answer") or "这次任务没有完成。"
        except Exception as exc:
            self.get_logger().error(
                f"Conditional task failed: {exc}\n{traceback.format_exc()}"
            )
            return "这个任务计划没有通过检查，所以我没有执行动作。"

    def _evaluate_camera_condition(self, frame, observation, condition):
        return self.vc.evaluate_image_condition(
            observation,
            condition,
            base64.b64encode(frame).decode("ascii"),
        )

    def _execute_workflow_action(self, name, arguments):
        return self._action_executor.execute(
            name,
            arguments,
            publish=lambda payload: self.action_pub.publish(String(data=payload)),
            owner_available=lambda: self.action_pub.get_subscription_count() > 0,
            timeout=20.0,
            source="voice_conditional_task",
        )

    def _on_expression(self, expression, intensity):
        turn_id = self._ensure_turn_id()
        self.dialog_expression_pub.publish(String(data=encode_dialog_expression(
            expression, intensity, turn_id
        )))

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
        decision = self._turn().consume_chunk(text)
        if decision is None or not decision.tts_text:
            return
        self.tts_pub.publish(String(data=decision.tts_text))
        self.get_logger().info(f"TTS: {decision.tts_text[:80]}")

    def _on_llm_reply(self, text):
        """最终完整回复 → 解析 you/ai → screen_dialog。"""
        decision = self._turn().finish_reply(text)
        if decision is None:
            return

        # 终端输出：让用户看到自己说了什么
        if decision.corrected_text:
            self.get_logger().info(f"[识别] {decision.corrected_text}")
        self.get_logger().info(f"[回复] {decision.ai_text[:80]}")

        # flush 残留 TTS 文本
        if decision.tts_tail:
            self.tts_pub.publish(String(data=decision.tts_tail))
            self.get_logger().info(f"TTS tail: {decision.tts_tail[:80]}")

        # 屏幕对话框（对齐 llm_ros_node 格式）
        dialog = String()
        action_results = getattr(self.vc, "last_action_results", [])
        if not isinstance(action_results, list):
            action_results = []
        dialog.data = json.dumps({
            "turn_id": decision.turn_id,
            "corrected_text": decision.corrected_text,
            "ai_text": decision.ai_text,
            "actions": action_results,
            "source": "voice_chat",
        }, ensure_ascii=False)
        self.dialog_pub.publish(dialog)
        self.get_logger().info(f"Screen: {decision.ai_text[:60]}")

    def _ensure_turn_id(self):
        return self._turn().ensure_turn_id()

    def _turn(self):
        controller = getattr(self, "_turn_controller", None)
        if controller is None:
            controller = DialogTurnController()
            self._turn_controller = controller
        return controller

    def _output(self):
        controller = getattr(self, "_output_controller", None)
        if controller is None:
            controller = DialogOutputController()
            self._output_controller = controller
        return controller

    def _on_llm_done(self):
        """关闭本轮 TTS；播放节点会在音频真正播完后结束回合。"""
        turn_id = self._turn().finish()
        self._output().mark_tts_queued()
        self.tts_pub.publish(String(data=encode_turn_end(turn_id)))
        self.get_logger().info(f"TTS turn queued: {turn_id}")

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
        self._output().shutdown()
        with self._timer_lock:
            if self._wake_watchdog is not None:
                self._wake_watchdog.cancel()
                self._wake_watchdog = None
            if self._resume_timer is not None:
                self._resume_timer.cancel()
                self._resume_timer = None
        if hasattr(self, "vc"):
            self.vc.stop()
        if getattr(self, "tft_preview", None) is not None:
            self.tft_preview.close()
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
