#!/usr/bin/env python3
import base64
import json
import queue
import random
import re
import threading
import time
import traceback
import uuid
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, UInt8MultiArray
from pypinyin import Style, pinyin

from services.action_acknowledgement import action_acknowledgement
from services.action_intent_guard import validate_action_call
from services.llm_service import LLMService
from services.camera_frame import (
    CameraFrameProvider,
    is_camera_inspection_request,
    is_camera_photo_request,
    save_camera_photo,
)
from services.game_frame_adapter import GameFrameAdapter
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
from services.tft_preview_server import (
    PreviewResult,
    load_tft_preview_settings,
)
from services.tracking_tft_preview import TrackingTftPreview
from services.tts_protocol import encode_turn_end
from services.dialog_expression_protocol import (
    DIALOG_EXPRESSION_TOPIC,
    encode_dialog_expression,
)
from services.vision_pipeline_protocol import (
    VISION_PIPELINE_COMMAND_TOPIC,
    decode_vision_pipeline_command,
)


class LLMBrainNode(Node):
    CHAT_HISTORY_MESSAGES = 12
    LONG_FORM_REQUEST_RE = re.compile(
        r"(?:背(?:诵)?|朗(?:诵|读)|念|读)(?:一下|一遍|给我听)?|"
        r"全文|完整(?:版|内容)?|全部|整首|从头到尾"
    )
    LONG_FORM_MAX_TOKENS = 2048
    FIRST_TTS_CLAUSE_MIN_CHARS = 10
    CLAUSE_PUNCTUATIONS = {'，', ',', '；', ';', '：', ':'}
    CORRECTION_LABELS = {
        "\u4fee\u6b63\u6587\u672c",
        "\u7ea0\u9519\u6587\u672c",
        "\u6821\u6b63\u6587\u672c",
        "\u8bc6\u522b\u4fee\u6b63",
        "\u4fee\u6b63\u540e\u6587\u672c",
        "corrected_text",
        "corrected text",
    }
    TTS_CLEAN_RE = re.compile(
        "[^\\w\\s\u4e00-\u9fa5\uff0c\u3002\uff1f\uff01\u3001\uff1a\uff1b\u201c\u201d\uff08\uff09\u300a\u300b.,?!]"
    )
    OUTPUT_LINE_PREFIX_RE = re.compile(
        r"^\s*(?:\u7b2c\u4e8c\u884c|\u6700\u7ec8\u56de\u7b54|\u6700\u7ec8\u7b54\u6848|\u56de\u7b54|\u56de\u590d)\s*[:\uff1a]\s*",
        re.IGNORECASE,
    )

    def __init__(self):
        super().__init__('walle_llm_brain')

        self.llm = None
        self.chat_history = deque(maxlen=24)
        self.punctuations = {'。', '？', '.', '?', '！', '!'}
        self._request_queue = queue.Queue(maxsize=8)
        self._worker_running = False
        self._game_mode = "robot"
        self._game_stream = None
        self._game_frame_adapter = None
        self._game_frame_lock = threading.Lock()
        self._latest_game_frame = None
        self._next_game_commentary = None
        self._game_commentary_pending = False
        self._tracking_was_enabled = False

        # Create ROS endpoints before the slow LLM client init. This lets DDS
        # discover `voice_text` while the model service is warming up.
        self.voice_subscription = self.create_subscription(
            String,
            'voice_text',
            self.voice_callback,
            10,
        )
        self.tts_publisher = self.create_publisher(String, 'tts_text', 10)
        self.action_publisher = self.create_publisher(String, 'action_cmd', 10)
        self.corrected_publisher = self.create_publisher(String, 'corrected_text', 10)
        self.full_ai_publisher = self.create_publisher(String, 'full_ai_text', 10)
        self.screen_dialog_publisher = self.create_publisher(String, 'screen_dialog', 10)
        self.busy_publisher = self.create_publisher(String, 'llm_busy', 10)
        self.dialog_expression_publisher = self.create_publisher(
            String, DIALOG_EXPRESSION_TOPIC, 10
        )
        self.game_request_publisher = self.create_publisher(String, GAME_MODE_REQUEST_TOPIC, 10)
        self.camera_frames = CameraFrameProvider(self)
        self.tft_preview_ready_publisher = self.create_publisher(
            String, 'tft_preview_ready', 10
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
            # Camera/photo business remains available when port 9000 is busy or
            # the network stack is unavailable.
            self.get_logger().error(f'TFT preview service failed to start: {exc}')
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
        self.create_subscription(String, GAME_MODE_STATE_TOPIC, self._on_game_state, 10)
        self.create_subscription(UInt8MultiArray, GAME_FRAME_TOPIC, self._on_game_frame, 1)
        self.create_timer(1.0, self._game_commentary_tick)

        try:
            self.llm = LLMService()
            self.get_logger().info('LLM service initialized.')
        except Exception as e:
            self.get_logger().error(f'LLM service initialization failed: {e}')
            return

        self._worker_running = True
        self._worker_thread = threading.Thread(
            target=self._llm_worker,
            name='llm-worker',
            daemon=True,
        )
        self._worker_thread.start()

    def _publish_tft_preview_ready(self):
        ready = String()
        ready.data = 'ready'
        self.tft_preview_ready_publisher.publish(ready)

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
                self._tracking_was_enabled = self.tracking_tft_preview.pause()
            self._ensure_game_stream()
            if mode == "playing" and previous != "playing":
                self._schedule_next_game_commentary()
            return

        self._close_game_stream()
        with self._game_frame_lock:
            self._latest_game_frame = None
        self._next_game_commentary = None
        self._game_commentary_pending = False
        if self._tracking_was_enabled:
            self.tracking_tft_preview.resume()
        self._tracking_was_enabled = False

    def _ensure_game_stream(self):
        if self._game_frame_adapter is not None:
            return
        stream = self.tft_preview.open_jpeg_stream(fps=10)
        if stream is None:
            self.get_logger().warning("游戏 TFT 流暂不可用")
            return
        self._game_stream = stream
        self._game_frame_adapter = GameFrameAdapter(stream, fps=10)
        self.game_request_publisher.publish(
            String(data=encode_game_request(GAME_SURFACE_READY))
        )

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
            self._latest_game_frame = (raw, width, height, pitch)
        adapter = self._game_frame_adapter
        if adapter is not None:
            adapter.submit_frame(raw, width, height, pitch)

    def _schedule_next_game_commentary(self):
        self._next_game_commentary = time.monotonic() + random.uniform(50.0, 120.0)

    def _game_commentary_tick(self):
        if self._game_mode != "playing" or self._game_commentary_pending:
            return
        if self._next_game_commentary is None:
            self._schedule_next_game_commentary()
            return
        if time.monotonic() < self._next_game_commentary:
            return
        with self._game_frame_lock:
            frame = self._latest_game_frame
        self._schedule_next_game_commentary()
        if frame is None or self.llm is None:
            return
        import numpy as np

        raw, width, height, pitch = frame
        image = np.frombuffer(raw, dtype=np.uint8).reshape(height, pitch // 4, 4)
        jpeg = prepare_game_bgr(image[:, :width, :3], quality=75)
        if not jpeg:
            return
        self._game_commentary_pending = True
        try:
            self._request_queue.put_nowait({
                'kind': 'game_vision',
                'turn_id': 'game-' + uuid.uuid4().hex[:8],
                'jpeg': jpeg,
            })
        except queue.Full:
            self._game_commentary_pending = False

    def voice_callback(self, msg):
        """Queue the request so the ROS callback thread is never blocked by LLM I/O."""
        if self._game_mode != "robot":
            self.get_logger().info("游戏模式热备中，忽略语音 LLM 输入")
            return
        user_prompt = (msg.data or '').strip()
        # 过滤掉常见的 ASR 噪声音译（如 #，或者单纯的标点符号）
        if not user_prompt or user_prompt == '#' or len(user_prompt.strip('.,?!。，？！# ')) == 0:
            self.get_logger().info(f'Ignored empty/noise ASR input: "{user_prompt}"')
            return

        turn_id = uuid.uuid4().hex[:12]
        self.get_logger().info(f'[{turn_id}] Voice text received: {user_prompt}')

        try:
            self._request_queue.put_nowait({
                'turn_id': turn_id,
                'user_prompt': user_prompt,
            })
        except queue.Full:
            self.get_logger().error('LLM request queue is full; dropped this voice input.')

    def _llm_worker(self):
        while self._worker_running:
            try:
                task = self._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if task is None:
                self._request_queue.task_done()
                continue

            try:
                if task.get('kind') == 'game_vision':
                    self._process_game_vision_task(task)
                else:
                    self._process_voice_task(task['turn_id'], task['user_prompt'])
            except Exception as e:
                self.get_logger().error(f'Unhandled LLM worker error: {e}\n{traceback.format_exc()}')
                failure_text = '\u6211\u521a\u624d\u5904\u7406\u5931\u8d25\u4e86\uff0c\u7a0d\u540e\u518d\u8bd5\u3002'
                self._publish_tts(failure_text, task.get('turn_id', ''))
                self._publish_screen_dialog(
                    task.get('turn_id', ''),
                    task.get('user_prompt', ''),
                    failure_text,
                    [],
                    error=str(e),
                )
                self._finish_tts_turn(task.get('turn_id', ''))
            finally:
                if task.get('kind') == 'game_vision':
                    self._game_commentary_pending = False
                self._request_queue.task_done()

    def _process_game_vision_task(self, task):
        if self._game_mode != "playing":
            return
        turn_id = task['turn_id']
        self.busy_publisher.publish(String(data="busy"))
        prompt = (
            "观察这张正在运行的 FC 游戏画面，以瓦力的口吻说一句简短自然的中文评论。"
            "可以提醒危险、鼓励玩家或描述关键局面；看不清时不要猜。只输出可直接播报的一句话。"
        )
        try:
            chunks = []
            for data in self.llm.chat_stream(
                prompt,
                [],
                image_base64=base64.b64encode(task['jpeg']).decode('ascii'),
                tools_enabled=False,
                structured_answer=False,
                system_prompt=(
                    "你是陪主人玩 FC 游戏的瓦力。只依据当前游戏截图简短评论，"
                    "不输出分析过程。"
                ),
                max_tokens_override=96,
            ):
                if data.get('type') == 'text' and data.get('content'):
                    chunks.append(data['content'])
            answer = self._clean_visual_answer(''.join(chunks))
            if not answer:
                raise RuntimeError('游戏视觉模型返回空答案')
            self._publish_tts(answer, turn_id)
            self.full_ai_publisher.publish(String(data=answer))
        except Exception as exc:
            self.get_logger().error(f'[{turn_id}] Game vision failed: {exc}')
        finally:
            self._finish_tts_turn(turn_id)

    def _process_voice_task(self, turn_id, user_prompt):
        # 通知 STT 节点暂停 ASR
        busy_msg = String()
        busy_msg.data = "busy"
        self.busy_publisher.publish(busy_msg)

        # 拍照只保存本地文件，不进入视觉模型。
        if is_camera_photo_request(user_prompt):
            self._process_camera_photo(turn_id, user_prompt)
            return

        # 视觉查看是一个两阶段技能：先立即确认，再预览并把末帧交给视觉模型。
        # 这样用户不会等待摄像头和第二次 LLM 请求时陷入沉默。
        if is_camera_inspection_request(user_prompt):
            self._process_camera_inspection(turn_id, user_prompt)
            return

        py_list = pinyin(user_prompt, style=Style.NORMAL)
        py_str = ' '.join([item[0] for item in py_list])
        is_long_form = self._is_long_form_request(user_prompt)
        # Tool availability must not depend on a keyword gate. ASR wording and
        # natural requests such as “旋转头/转个头” are semantic decisions for
        # the model, not a brittle regex. Explicit visual/retry paths below
        # still call chat_stream(tools_enabled=False) by design.
        tools_enabled = True
        if is_long_form:
            response_policy = (
                "这是朗读、背诵或完整内容请求。请连续完整输出用户要求的正文，"
                "不要只给标题、简介或开头一句；除非用户明确只要片段。"
            )
        else:
            response_policy = "普通对话保持一到两句、简短自然。"

        augmented_prompt = (
            f"\u539f\u59cb ASR \u6587\u672c\uff1a{user_prompt}\n"
            f"\u62fc\u97f3\u53c2\u8003\uff1a{py_str}\n\n"
            "\u8bf7\u7ed3\u5408\u5bf9\u8bdd\u4e0a\u4e0b\u6587\u548c\u62fc\u97f3\u9759\u9ed8\u7406\u89e3\u7528\u6237\u672c\u610f\uff0c\u7136\u540e\u76f4\u63a5\u56de\u7b54\u3002"
            "\u53ea\u8f93\u51fa\u53ef\u4ee5\u901a\u8fc7\u626c\u58f0\u5668\u64ad\u653e\u7684\u6700\u7ec8\u53f0\u8bcd\uff0c\u4e0d\u8981\u8f93\u51fa\u6216\u590d\u8ff0\u539f\u59cb ASR \u6587\u672c\u3001"
            "\u62fc\u97f3\u3001\u4fee\u6b63\u6587\u672c\u3001\u7ea0\u9519\u7ed3\u679c\u3001\u5206\u6790\u3001\u601d\u8003\u3001\u8ba1\u5212\u3001\u89c4\u5219\u590d\u8ff0\u3001\u793a\u4f8b\u3001\u5217\u8868\u3001Markdown\u3001"
            "\u62ec\u53f7\u8bf4\u660e\u3001Function Calling \u5b57\u6837\u3001\u5de5\u5177\u540d\u6216\u5de5\u5177\u53c2\u6570\u3002"
            "\u9700\u8981\u52a8\u4f5c\u65f6\u53ea\u4f7f\u7528\u539f\u751f\u5de5\u5177\u8c03\u7528\uff0c\u4e0d\u8981\u5728\u6587\u5b57\u4e2d\u63cf\u8ff0\u8c03\u7528\u8fc7\u7a0b\u3002"
            f"{response_policy}"
        )

        self.get_logger().info(f'[{turn_id}] Sending request to LLM...')
        self.get_logger().info(f'[{turn_id}] Control tools enabled for semantic handling.')
        max_tokens_override = self._max_tokens_for_request(is_long_form)
        if max_tokens_override is not None:
            self.get_logger().info(
                f'[{turn_id}] Long-form request detected; max_tokens={max_tokens_override}.'
            )

        text_buffer = ''
        sentence_buffer = ''
        corrected_text = ''
        corrected_text_published = False
        actions = []
        rejected_actions = []
        spoken_parts = []
        expression_published = False

        def publish_corrected(value):
            nonlocal corrected_text, corrected_text_published
            corrected_text = (value or user_prompt).strip() or user_prompt
            corrected_text_published = True
            msg = String()
            msg.data = corrected_text
            self.corrected_publisher.publish(msg)
            self.get_logger().info(
                f'[{turn_id}] Corrected text: raw="{user_prompt}" corrected="{corrected_text}"'
            )

        def publish_spoken(value):
            nonlocal expression_published
            if not expression_published:
                expression_publisher = getattr(
                    self, "dialog_expression_publisher", None
                )
                if expression_publisher is not None:
                    expression_publisher.publish(String(
                        data=encode_dialog_expression("neutral", "low", turn_id)
                    ))
                expression_published = True
            spoken = self._publish_tts(value, turn_id)
            if spoken:
                spoken_parts.append(spoken)

        # Correction metadata is an internal concern. Publish the ASR text for
        # the existing topic contract and ask the model for speech only.
        publish_corrected(user_prompt)

        try:
            stream = self.llm.chat_stream(
                augmented_prompt,
                self._history_for_request(),
                tools_enabled=tools_enabled,
                max_tokens_override=max_tokens_override,
            )

            for data in stream:
                data_type = data.get('type')

                if data_type == 'text':
                    chunk = data.get('content', '')
                    text_buffer += chunk
                    for char in chunk:
                        sentence_buffer += char
                        sentence_boundary = char in self.punctuations
                        first_clause_boundary = (
                            not spoken_parts
                            and char in self.CLAUSE_PUNCTUATIONS
                            and len(self.TTS_CLEAN_RE.sub('', sentence_buffer).strip())
                            >= self.FIRST_TTS_CLAUSE_MIN_CHARS
                        )
                        if sentence_boundary or first_clause_boundary:
                            clean_sentence = sentence_buffer.strip()
                            tts_safe = self.TTS_CLEAN_RE.sub('', clean_sentence)

                            if tts_safe.strip(' .,?!。，？！'):
                                publish_spoken(tts_safe)

                            sentence_buffer = ''

                elif data_type == 'dialog_expression':
                    self.dialog_expression_publisher.publish(String(
                        data=encode_dialog_expression(
                            data.get('expression'),
                            data.get('intensity'),
                            turn_id,
                        )
                    ))
                    expression_published = True

                elif data_type == 'tool_call':
                    action_name = data.get('name')
                    try:
                        action_arguments = json.loads(data.get('arguments') or '{}')
                    except (TypeError, json.JSONDecodeError):
                        self.get_logger().warning(
                            f'[{turn_id}] Rejected malformed tool arguments: {action_name}'
                        )
                        continue
                    allowed, rejection_reason = validate_action_call(
                        user_prompt,
                        action_name,
                        action_arguments,
                    )
                    if not allowed:
                        rejected_actions.append((action_name, rejection_reason))
                        self.get_logger().warning(
                            f'[{turn_id}] Rejected tool proposal: '
                            f'name={action_name} reason={rejection_reason}'
                        )
                        continue
                    if action_name == 'inspect_camera':
                        self.get_logger().info(f'[{turn_id}] Camera inspection tool requested.')
                        self._process_camera_inspection(turn_id, user_prompt)
                        return
                    action_payload = {
                        'turn_id': turn_id,
                        'name': action_name,
                        'arguments': json.dumps(action_arguments, ensure_ascii=False),
                    }
                    actions.append(action_payload)

                    self.get_logger().info(f'[{turn_id}] Tool call: {action_payload["name"]}')
                    action_msg = String()
                    action_msg.data = json.dumps(action_payload, ensure_ascii=False)
                    self.action_publisher.publish(action_msg)
                elif data_type == 'done':
                    finish_reason = data.get('finish_reason') or 'unknown'
                    log = self.get_logger().warning if finish_reason == 'length' else self.get_logger().info
                    log(f'[{turn_id}] LLM stream completed: finish_reason={finish_reason}')

        except Exception as e:
            self.get_logger().error(f'[{turn_id}] LLM request/stream failed: {e}\n{traceback.format_exc()}')
            if not corrected_text_published:
                publish_corrected(user_prompt)
            failure_text = '\u6211\u521a\u624d\u5904\u7406\u5931\u8d25\u4e86\uff0c\u7a0d\u540e\u518d\u8bd5\u3002'
            publish_spoken(failure_text)
            self._publish_screen_dialog(turn_id, corrected_text or user_prompt, failure_text, actions, error=str(e))
            self._finish_tts_turn(turn_id)
            return

        clean_tail = sentence_buffer.strip()
        if clean_tail:
            tts_safe_tail = self.TTS_CLEAN_RE.sub('', clean_tail)
            if tts_safe_tail.strip(' .,?!。，？！'):
                publish_spoken(tts_safe_tail)

        final_user_memory = corrected_text if corrected_text else user_prompt

        clean_text = self._sanitize_speech_text(text_buffer)
        if '\n' not in text_buffer and self._extract_corrected_text(text_buffer):
            clean_text = ''

        if not clean_text and spoken_parts:
            clean_text = ''.join(spoken_parts).strip()
        if not clean_text and actions:
            clean_text = action_acknowledgement(actions)
        if not clean_text and rejected_actions:
            clean_text = self._rejected_action_reply(rejected_actions)
        if not clean_text:
            clean_text = self._retry_empty_answer(
                turn_id,
                corrected_text or user_prompt,
            )
        if not clean_text:
            clean_text = '\u6211\u521a\u624d\u5361\u4f4f\u4e86\uff0c\u7b49\u6211\u7f13\u4e00\u4e0b\u3002'
        if not spoken_parts:
            publish_spoken(clean_text)

        self.chat_history.append({'role': 'user', 'content': final_user_memory})
        
        if actions:
            openai_tool_calls = []
            for i, act in enumerate(actions):
                # We need a dummy ID for the history
                call_id = f"call_{turn_id}_{i}"
                act['id'] = call_id  # Save it so we can reference it in the tool message
                openai_tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": act["name"],
                        "arguments": act.get("arguments", "{}")
                    }
                })
            # Keep the history protocol-valid and reinforce the same contract
            # used by the system prompt: action proposals have no speech
            # content; the deterministic acknowledgement is a following
            # assistant message after all tool results.
            self.chat_history.append({
                'role': 'assistant',
                'content': None,
                'tool_calls': openai_tool_calls,
            })
        else:
            self.chat_history.append({'role': 'assistant', 'content': clean_text})

        if clean_text:
            full_msg = String()
            full_msg.data = clean_text
            self.full_ai_publisher.publish(full_msg)

        # Append tool responses so the LLM knows the tools succeeded
        if actions:
            for act in actions:
                self.chat_history.append({
                    'role': 'tool',
                    'tool_call_id': act['id'],
                    'name': act['name'],
                    'content': '{"status": "accepted"}'
                })
            self.chat_history.append({'role': 'assistant', 'content': clean_text})

        self._publish_screen_dialog(turn_id, final_user_memory, clean_text, actions)

        # TTS 和播放节点会按顺序处理该标记；真正播完后再恢复 ASR。
        self._finish_tts_turn(turn_id)

    def _process_camera_inspection(self, turn_id, user_prompt):
        """确认 → TFT 预览 1.5 秒 → 末帧视觉 LLM → 播报结果。"""
        confirmation = '好的，我看一下。'
        self._publish_tts(confirmation, turn_id)
        self._publish_screen_dialog(turn_id, user_prompt, confirmation, [])

        preview = self._run_camera_preview(
            duration_ms=self.tft_preview_settings.recognition_duration_ms,
        )
        if preview.busy:
            failure = '我正在处理上一张画面，等一下再看。'
            self._publish_tts(failure, turn_id)
            self._publish_screen_dialog(
                turn_id, user_prompt, failure, [], error='camera_preview_busy'
            )
            self._finish_tts_turn(turn_id)
            return
        frame = preview.last_frame
        if not frame:
            failure = '我现在看不到画面，检查一下摄像头连接。'
            self._publish_tts(failure, turn_id)
            self._publish_screen_dialog(
                turn_id, user_prompt, failure, [], error='camera_frame_unavailable'
            )
            self._finish_tts_turn(turn_id)
            return

        import base64
        image_b64 = base64.b64encode(frame).decode('ascii')
        visual_prompt = (
            '请根据我附带的摄像头画面回答用户的问题。只输出简短、自然、可直接播报的中文答案，'
            '不要输出修正文本标签、分析过程、工具调用或括号说明。\n'
            f'用户问题：{user_prompt}'
        )
        try:
            chunks = []
            for data in self.llm.chat_stream(
                visual_prompt,
                self._visual_history(),
                image_base64=image_b64,
                tools_enabled=False,
                structured_answer=False,
                system_prompt=(
                    '你是瓦力的视觉。只依据当前摄像头图片回答问题；看不清时明确说看不清，'
                    '不要猜测。答案使用简短自然的中文，不能输出分析过程或任何标签。'
                ),
            ):
                if data.get('type') == 'text' and data.get('content'):
                    chunks.append(data['content'])
            answer = self._clean_visual_answer(''.join(chunks))
            if not answer:
                raise RuntimeError('视觉模型返回空答案')

            self._publish_tts(answer, turn_id)
            self.chat_history.append({'role': 'user', 'content': user_prompt})
            self.chat_history.append({'role': 'assistant', 'content': answer})
            self.full_ai_publisher.publish(String(data=answer))
            self._publish_screen_dialog(turn_id, user_prompt, answer, [])
        except Exception as exc:
            self.get_logger().error(f'[{turn_id}] Vision request failed: {exc}\n{traceback.format_exc()}')
            failure = '这张图我没分析出来，你换个角度再让我看看。'
            self._publish_tts(failure, turn_id)
            self._publish_screen_dialog(turn_id, user_prompt, failure, [], error=str(exc))
        finally:
            self._finish_tts_turn(turn_id)

    def _process_camera_photo(self, turn_id, user_prompt):
        """确认 → TFT 预览 3 秒 → 保存末帧；不调用视觉模型。"""
        confirmation = '好的，准备拍照。'
        self._publish_tts(confirmation, turn_id)
        self._publish_screen_dialog(turn_id, user_prompt, confirmation, [])

        preview = self._run_camera_preview(
            duration_ms=self.tft_preview_settings.photo_duration_ms,
        )
        if preview.busy:
            answer = '我正在拍上一张，等一下再试。'
            error = 'camera_preview_busy'
        elif not preview.last_frame:
            answer = '这次没有拍到，检查一下摄像头连接。'
            error = preview.error or 'camera_frame_unavailable'
        else:
            try:
                photo_path = save_camera_photo(
                    preview.last_frame,
                    self.tft_preview_settings.photo_directory,
                )
                self.get_logger().info(f'[{turn_id}] Photo saved: {photo_path}')
                answer = '拍好了，照片已经保存。'
                error = None
            except Exception as exc:
                self.get_logger().error(
                    f'[{turn_id}] Photo save failed: {exc}\n{traceback.format_exc()}'
                )
                answer = '画面拍到了，但照片保存失败了。'
                error = str(exc)

        self._publish_tts(answer, turn_id)
        self.chat_history.append({'role': 'user', 'content': user_prompt})
        self.chat_history.append({'role': 'assistant', 'content': answer})
        self.full_ai_publisher.publish(String(data=answer))
        self._publish_screen_dialog(turn_id, user_prompt, answer, [], error=error)
        self._finish_tts_turn(turn_id)

    def _run_camera_preview(self, *, duration_ms):
        """Run camera capture on the LLM worker, never on the ROS callback thread."""
        preview_service = getattr(self, 'tft_preview', None)
        if preview_service is None:
            return PreviewResult(
                last_frame=self.camera_frames.capture(timeout=10.0, request_timeout=15.0)
            )
        tracking_preview = getattr(self, 'tracking_tft_preview', None)
        was_tracking = tracking_preview.pause() if tracking_preview is not None else False
        try:
            return preview_service.send_camera_preview(
                self.camera_frames,
                duration_ms=duration_ms,
                hold_ms=self.tft_preview_settings.hold_ms,
                fps=self.tft_preview_settings.fps,
            )
        finally:
            if was_tracking:
                tracking_preview.resume()

    def _visual_history(self):
        """只保留文本形式的最近上下文，避免把旧 tool 消息传给视觉模型。"""
        history = []
        for item in list(self.chat_history)[-8:]:
            if item.get('role') not in {'user', 'assistant'}:
                continue
            if isinstance(item.get('content'), str) and item['content'].strip():
                history.append({'role': item['role'], 'content': item['content']})
        return history

    def _history_for_request(self):
        history = [
            message
            for item in list(self.chat_history)[-self.CHAT_HISTORY_MESSAGES:]
            if (message := self._text_only_history_message(item)) is not None
        ]
        while history and history[0].get('role') != 'user':
            history.pop(0)
        return history

    @staticmethod
    def _text_only_history_message(item):
        """Copy one history item while permanently dropping image blocks."""
        if not isinstance(item, dict):
            return None
        message = dict(item)
        content = message.get('content')
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if not isinstance(block, dict) or block.get('type') != 'text':
                    continue
                value = block.get('text')
                if isinstance(value, str) and value.strip():
                    text_parts.append(value.strip())
            message['content'] = '\n'.join(text_parts)
        elif content is not None and not isinstance(content, str):
            message['content'] = ''
        return message

    @staticmethod
    def _rejected_action_reply(rejected_actions):
        names = {name for name, _reason in rejected_actions}
        if 'inspect_camera' in names:
            return '你是想让我打开摄像头看一下吗？'
        return '我不太确定你是不是要我执行这个动作，可以再明确说一下吗？'

    @staticmethod
    def _clean_visual_answer(text):
        text = (text or '').strip()
        if text.startswith('```'):
            text = text.strip('`').strip()
        # 防止兼容模型仍然套用本项目普通对话的标签格式。
        for prefix in ('【修正文本】', '修正文本:', '修正文本：', 'ai:', 'AI:'):
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip(' ：:')
        return LLMBrainNode.TTS_CLEAN_RE.sub('', text).strip()

    def _publish_tts(self, text, turn_id=''):
        safe = self._sanitize_speech_text(text)
        if not safe:
            return ''
        self.tts_publisher.publish(String(data=safe))
        self.get_logger().info(f'[{turn_id}] Published TTS sentence: {safe}')
        return safe

    def _finish_tts_turn(self, turn_id):
        self.tts_publisher.publish(String(data=encode_turn_end(turn_id)))
        self.get_logger().info(f'[{turn_id}] TTS turn queued; waiting for playback completion.')

    def _retry_empty_answer(self, turn_id, user_prompt):
        """Retry once when the model returned only the ASR correction line."""
        self.get_logger().warning(
            f'[{turn_id}] LLM returned no answer text; retrying once without tools.'
        )
        retry_prompt = (
            f'用户说：{user_prompt}\n'
            '请直接用一到两句简短自然的中文回答。只输出回答正文，不要输出纠错标签、'
            '分析过程、Markdown、动作说明或任何前缀。'
        )
        settings = getattr(self.llm, 'settings', {})
        configured_tokens = settings.get('max_tokens', 0) if isinstance(settings, dict) else 0
        retry_tokens = configured_tokens if isinstance(configured_tokens, int) else 0
        if retry_tokens <= 0:
            retry_tokens = 256
        retry_tokens = min(max(retry_tokens, 128), 256)
        try:
            chunks = []
            for data in self.llm.chat_stream(
                retry_prompt,
                self._history_for_request(),
                tools_enabled=False,
                structured_answer=False,
                max_tokens_override=retry_tokens,
            ):
                if data.get('type') == 'text' and data.get('content'):
                    chunks.append(data['content'])
            raw = ''.join(chunks).strip()
            if not raw:
                return ''
            if '\n' not in raw and self._extract_corrected_text(raw):
                return ''
            answer = self._sanitize_speech_text(raw)
            if answer:
                self.get_logger().info(f'[{turn_id}] Empty-answer retry succeeded.')
            return answer
        except Exception as exc:
            self.get_logger().error(
                f'[{turn_id}] Empty-answer retry failed: {exc}\n{traceback.format_exc()}'
            )
            return ''

    @classmethod
    def _is_long_form_request(cls, user_prompt):
        return bool(cls.LONG_FORM_REQUEST_RE.search(user_prompt or ''))

    @classmethod
    def _needs_action_tools(cls, user_prompt):
        """Compatibility helper: ordinary dialogue always receives control tools.

        Dedicated visual inspection and empty-answer retry paths explicitly
        disable tools at their call sites, rather than relying on wording.
        """
        del user_prompt
        return True

    def _max_tokens_for_request(self, is_long_form):
        if not is_long_form:
            return None
        settings = getattr(self.llm, 'settings', {})
        configured_tokens = settings.get('max_tokens', 0) if isinstance(settings, dict) else 0
        if not isinstance(configured_tokens, int):
            configured_tokens = 0
        return max(self.LONG_FORM_MAX_TOKENS, configured_tokens)

    def _extract_corrected_text(self, first_line):
        first_line = (first_line or '').strip()
        if not first_line:
            return None

        cleaned = first_line.lstrip(' \t>*-#')
        cleaned = re.sub(
            r"^\s*\u7b2c\u4e00\u884c\s*[:\uff1a]\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        label = ''
        value = ''
        if cleaned.startswith('\u3010') and '\u3011' in cleaned:
            label, value = cleaned[1:].split('\u3011', 1)
        elif cleaned.startswith('[') and ']' in cleaned:
            label, value = cleaned[1:].split(']', 1)
        else:
            value = cleaned

        if label:
            label = label.strip().lower()
            if label not in self.CORRECTION_LABELS:
                return None
            value = value.lstrip(' \t:\uff1a')
            return value.strip().strip('"\u201c\u201d') or None

        # Fallbacks for plain labeled responses, e.g. corrected_text: hello.
        for sep in (':', '\uff1a'):
            if sep not in value:
                continue
            maybe_label, maybe_text = value.split(sep, 1)
            if maybe_label.strip().lower() in self.CORRECTION_LABELS:
                return maybe_text.strip().strip('"\u201c\u201d') or None

        for label in self.CORRECTION_LABELS:
            if value.lower().startswith(label.lower()):
                remainder = value[len(label):]
                if remainder and remainder[0] not in ' \t:\uff1a':
                    continue
                maybe_text = remainder.strip(' \t:\uff1a')
                return maybe_text.strip().strip('"\u201c\u201d') or None

        return None

    def _strip_correction_line(self, text):
        if '\n' not in text:
            return text
        first_line, rest = text.split('\n', 1)
        if self._extract_corrected_text(first_line):
            return self._strip_answer_prefix(rest)
        if self._is_correction_label_only(first_line):
            if '\n' not in rest:
                return ''
            _, answer = rest.split('\n', 1)
            return self._strip_answer_prefix(answer)
        return text

    def _sanitize_speech_text(self, text):
        clean = self._strip_answer_prefix(self._strip_correction_line(text)).strip()
        if '\n' not in clean and self._extract_corrected_text(clean):
            return ''
        return self.TTS_CLEAN_RE.sub('', clean).strip()

    def _is_correction_label_only(self, text):
        cleaned = (text or '').strip().lstrip(' \t>*-#')
        cleaned = cleaned.strip('[]\u3010\u3011').strip(' \t:\uff1a').lower()
        return cleaned in self.CORRECTION_LABELS

    @classmethod
    def _strip_answer_prefix(cls, text):
        return cls.OUTPUT_LINE_PREFIX_RE.sub('', text or '', count=1)

    def _publish_screen_dialog(self, turn_id, corrected_text, ai_text, actions, error=None):
        payload = {
            'turn_id': turn_id,
            'corrected_text': corrected_text or '',
            'ai_text': ai_text or '',
            'actions': actions or [],
        }
        if error:
            payload['error'] = error

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.screen_dialog_publisher.publish(msg)
        self.get_logger().info(f'[{turn_id}] Published atomic screen dialog.')

    def destroy_node(self):
        self._worker_running = False
        if hasattr(self, '_request_queue'):
            try:
                self._request_queue.put_nowait(None)
            except queue.Full:
                pass
        if hasattr(self, '_worker_thread') and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        self._close_game_stream()
        if getattr(self, 'tracking_tft_preview', None) is not None:
            self.tracking_tft_preview.stop()
        if getattr(self, 'tft_preview', None) is not None:
            self.tft_preview.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LLMBrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

