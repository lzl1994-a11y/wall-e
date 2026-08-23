#!/usr/bin/env python3
import json
import queue
import re
import threading
import traceback
import uuid
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pypinyin import Style, pinyin

from services.llm_service import LLMService
from services.camera_frame import (
    CameraFrameProvider,
    is_camera_inspection_request,
    is_camera_photo_request,
    save_camera_photo,
)
from services.tft_preview_server import (
    PreviewResult,
    TftPreviewServer,
    load_tft_preview_settings,
)
from services.tts_protocol import encode_turn_end


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
        self.camera_frames = CameraFrameProvider(self)
        self.tft_preview_ready_publisher = self.create_publisher(
            String, 'tft_preview_ready', 10
        )
        self.tft_preview_settings = load_tft_preview_settings()
        self.tft_preview = TftPreviewServer(
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

    def voice_callback(self, msg):
        """Queue the request so the ROS callback thread is never blocked by LLM I/O."""
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
                self._request_queue.task_done()

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
        spoken_parts = []

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

                elif data_type == 'tool_call':
                    if data.get('name') == 'inspect_camera':
                        self.get_logger().info(f'[{turn_id}] Camera inspection tool requested.')
                        self._process_camera_inspection(turn_id, user_prompt)
                        return
                    action_payload = {
                        'turn_id': turn_id,
                        'name': data.get('name'),
                        'arguments': data.get('arguments', '{}'),
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
        
        assistant_msg = {'role': 'assistant', 'content': clean_text}
        
        # If tools were called, we must append them to the assistant message in OpenAI format
        # and also provide a mock 'tool' response to satisfy the conversation schema.
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
            assistant_msg['tool_calls'] = openai_tool_calls

        self.chat_history.append(assistant_msg)

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
                    'content': '{"status": "success"}'
                })

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
                structured_answer=True,
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
        return preview_service.send_camera_preview(
            self.camera_frames,
            duration_ms=duration_ms,
            hold_ms=self.tft_preview_settings.hold_ms,
            fps=self.tft_preview_settings.fps,
        )

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
        history = list(self.chat_history)[-self.CHAT_HISTORY_MESSAGES:]
        while history and history[0].get('role') != 'user':
            history.pop(0)
        return history

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
                structured_answer=True,
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

