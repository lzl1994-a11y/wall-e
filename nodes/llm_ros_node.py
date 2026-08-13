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
from services.camera_frame import CameraFrameProvider, is_camera_inspection_request


class LLMBrainNode(Node):
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
        self.chat_history = deque(maxlen=40)  # 和 VoiceChatService 一致，防止 OOM
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
                self._publish_screen_dialog(
                    task.get('turn_id', ''),
                    task.get('user_prompt', ''),
                    '\u6211\u521a\u624d\u5904\u7406\u5931\u8d25\u4e86\uff0c\u7a0d\u540e\u518d\u8bd5\u3002',
                    [],
                    error=str(e),
                )
            finally:
                self._request_queue.task_done()

    def _process_voice_task(self, turn_id, user_prompt):
        # 通知 STT 节点暂停 ASR
        busy_msg = String()
        busy_msg.data = "busy"
        self.busy_publisher.publish(busy_msg)

        # 视觉查看是一个两阶段技能：先立即确认，再抓取一帧交给视觉模型。
        # 这样用户不会等待摄像头和第二次 LLM 请求时陷入沉默。
        if is_camera_inspection_request(user_prompt):
            self._process_camera_inspection(turn_id, user_prompt)
            return

        py_list = pinyin(user_prompt, style=Style.NORMAL)
        py_str = ' '.join([item[0] for item in py_list])

        augmented_prompt = (
            f"\u539f\u59cb ASR \u6587\u672c\uff1a{user_prompt}\n"
            f"\u62fc\u97f3\u53c2\u8003\uff1a{py_str}\n\n"
            "\u4e25\u683c\u6309\u4ee5\u4e0b\u4e24\u884c\u683c\u5f0f\u8f93\u51fa\uff0c\u4e0d\u5f97\u589e\u52a0\u5176\u4ed6\u5185\u5bb9\uff1a\n"
            "\u3010\u4fee\u6b63\u6587\u672c\u3011: <\u6839\u636e\u4e0a\u4e0b\u6587\u548c\u62fc\u97f3\u7ea0\u6b63\u540e\u7684\u5b8c\u6574\u53e5\u5b50>\n"
            "<\u53ef\u76f4\u63a5\u901a\u8fc7\u626c\u58f0\u5668\u64ad\u653e\u7684\u6700\u7ec8\u56de\u590d>\n\n"
            "\u7b2c\u4e8c\u884c\u6700\u591a\u4e24\u53e5\uff0c\u7b80\u77ed\u81ea\u7136\u3002\u4e0d\u8981\u8f93\u51fa\u5206\u6790\u3001\u601d\u8003\u3001\u8ba1\u5212\u3001\u89c4\u5219\u590d\u8ff0\u3001\u793a\u4f8b\u3001\u5217\u8868\u3001Markdown\u3001"
            "\u62ec\u53f7\u8bf4\u660e\u3001Function Calling \u5b57\u6837\u3001\u5de5\u5177\u540d\u6216\u5de5\u5177\u53c2\u6570\u3002"
            "\u4e0d\u8981\u8f93\u51fa\u201c\u7b2c\u4e00\u884c\u201d\u6216\u201c\u7b2c\u4e8c\u884c\u201d\u524d\u7f00\u3002"
            "\u9700\u8981\u52a8\u4f5c\u65f6\u53ea\u4f7f\u7528\u539f\u751f\u5de5\u5177\u8c03\u7528\uff0c\u4e0d\u8981\u5728\u6587\u5b57\u4e2d\u63cf\u8ff0\u8c03\u7528\u8fc7\u7a0b\u3002"
            "\u76f4\u63a5\u7ed9\u51fa\u4e24\u884c\u7ed3\u679c\u3002"
        )

        self.get_logger().info(f'[{turn_id}] Sending request to LLM...')

        text_buffer = ''
        sentence_buffer = ''
        punc_count = 0
        corrected_text_extracted = False
        corrected_text = ''
        corrected_text_published = False
        actions = []

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

        try:
            stream = self.llm.chat_stream(augmented_prompt, list(self.chat_history))

            for data in stream:
                data_type = data.get('type')

                if data_type == 'text':
                    chunk = data.get('content', '')
                    text_buffer += chunk
                    sentence_buffer += chunk

                    # Keep the existing first-line split idea, but accept several label variants.
                    if not corrected_text_extracted:
                        if '\n' in sentence_buffer:
                            parts = sentence_buffer.split('\n', 1)
                            first_line = parts[0].strip()
                            extracted = self._extract_corrected_text(first_line)

                            if extracted:
                                publish_corrected(extracted)
                                sentence_buffer = self._strip_answer_prefix(
                                    parts[1] if len(parts) > 1 else ''
                                )
                            else:
                                publish_corrected(user_prompt)

                            corrected_text_extracted = True
                        elif len(sentence_buffer) > 60:
                            publish_corrected(user_prompt)
                            corrected_text_extracted = True
                        else:
                            continue

                    for char in chunk:
                        if not corrected_text_extracted:
                            break

                        if char in self.punctuations:
                            punc_count += 1

                        if punc_count >= 2:
                            clean_sentence = sentence_buffer.strip()
                            tts_safe = self.TTS_CLEAN_RE.sub('', clean_sentence)

                            if tts_safe.strip():
                                out_msg = String()
                                out_msg.data = tts_safe.strip()
                                self.tts_publisher.publish(out_msg)
                                self.get_logger().info(f'[{turn_id}] Published TTS sentence: {out_msg.data}')

                            sentence_buffer = ''
                            punc_count = 0

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

        except Exception as e:
            self.get_logger().error(f'[{turn_id}] LLM request/stream failed: {e}\n{traceback.format_exc()}')
            if not corrected_text_published:
                publish_corrected(user_prompt)
            failure_text = '\u6211\u521a\u624d\u5904\u7406\u5931\u8d25\u4e86\uff0c\u7a0d\u540e\u518d\u8bd5\u3002'
            self._publish_screen_dialog(turn_id, corrected_text or user_prompt, failure_text, actions, error=str(e))
            idle_msg = String()
            idle_msg.data = "idle"
            self.busy_publisher.publish(idle_msg)
            return

        if not corrected_text_published:
            publish_corrected(user_prompt)

        clean_tail = sentence_buffer.strip()
        if clean_tail:
            tts_safe_tail = self.TTS_CLEAN_RE.sub('', clean_tail)
            if tts_safe_tail.strip():
                out_msg = String()
                out_msg.data = tts_safe_tail.strip()
                self.tts_publisher.publish(out_msg)
                self.get_logger().info(f'[{turn_id}] Published TTS tail: {out_msg.data}')

        final_user_memory = corrected_text if corrected_text else user_prompt
        self.chat_history.append({'role': 'user', 'content': final_user_memory})

        clean_assistant_memory = self._strip_correction_line(text_buffer)
        clean_text = clean_assistant_memory.strip()
        
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

        # 通知 STT 节点恢复 ASR
        idle_msg = String()
        idle_msg.data = "idle"
        self.busy_publisher.publish(idle_msg)

    def _process_camera_inspection(self, turn_id, user_prompt):
        """确认 → 抓帧 → 视觉 LLM → 播报结果。"""
        confirmation = '好的，我看一下。'
        self._publish_tts(confirmation, turn_id)
        self._publish_screen_dialog(turn_id, user_prompt, confirmation, [])

        frame = self.camera_frames.capture(timeout=1.5)
        if not frame:
            failure = '我现在看不到画面，检查一下摄像头连接。'
            self._publish_tts(failure, turn_id)
            self._publish_screen_dialog(
                turn_id, user_prompt, failure, [], error='camera_frame_unavailable'
            )
            self._set_llm_idle()
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
            self._set_llm_idle()

    def _visual_history(self):
        """只保留文本形式的最近上下文，避免把旧 tool 消息传给视觉模型。"""
        history = []
        for item in list(self.chat_history)[-8:]:
            if item.get('role') not in {'user', 'assistant'}:
                continue
            if isinstance(item.get('content'), str) and item['content'].strip():
                history.append({'role': item['role'], 'content': item['content']})
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
        safe = self.TTS_CLEAN_RE.sub('', (text or '').strip())
        if not safe:
            return
        self.tts_publisher.publish(String(data=safe))
        self.get_logger().info(f'[{turn_id}] Published TTS sentence: {safe}')

    def _set_llm_idle(self):
        self.busy_publisher.publish(String(data='idle'))

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
                maybe_text = value[len(label):].strip(' \t:\uff1a')
                return maybe_text.strip().strip('"\u201c\u201d') or None

        return None

    def _strip_correction_line(self, text):
        if '\n' not in text:
            return text
        first_line, rest = text.split('\n', 1)
        if self._extract_corrected_text(first_line):
            return self._strip_answer_prefix(rest)
        return text

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

