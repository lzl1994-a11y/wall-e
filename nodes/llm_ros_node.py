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

        py_list = pinyin(user_prompt, style=Style.NORMAL)
        py_str = ' '.join([item[0] for item in py_list])

        augmented_prompt = (
            f"\u3010\u539f\u59cb\u8bed\u97f3\u8bc6\u522b\u6587\u672c\u3011: \"{user_prompt}\"\n"
            f"\u3010\u53c2\u8003\u62fc\u97f3\u5bf9\u7167\u8868\u3011: {py_str}\n\n"
            f"\u3010\u6838\u5fc3\u6307\u4ee4\u4e0e\u8f93\u51fa\u89c4\u8303\u3011\uff08\u4f60\u5fc5\u987b\u4e25\u683c\u9075\u5b88\u4ee5\u4e0b2\u6761\u94c1\u5f8b\uff0c\u4e0d\u53ef\u504f\u5e9f\uff09\uff1a\n"
            f"1. \u7ea0\u9519\u4e0e\u9996\u884c\u62e6\u622a\u683c\u5f0f\uff1a\u4f60\u7684\u56de\u590d\u7684\u7b2c\u4e00\u884c\u5fc5\u987b\u662f\u7ea0\u6b63\u540e\u7684\u6807\u51c6\u6587\u672c\u3002"
            f"\u63a8\u8350\u683c\u5f0f\u4e3a\uff1a\u3010\u4fee\u6b63\u6587\u672c\u3011: [\u7ea0\u6b63\u540e\u7684\u6587\u672c]\uff0c\u968f\u540e\u7acb\u523b\u6362\u884c\u3002\n"
            f"2. \u610f\u56fe\u54cd\u5e94\uff1a\u4ece\u7b2c\u4e8c\u884c\u5f00\u59cb\uff0c\u6839\u636e\u771f\u5b9e\u610f\u56fe\u7ed9\u51fa\u6bd2\u820c\u56de\u590d\u3002\u5982\u679c\u7528\u6237\u8981\u6c42\u4f60\u505a\u52a8\u4f5c\uff08\u5982\u8f6c\u5934\u3001\u62ac\u773c\u7b49\uff09\uff0c\u4f60\u3010\u5fc5\u987b\u3011\u5728\u56de\u590d\u7684\u540c\u65f6\u8c03\u7528\u76f8\u5e94\u7684 Function Calling \u5de5\u5177\uff0c\u7edd\u5bf9\u4e0d\u80fd\u53ea\u8bf4\u8bdd\u4e0d\u52a8\u4f5c\uff01\n\n"
            f"\u8bf7\u5f00\u59cb\u4f60\u7684\u601d\u8003\u4e0e\u54cd\u5e94\uff1a"
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
                                sentence_buffer = parts[1] if len(parts) > 1 else ''
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

    def _extract_corrected_text(self, first_line):
        first_line = (first_line or '').strip()
        if not first_line:
            return None

        cleaned = first_line.lstrip(' \t>*-#')

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
            return rest
        return text

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

