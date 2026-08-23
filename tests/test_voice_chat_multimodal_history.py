import threading
import types
import unittest
from collections import deque
from unittest.mock import MagicMock

from services.voice_chat_service import VoiceChatService


class VoiceChatMultimodalHistoryTests(unittest.TestCase):
    def _service(self):
        service = VoiceChatService.__new__(VoiceChatService)
        service._chat_history = deque(maxlen=40)
        service._cancel_llm = threading.Event()
        service._last_llm_activity = 0.0
        return service

    def test_history_turns_are_always_user_assistant_pairs(self):
        service = self._service()

        service._append_history_turn("拍照", "好的，我看看。")
        service._append_history_turn("你好", "你好呀。")

        self.assertEqual(
            [message["role"] for message in service._validated_history()],
            ["user", "assistant", "user", "assistant"],
        )

    def test_legacy_assistant_only_history_is_cleared(self):
        service = self._service()
        service._chat_history.append({"role": "assistant", "content": "旧回答"})

        self.assertEqual(service._validated_history(), [])
        self.assertEqual(list(service._chat_history), [])

    def test_direct_answer_extracts_transcript_and_response(self):
        heard, response = VoiceChatService._direct_answer([{
            "name": "direct_answer",
            "arguments": {"heard_text": "  拍照  ", "response": " 好的。 "},
        }])

        self.assertEqual(heard, "拍照")
        self.assertEqual(response, "好的。")

    def test_missing_direct_answer_retries_without_action_tools(self):
        service = self._service()
        service.multimodal = MagicMock()
        audio_message = {"role": "user", "content": "audio"}
        service.multimodal.build_audio_message.return_value = audio_message
        service.system_prompt = "system"
        service.model = "test-model"
        service.on_llm_chunk = MagicMock()
        service.on_llm_reply = MagicMock()
        service.on_tool_call = MagicMock()
        service._llm_done = MagicMock()
        service._stream_tool_calls = MagicMock(side_effect=[
            ([{"name": "inspect_camera", "arguments": {"question": "前面有什么"}}], "普通文本"),
            ([{
                "name": "direct_answer",
                "arguments": {"heard_text": "看看前面", "response": "好的，我看看。"},
            }], ""),
        ])

        service._send_to_llm("encoded-audio")

        self.assertEqual(service._stream_tool_calls.call_count, 2)
        retry = service._stream_tool_calls.call_args_list[1].kwargs
        self.assertEqual(len(retry["tools"]), 1)
        service.on_llm_chunk.assert_called_once_with("好的，我看看。")
        service.on_llm_reply.assert_called_once_with("好的，我看看。")
        service.on_tool_call.assert_called_once_with(
            "inspect_camera", {"question": "前面有什么"}
        )
        self.assertEqual(
            [message["role"] for message in service._chat_history],
            ["user", "assistant"],
        )
        service._llm_done.assert_called_once_with()

    def test_second_missing_direct_answer_uses_audible_fallback(self):
        service = self._service()
        service.multimodal = MagicMock()
        service.multimodal.build_audio_message.return_value = {
            "role": "user", "content": "audio"
        }
        service.system_prompt = "system"
        service.model = "test-model"
        service.on_llm_chunk = MagicMock()
        service.on_llm_reply = MagicMock()
        service.on_tool_call = MagicMock()
        service._llm_done = MagicMock()
        service._stream_tool_calls = MagicMock(side_effect=[
            ([{"name": "play_sequence", "arguments": {"sequence_name": "wave_hello"}}], "raw"),
            ([], "raw"),
        ])

        service._send_to_llm("encoded-audio")

        service.on_llm_chunk.assert_called_once_with(service.FALLBACK_REPLY)
        service.on_llm_reply.assert_called_once_with(service.FALLBACK_REPLY)
        service.on_tool_call.assert_not_called()
        self.assertEqual(list(service._chat_history), [])


if __name__ == "__main__":
    unittest.main()
