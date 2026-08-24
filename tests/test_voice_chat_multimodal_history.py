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

    def test_validated_history_removes_image_blocks_from_storage(self):
        service = self._service()
        service._chat_history.extend([
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "刚才看到了什么"},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/jpeg;base64,old-image",
                    }},
                ],
            },
            {"role": "assistant", "content": "一个杯子。"},
        ])

        history = service._validated_history()

        self.assertEqual(history[0]["content"], "刚才看到了什么")
        self.assertNotIn("old-image", repr(history))
        self.assertNotIn("old-image", repr(list(service._chat_history)))

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

    def test_camera_skill_replaces_acknowledgement_with_visual_answer(self):
        service = self._service()
        service.multimodal = MagicMock()
        service.multimodal.build_audio_message.return_value = {
            "role": "user", "content": "audio"
        }
        service.system_prompt = "system"
        service.model = "test-model"
        service.on_llm_chunk = MagicMock()
        service.on_llm_reply = MagicMock()
        service.on_tool_call = MagicMock(return_value="前面有一只杯子。")
        service._llm_done = MagicMock()
        service._stream_tool_calls = MagicMock(return_value=([
            {
                "name": "direct_answer",
                "arguments": {"heard_text": "前面有什么", "response": "我看看。"},
            },
            {"name": "inspect_camera", "arguments": {"question": "前面有什么"}},
        ], ""))

        service._send_to_llm("encoded-audio")

        service.on_llm_chunk.assert_called_once_with("前面有一只杯子。")
        service.on_llm_reply.assert_called_once_with("前面有一只杯子。")
        self.assertEqual(service._chat_history[-1]["content"], "前面有一只杯子。")

    def test_image_analysis_forces_direct_answer_only(self):
        service = self._service()
        service.system_prompt = "system"
        service._stream_tool_calls = MagicMock(return_value=([
            {
                "name": "direct_answer",
                "arguments": {"response": "桌上有一只杯子。"},
            }
        ], ""))

        answer = service.analyze_image("桌上有什么", "aW1hZ2U=")

        self.assertEqual(answer, "桌上有一只杯子。")
        request = service._stream_tool_calls.call_args
        self.assertEqual(len(request.kwargs["tools"]), 1)
        image_url = request.args[0][1]["content"][1]["image_url"]["url"]
        self.assertEqual(image_url, "data:image/jpeg;base64,aW1hZ2U=")

    def test_photo_transcript_runs_capture_callback_before_reply(self):
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
        service.on_photo_request = MagicMock(return_value="拍好了，照片已经保存。")
        service.on_inspection_request = MagicMock()
        service._llm_done = MagicMock()
        service._stream_tool_calls = MagicMock(return_value=([
            {
                "name": "direct_answer",
                "arguments": {"heard_text": "给我拍个照", "response": "好的。"},
            },
            {"name": "inspect_camera", "arguments": {"question": "拍一张"}},
        ], ""))

        service._send_to_llm("encoded-audio")

        service.on_photo_request.assert_called_once_with()
        service.on_tool_call.assert_not_called()
        service.on_llm_reply.assert_called_once_with("拍好了，照片已经保存。")

    def test_inspection_transcript_runs_camera_without_model_tool_call(self):
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
        service.on_photo_request = MagicMock()
        service.on_inspection_request = MagicMock(return_value="前面有一只杯子。")
        service._llm_done = MagicMock()
        service._stream_tool_calls = MagicMock(return_value=([
            {
                "name": "direct_answer",
                "arguments": {"heard_text": "看一下前面有什么", "response": "好的。"},
            }
        ], ""))

        service._send_to_llm("encoded-audio")

        service.on_inspection_request.assert_called_once_with("看一下前面有什么")
        service.on_llm_reply.assert_called_once_with("前面有一只杯子。")

    def test_malformed_structured_answer_never_starts_camera(self):
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
        service.on_photo_request = MagicMock()
        service.on_inspection_request = MagicMock()
        service._llm_done = MagicMock()
        service._stream_tool_calls = MagicMock(side_effect=[
            ([{
                "name": "direct_answer",
                "arguments": {"heard_text": "给我拍个照"},
            }], ""),
            ([], ""),
        ])

        service._send_to_llm("encoded-audio")

        service.on_photo_request.assert_not_called()
        service.on_inspection_request.assert_not_called()
        service.on_llm_reply.assert_called_once_with(service.FALLBACK_REPLY)

    def test_photo_does_not_suppress_unrelated_action_tool(self):
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
        service.on_photo_request = MagicMock(return_value="拍好了。")
        service.on_inspection_request = MagicMock()
        service._llm_done = MagicMock()
        service._stream_tool_calls = MagicMock(return_value=([
            {
                "name": "direct_answer",
                "arguments": {"heard_text": "拍张照再挥手", "response": "好的。"},
            },
            {"name": "inspect_camera", "arguments": {"question": "拍照"}},
            {"name": "play_sequence", "arguments": {"sequence_name": "wave_hello"}},
        ], ""))

        service._send_to_llm("encoded-audio")

        service.on_tool_call.assert_called_once_with(
            "play_sequence", {"sequence_name": "wave_hello"}
        )

    def test_photo_callback_exception_returns_audible_failure(self):
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
        service.on_photo_request = MagicMock(side_effect=RuntimeError("camera error"))
        service.on_inspection_request = MagicMock()
        service._llm_done = MagicMock()
        service._stream_tool_calls = MagicMock(return_value=([
            {
                "name": "direct_answer",
                "arguments": {"heard_text": "给我拍个照", "response": "好的。"},
            }
        ], ""))

        service._send_to_llm("encoded-audio")

        service.on_llm_reply.assert_called_once_with(
            "这次没拍成功，请检查摄像头后再试。"
        )


if __name__ == "__main__":
    unittest.main()
