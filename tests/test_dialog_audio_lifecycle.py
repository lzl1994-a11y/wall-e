import base64
import importlib
import json
import queue
import sys
import threading
import types
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from services.playback_service import PlaybackService
from services.stt_service import STTService
from services.tts_protocol import decode_turn_end, encode_turn_end


class TTSProtocolTests(unittest.TestCase):
    def test_turn_end_marker_round_trips_without_matching_plain_speech(self):
        marker = encode_turn_end("turn-123")

        self.assertEqual(decode_turn_end(marker), "turn-123")
        self.assertIsNone(decode_turn_end("你好，我是普通播报文本。"))


class PlaybackLifecycleTests(unittest.TestCase):
    def test_empty_queue_writes_silence_while_output_stream_is_open(self):
        player = PlaybackService.__new__(PlaybackService)
        player.sample_rate = 1000
        player._queue = queue.Queue()
        player._stream = MagicMock()

        player._play_next_item()

        player._stream.write.assert_called_once()
        idle_audio = player._stream.write.call_args.args[0]
        self.assertEqual(idle_audio.shape, (20, 1))
        self.assertTrue(np.all(idle_audio == 0.0))

    def test_turn_complete_runs_after_preceding_audio_finishes(self):
        events = []
        player = PlaybackService.__new__(PlaybackService)
        player.sample_rate = 48000
        player._device = 2
        player._stream = None
        player.on_turn_complete = lambda: events.append("complete")
        samples = np.array([1, -2, 3], dtype=np.int16)
        stream = MagicMock()
        writes = []

        def record_write(audio):
            writes.append(audio.copy())
            events.append("queued")

        stream.write.side_effect = record_write
        stream.stop.side_effect = lambda: events.append("played")

        with (
            patch.object(player, "_refresh_device", return_value=True),
            patch("services.playback_service.sd.OutputStream", return_value=stream),
        ):
            player._play_item(samples)
            player._play_item(PlaybackService._TURN_END)

        self.assertEqual(events, ["queued", "queued", "played", "complete"])
        self.assertEqual(writes[-1].shape, (4800, 1))
        self.assertTrue(np.all(writes[-1] == 0.0))
        stream.start.assert_called_once_with()
        stream.close.assert_called_once_with()

    def test_multiple_segments_share_one_output_stream(self):
        player = PlaybackService.__new__(PlaybackService)
        player.sample_rate = 48000
        player._device = 2
        player._stream = None
        player.on_turn_complete = None
        stream = MagicMock()

        with (
            patch.object(player, "_refresh_device", return_value=True),
            patch("services.playback_service.sd.OutputStream", return_value=stream) as stream_class,
        ):
            player._play_item(np.array([1, 2], dtype=np.int16))
            player._play_item(np.array([3, 4], dtype=np.int16))
            player._play_item(PlaybackService._TURN_END)

        stream_class.assert_called_once()
        self.assertEqual(stream.write.call_count, 3)
        np.testing.assert_allclose(
            stream.write.call_args_list[0].args[0].reshape(-1),
            np.array([1, 2], dtype=np.float32) / 32768.0,
        )
        np.testing.assert_allclose(
            stream.write.call_args_list[1].args[0].reshape(-1),
            np.array([3, 4], dtype=np.float32) / 32768.0,
        )
        final_write = stream.write.call_args_list[-1].args[0]
        self.assertEqual(final_write.shape, (4800, 1))
        self.assertTrue(np.all(final_write == 0.0))


class STTTimerLifecycleTests(unittest.TestCase):
    def _service(self):
        service = STTService.__new__(STTService)
        service._pipe = MagicMock()
        service._awake = True
        service._awake_timeout = 8.0
        service._awake_timer = MagicMock()
        service._awake_lock = threading.Lock()
        service._awake_timer_generation = 0
        return service

    def test_pause_cancels_timer_and_resume_starts_a_fresh_eight_seconds(self):
        service = self._service()
        old_timer = service._awake_timer

        service.pause()

        old_timer.cancel.assert_called_once_with()
        self.assertIsNone(service._awake_timer)

        with patch("services.stt_service.threading.Timer") as timer_class:
            new_timer = timer_class.return_value
            service.resume()

        timer_class.assert_called_once_with(
            8.0,
            service._on_awake_timeout,
            args=(service._awake_timer_generation,),
        )
        new_timer.start.assert_called_once_with()
        service._pipe.pause.assert_called_once_with()
        service._pipe.resume.assert_called_once_with()

    def test_cancelled_timer_callback_cannot_close_a_resumed_session(self):
        service = self._service()
        service._awake_timer_generation = 4

        service._on_awake_timeout(3)

        self.assertTrue(service._awake)
        service._pipe.set_awake.assert_not_called()

        service._on_awake_timeout(4)

        self.assertFalse(service._awake)
        service._pipe.set_awake.assert_called_once_with(False)


class LLMEmptyAnswerTests(unittest.TestCase):
    @staticmethod
    def _load_node_class():
        fake_rclpy = types.ModuleType("rclpy")
        fake_rclpy_node = types.ModuleType("rclpy.node")
        fake_rclpy_node.Node = object

        class String:
            def __init__(self, data=""):
                self.data = data

        class UInt8MultiArray(String):
            pass

        fake_std_msgs = types.ModuleType("std_msgs")
        fake_std_msgs_msg = types.ModuleType("std_msgs.msg")
        fake_std_msgs_msg.String = String
        fake_std_msgs_msg.UInt8MultiArray = UInt8MultiArray

        fake_pypinyin = types.ModuleType("pypinyin")
        fake_pypinyin.Style = types.SimpleNamespace(NORMAL="normal")
        fake_pypinyin.pinyin = lambda text, style=None: [[text]]

        fake_llm_service = types.ModuleType("services.llm_service")
        fake_llm_service.LLMService = object
        fake_camera = types.ModuleType("services.camera_frame")
        fake_camera.CameraFrameProvider = object
        fake_camera.is_camera_inspection_request = lambda _text: False
        fake_camera.is_camera_photo_request = lambda _text: False
        fake_camera.save_camera_photo = lambda _jpeg, _directory: None

        modules = {
            "rclpy": fake_rclpy,
            "rclpy.node": fake_rclpy_node,
            "std_msgs": fake_std_msgs,
            "std_msgs.msg": fake_std_msgs_msg,
            "pypinyin": fake_pypinyin,
            "services.llm_service": fake_llm_service,
            "services.camera_frame": fake_camera,
        }
        sys.modules.pop("nodes.llm_ros_node", None)
        with patch.dict(sys.modules, modules):
            module = importlib.import_module("nodes.llm_ros_node")
        return module.LLMBrainNode

    def test_correction_only_response_retries_and_waits_for_playback_idle(self):
        node_class = self._load_node_class()
        node = node_class.__new__(node_class)
        node.llm = MagicMock()
        node.llm.chat_stream.side_effect = [
            iter([{"type": "text", "content": "【修正文本】: 你今天怎么样？\n"}]),
            iter([{"type": "text", "content": "我挺好的，还能陪你聊两句。"}]),
        ]
        node.chat_history = deque(maxlen=40)
        node.punctuations = {'。', '？', '.', '?', '！', '!'}
        node.tts_publisher = MagicMock()
        node.action_publisher = MagicMock()
        node.corrected_publisher = MagicMock()
        node.full_ai_publisher = MagicMock()
        node.screen_dialog_publisher = MagicMock()
        node.busy_publisher = MagicMock()
        logger = MagicMock()
        node.get_logger = lambda: logger

        node._process_voice_task("turn-2", "你今天怎么样？")

        spoken_message, end_message = [
            call.args[0].data for call in node.tts_publisher.publish.call_args_list
        ]
        self.assertEqual(spoken_message, "我挺好的，还能陪你聊两句。")
        self.assertEqual(decode_turn_end(end_message), "turn-2")
        self.assertEqual(
            [call.args[0].data for call in node.busy_publisher.publish.call_args_list],
            ["busy"],
        )
        dialog = json.loads(node.screen_dialog_publisher.publish.call_args.args[0].data)
        self.assertEqual(dialog["ai_text"], spoken_message)
        self.assertEqual(len(node.llm.chat_stream.call_args_list), 2)
        initial_prompt = node.llm.chat_stream.call_args_list[0].args[0]
        self.assertIn("\u9759\u9ed8\u7406\u89e3", initial_prompt)
        self.assertNotIn("\u3010\u4fee\u6b63\u6587\u672c\u3011", initial_prompt)
        self.assertEqual(
            node.llm.chat_stream.call_args_list[1].kwargs["max_tokens_override"],
            256,
        )
        logger.warning.assert_called_once()
        sys.modules.pop("nodes.llm_ros_node", None)

    def test_streamed_answer_reaches_tts_before_llm_stream_finishes(self):
        node_class = self._load_node_class()
        node = node_class.__new__(node_class)
        node.llm = MagicMock()
        node.chat_history = deque(maxlen=40)
        node.punctuations = {'\u3002', '\uff1f', '.', '?', '\uff01', '!'}
        node.tts_publisher = MagicMock()
        node.action_publisher = MagicMock()
        node.corrected_publisher = MagicMock()
        node.full_ai_publisher = MagicMock()
        node.screen_dialog_publisher = MagicMock()
        node.busy_publisher = MagicMock()
        node.get_logger = lambda: MagicMock()

        tts_was_published_before_done = []

        def response_stream():
            yield {"type": "text", "content": "\u4f60\u597d\uff0c\u6211\u5728\u3002"}
            tts_was_published_before_done.append(
                node.tts_publisher.publish.called
            )
            yield {"type": "done", "finish_reason": "stop"}

        node.llm.chat_stream.return_value = response_stream()

        node._process_voice_task("turn-direct", "\u4f60\u597d")

        self.assertEqual(node.llm.chat_stream.call_count, 1)
        self.assertEqual(tts_was_published_before_done, [True])
        self.assertTrue(node.llm.chat_stream.call_args.kwargs["tools_enabled"])
        self.assertEqual(node.corrected_publisher.publish.call_args.args[0].data, "\u4f60\u597d")
        messages = [call.args[0].data for call in node.tts_publisher.publish.call_args_list]
        self.assertEqual(messages[0], "\u4f60\u597d\uff0c\u6211\u5728\u3002")
        self.assertEqual(decode_turn_end(messages[1]), "turn-direct")
        sys.modules.pop("nodes.llm_ros_node", None)

    def test_conditional_intent_cannot_fall_back_to_unverified_speech(self):
        node_class = self._load_node_class()
        node = node_class.__new__(node_class)
        node.llm = MagicMock()
        node.llm.chat_stream.return_value = iter([
            {"type": "text", "content": "我看到了，所以没有点头。"},
            {"type": "done", "finish_reason": "stop"},
        ])
        node.chat_history = deque(maxlen=40)
        node.tts_publisher = MagicMock()
        node.corrected_publisher = MagicMock()
        node.full_ai_publisher = MagicMock()
        node.screen_dialog_publisher = MagicMock()
        node.busy_publisher = MagicMock()
        node.get_logger = lambda: MagicMock()

        node._process_voice_task(
            "turn-conditional",
            "看看前面，如果画面是空白的你就点头。",
        )

        self.assertEqual(node.llm.chat_stream.call_count, 2)
        request = node.llm.chat_stream.call_args_list[0]
        self.assertEqual(
            request.kwargs["only_action_name"], "run_conditional_task"
        )
        messages = [
            call.args[0].data for call in node.tts_publisher.publish.call_args_list
        ]
        self.assertEqual(
            messages[0],
            "这个条件任务没有生成可执行计划，所以我没有观察或执行动作。",
        )
        self.assertNotIn("我看到了", "".join(messages))
        self.assertEqual(decode_turn_end(messages[1]), "turn-conditional")
        sys.modules.pop("nodes.llm_ros_node", None)

    def test_photo_preview_saves_last_frame_without_calling_llm(self):
        node_class = self._load_node_class()
        node = node_class.__new__(node_class)
        node.llm = MagicMock()
        node.chat_history = deque(maxlen=40)
        node.tts_publisher = MagicMock()
        node.full_ai_publisher = MagicMock()
        node.screen_dialog_publisher = MagicMock()
        node.tft_preview_settings = types.SimpleNamespace(
            photo_duration_ms=3000,
            recognition_duration_ms=1500,
            hold_ms=3000,
            fps=10,
            photo_directory="/tmp/wali-photos",
        )
        node.tft_preview = MagicMock()
        from services.tft_preview_server import PreviewResult
        node.tft_preview.send_camera_preview.return_value = PreviewResult(
            last_frame=b"\xff\xd8photo\xff\xd9"
        )
        node.camera_frames = MagicMock()
        node.get_logger = lambda: MagicMock()

        save = MagicMock(return_value=Path("/tmp/photo.jpg"))
        with patch.dict(
            node_class._process_camera_photo.__globals__,
            {"save_camera_photo": save},
        ):
            node._process_camera_photo("turn-photo", "帮我拍张照片")

        node.llm.chat_stream.assert_not_called()
        save.assert_called_once_with(b"\xff\xd8photo\xff\xd9", "/tmp/wali-photos")
        preview_call = node.tft_preview.send_camera_preview.call_args
        self.assertEqual(preview_call.kwargs["duration_ms"], 3000)
        self.assertEqual(preview_call.kwargs["hold_ms"], 3000)
        messages = [call.args[0].data for call in node.tts_publisher.publish.call_args_list]
        self.assertEqual(messages[:2], ["好的，准备拍照。", "拍好了，照片已经保存。"])
        self.assertEqual(decode_turn_end(messages[2]), "turn-photo")
        sys.modules.pop("nodes.llm_ros_node", None)

    def test_inspection_previews_for_1500ms_then_sends_last_frame_to_llm(self):
        node_class = self._load_node_class()
        node = node_class.__new__(node_class)
        node.llm = MagicMock()
        node.llm.chat_stream.return_value = iter([
            {"type": "text", "content": "前面是一只杯子。"},
            {"type": "done", "finish_reason": "stop"},
        ])
        node.chat_history = deque(maxlen=40)
        node.tts_publisher = MagicMock()
        node.full_ai_publisher = MagicMock()
        node.screen_dialog_publisher = MagicMock()
        node.tft_preview_settings = types.SimpleNamespace(
            photo_duration_ms=3000,
            recognition_duration_ms=1500,
            hold_ms=3000,
            fps=10,
            photo_directory="/tmp/wali-photos",
        )
        node.tft_preview = MagicMock()
        from services.tft_preview_server import PreviewResult
        node.tft_preview.send_camera_preview.return_value = PreviewResult(
            last_frame=b"\xff\xd8vision\xff\xd9"
        )
        node.camera_frames = MagicMock()
        node.get_logger = lambda: MagicMock()

        node._process_camera_inspection("turn-vision", "帮我看看这是什么")

        preview_call = node.tft_preview.send_camera_preview.call_args
        self.assertEqual(preview_call.kwargs["duration_ms"], 1500)
        llm_call = node.llm.chat_stream.call_args
        expected_image = base64.b64encode(b"\xff\xd8vision\xff\xd9").decode("ascii")
        self.assertEqual(llm_call.kwargs["image_base64"], expected_image)
        self.assertFalse(llm_call.kwargs["structured_answer"])
        messages = [call.args[0].data for call in node.tts_publisher.publish.call_args_list]
        self.assertEqual(messages[0], "前面是一只杯子。")
        self.assertEqual(decode_turn_end(messages[1]), "turn-vision")
        sys.modules.pop("nodes.llm_ros_node", None)

    def test_first_long_comma_clause_is_published_before_sentence_end(self):
        node_class = self._load_node_class()
        node = node_class.__new__(node_class)
        node.llm = MagicMock()
        node.llm.chat_stream.return_value = iter([
            {"type": "text", "content": "这是白居易琵琶行的开头两句，后面描写秋夜送客。"},
        ])
        node.chat_history = deque(maxlen=40)
        node.punctuations = {'。', '？', '.', '?', '！', '!'}
        node.tts_publisher = MagicMock()
        node.action_publisher = MagicMock()
        node.corrected_publisher = MagicMock()
        node.full_ai_publisher = MagicMock()
        node.screen_dialog_publisher = MagicMock()
        node.busy_publisher = MagicMock()
        node.get_logger = lambda: MagicMock()

        node._process_voice_task("turn-clause", "这句诗是什么意思")

        messages = [call.args[0].data for call in node.tts_publisher.publish.call_args_list]
        self.assertEqual(
            messages[:2],
            ["这是白居易琵琶行的开头两句，", "后面描写秋夜送客。"],
        )
        self.assertEqual(decode_turn_end(messages[2]), "turn-clause")
        sys.modules.pop("nodes.llm_ros_node", None)

    def test_recitation_request_uses_long_form_prompt_and_token_budget(self):
        node_class = self._load_node_class()
        node = node_class.__new__(node_class)
        node.llm = MagicMock()
        node.llm.settings = {"max_tokens": 256}
        node.llm.chat_stream.return_value = iter([
            {"type": "text", "content": "浔阳江头夜送客。枫叶荻花秋瑟瑟。"},
            {"type": "done", "finish_reason": "stop"},
        ])
        node.chat_history = deque(maxlen=40)
        node.punctuations = {'。', '？', '.', '?', '！', '!'}
        node.tts_publisher = MagicMock()
        node.action_publisher = MagicMock()
        node.corrected_publisher = MagicMock()
        node.full_ai_publisher = MagicMock()
        node.screen_dialog_publisher = MagicMock()
        node.busy_publisher = MagicMock()
        logger = MagicMock()
        node.get_logger = lambda: logger

        node._process_voice_task("turn-poem", "背一下琵琶行")

        request = node.llm.chat_stream.call_args
        self.assertIn("连续完整输出", request.args[0])
        self.assertEqual(request.kwargs["max_tokens_override"], 2048)
        messages = [call.args[0].data for call in node.tts_publisher.publish.call_args_list]
        self.assertEqual(messages[:2], ["浔阳江头夜送客。", "枫叶荻花秋瑟瑟。"])
        self.assertEqual(decode_turn_end(messages[2]), "turn-poem")
        logger.info.assert_any_call(
            "[turn-poem] LLM stream completed: finish_reason=stop"
        )
        sys.modules.pop("nodes.llm_ros_node", None)

    def test_stream_completion_can_change_severity_between_turns(self):
        """Guard against rclpy's source-location based severity restriction."""
        import inspect

        class SourceAwareLogger:
            def __init__(self):
                self.severities = {}

            def _log(self, severity, _message):
                caller = inspect.currentframe().f_back.f_back
                location = (caller.f_code.co_filename, caller.f_lineno)
                previous = self.severities.setdefault(location, severity)
                if previous != severity:
                    raise ValueError("Logger severity cannot be changed between calls.")

            def info(self, message):
                self._log("info", message)

            def warning(self, message):
                self._log("warning", message)

            def error(self, message):
                self._log("error", message)

        node_class = self._load_node_class()
        node = node_class.__new__(node_class)
        node.llm = MagicMock()
        node.llm.settings = {"max_tokens": 512}
        node.llm.chat_stream.side_effect = [
            iter([{"type": "text", "content": "第一轮。"}, {"type": "done", "finish_reason": "stop"}]),
            iter([{"type": "text", "content": "第二轮。"}, {"type": "done", "finish_reason": "length"}]),
        ]
        node.chat_history = deque(maxlen=40)
        node.punctuations = {'。', '？', '.', '?', '！', '!'}
        node.tts_publisher = MagicMock()
        node.action_publisher = MagicMock()
        node.corrected_publisher = MagicMock()
        node.full_ai_publisher = MagicMock()
        node.screen_dialog_publisher = MagicMock()
        node.busy_publisher = MagicMock()
        logger = SourceAwareLogger()
        node.get_logger = lambda: logger

        node._process_voice_task("turn-stop", "你好")
        node._process_voice_task("turn-length", "你好")

        self.assertEqual(set(logger.severities.values()), {"info", "warning"})
        sys.modules.pop("nodes.llm_ros_node", None)

    def test_correction_only_response_without_newline_is_not_spoken(self):
        node_class = self._load_node_class()
        node = node_class.__new__(node_class)
        node.llm = MagicMock()
        node.llm.chat_stream.side_effect = [
            iter([{"type": "text", "content": "【修正文本】: 你好"}]),
            iter([{"type": "text", "content": "你好，又来找我了？"}]),
        ]
        node.chat_history = deque(maxlen=40)
        node.punctuations = {'。', '？', '.', '?', '！', '!'}
        node.tts_publisher = MagicMock()
        node.action_publisher = MagicMock()
        node.corrected_publisher = MagicMock()
        node.full_ai_publisher = MagicMock()
        node.screen_dialog_publisher = MagicMock()
        node.busy_publisher = MagicMock()
        node.get_logger = lambda: MagicMock()

        node._process_voice_task("turn-3", "你好")

        messages = [call.args[0].data for call in node.tts_publisher.publish.call_args_list]
        self.assertEqual(messages[0], "你好，又来找我了？")
        self.assertNotIn("修正文本", messages[0])
        self.assertEqual(decode_turn_end(messages[1]), "turn-3")
        sys.modules.pop("nodes.llm_ros_node", None)

    def test_legacy_correction_metadata_is_never_spoken(self):
        node_class = self._load_node_class()
        node = node_class.__new__(node_class)

        self.assertEqual(
            node._sanitize_speech_text("\u4fee\u6b63\u6587\u672c \u80cc\u4e00\u4e0b\u7435\u7436\u884c\u3002\n\u6d54\u9633\u6c5f\u5934\u591c\u9001\u5ba2\u3002"),
            "\u6d54\u9633\u6c5f\u5934\u591c\u9001\u5ba2\u3002",
        )
        self.assertEqual(
            node._sanitize_speech_text("\u4fee\u6b63\u6587\u672c\n\u80cc\u4e00\u4e0b\u7435\u7436\u884c\u3002\n\u6d54\u9633\u6c5f\u5934\u591c\u9001\u5ba2\u3002"),
            "\u6d54\u9633\u6c5f\u5934\u591c\u9001\u5ba2\u3002",
        )
        self.assertEqual(
            node._sanitize_speech_text("\u4fee\u6b63\u6587\u672c\u662f\u4ec0\u4e48\u610f\u601d\uff1f"),
            "\u4fee\u6b63\u6587\u672c\u662f\u4ec0\u4e48\u610f\u601d\uff1f",
        )
        sys.modules.pop("nodes.llm_ros_node", None)

    def test_tools_are_enabled_for_all_ordinary_semantic_requests(self):
        node_class = self._load_node_class()

        self.assertTrue(node_class._needs_action_tools("今天的天气怎么样"))
        self.assertTrue(node_class._needs_action_tools("背一下琵琶行"))
        self.assertTrue(node_class._needs_action_tools("旋转头"))
        self.assertTrue(node_class._needs_action_tools("转个头"))
        self.assertTrue(node_class._needs_action_tools("向前走一下"))
        self.assertTrue(node_class._needs_action_tools("看着我"))
        self.assertTrue(node_class._needs_action_tools("挥挥手"))
        sys.modules.pop("nodes.llm_ros_node", None)

    def test_semantic_head_turn_tool_call_is_published_as_action_cmd(self):
        node_class = self._load_node_class()
        node = node_class.__new__(node_class)
        node.llm = MagicMock()
        node.llm.chat_stream.return_value = iter([
            {"type": "text", "content": "好的。"},
            {"type": "tool_call", "name": "play_sequence", "arguments": '{"sequence_name":"turn_head_left"}'},
            {"type": "done", "finish_reason": "tool_calls"},
        ])
        node.chat_history = deque(maxlen=40)
        node.punctuations = {'。', '？', '.', '?', '！', '!'}
        node.tts_publisher = MagicMock()
        node.action_publisher = MagicMock()
        node.corrected_publisher = MagicMock()
        node.full_ai_publisher = MagicMock()
        node.screen_dialog_publisher = MagicMock()
        node.busy_publisher = MagicMock()
        node.get_logger = lambda: MagicMock()

        node._process_voice_task("turn-head", "转个头")

        self.assertTrue(node.llm.chat_stream.call_args.kwargs["tools_enabled"])
        published = json.loads(node.action_publisher.publish.call_args.args[0].data)
        self.assertEqual(published["turn_id"], "turn-head")
        self.assertEqual(published["name"], "play_sequence")
        # sequence_ros_node accepts this string form and parses it to the same dict.
        self.assertEqual(json.loads(published["arguments"]), {"sequence_name": "turn_head_left"})
        sys.modules.pop("nodes.llm_ros_node", None)

    def test_tracking_stop_bypasses_model_and_publishes_idle_action(self):
        node_class = self._load_node_class()
        node = node_class.__new__(node_class)
        node.llm = MagicMock()
        node.chat_history = deque(maxlen=40)
        node.tts_publisher = MagicMock()
        node.action_publisher = MagicMock()
        node.corrected_publisher = MagicMock()
        node.full_ai_publisher = MagicMock()
        node.screen_dialog_publisher = MagicMock()
        node.busy_publisher = MagicMock()
        node.get_logger = lambda: MagicMock()

        node._process_voice_task("turn-stop", "不要看我了。")

        node.llm.chat_stream.assert_not_called()
        published = json.loads(node.action_publisher.publish.call_args.args[0].data)
        self.assertEqual(published["name"], "set_tracking_mode")
        self.assertEqual(json.loads(published["arguments"]), {"mode": "idle"})
        self.assertEqual(node.tts_publisher.publish.call_args_list[0].args[0].data, "好的，已停止跟随。")
        self.assertEqual(node.corrected_publisher.publish.call_args.args[0].data, "不要看我了。")
        sys.modules.pop("nodes.llm_ros_node", None)

    def test_capability_question_tool_proposal_is_not_published(self):
        node_class = self._load_node_class()
        node = node_class.__new__(node_class)
        node.llm = MagicMock()
        node.llm.chat_stream.return_value = iter([
            {"type": "text", "content": "我可以转头呀。"},
            {"type": "tool_call", "name": "play_sequence", "arguments": '{"sequence_name":"turn_head_left"}'},
            {"type": "done", "finish_reason": "tool_calls"},
        ])
        node.chat_history = deque(maxlen=40)
        node.punctuations = {'。', '？', '.', '?', '！', '!'}
        node.tts_publisher = MagicMock()
        node.action_publisher = MagicMock()
        node.corrected_publisher = MagicMock()
        node.full_ai_publisher = MagicMock()
        node.screen_dialog_publisher = MagicMock()
        node.busy_publisher = MagicMock()
        logger = MagicMock()
        node.get_logger = lambda: logger

        node._process_voice_task("turn-question", "你能转头吗？")

        node.action_publisher.publish.assert_not_called()
        self.assertEqual(
            node.tts_publisher.publish.call_args_list[0].args[0].data,
            "我可以转头呀。",
        )
        self.assertTrue(logger.warning.called)
        sys.modules.pop("nodes.llm_ros_node", None)

    def test_rejected_camera_tool_clarifies_without_vision_hallucination_retry(self):
        node_class = self._load_node_class()
        node = node_class.__new__(node_class)
        node.llm = MagicMock()
        node.llm.chat_stream.return_value = iter([
            {"type": "tool_call", "name": "inspect_camera", "arguments": '{"question":"你能看见吗？"}'},
            {"type": "done", "finish_reason": "tool_calls"},
        ])
        node.chat_history = deque(maxlen=40)
        node.punctuations = {'。', '？', '.', '?', '！', '!'}
        node.tts_publisher = MagicMock()
        node.action_publisher = MagicMock()
        node.corrected_publisher = MagicMock()
        node.full_ai_publisher = MagicMock()
        node.screen_dialog_publisher = MagicMock()
        node.busy_publisher = MagicMock()
        node.get_logger = lambda: MagicMock()

        node._process_voice_task("camera-question", "你能看见吗？")

        self.assertEqual(node.llm.chat_stream.call_count, 1)
        self.assertEqual(
            node.tts_publisher.publish.call_args_list[0].args[0].data,
            "你是想让我打开摄像头看一下吗？",
        )
        node.action_publisher.publish.assert_not_called()
        sys.modules.pop("nodes.llm_ros_node", None)

    def test_action_only_turn_uses_local_ack_without_second_llm_request(self):
        node_class = self._load_node_class()
        node = node_class.__new__(node_class)
        node.llm = MagicMock()
        node.llm.chat_stream.return_value = iter([
            {"type": "tool_call", "name": "play_sequence", "arguments": '{"sequence_name":"turn_head_right"}'},
            {"type": "done", "finish_reason": "tool_calls"},
        ])
        node.chat_history = deque(maxlen=40)
        node.punctuations = {'。', '？', '.', '?', '！', '!'}
        node.tts_publisher = MagicMock()
        node.action_publisher = MagicMock()
        node.corrected_publisher = MagicMock()
        node.full_ai_publisher = MagicMock()
        node.screen_dialog_publisher = MagicMock()
        node.busy_publisher = MagicMock()
        node.get_logger = lambda: MagicMock()

        node._process_voice_task("turn-action-only", "看看右边")

        self.assertEqual(node.llm.chat_stream.call_count, 1)
        self.assertEqual(
            node.tts_publisher.publish.call_args_list[0].args[0].data,
            "好的，我向右看。",
        )
        node.action_publisher.publish.assert_called_once()
        history = list(node.chat_history)
        self.assertEqual(
            [item["role"] for item in history],
            ["user", "assistant", "tool", "assistant"],
        )
        self.assertIsNone(history[1]["content"])
        self.assertEqual(history[2]["content"], '{"status": "accepted"}')
        self.assertEqual(history[3]["content"], "好的，我向右看。")
        sys.modules.pop("nodes.llm_ros_node", None)

    def test_request_history_is_limited_and_starts_with_user(self):
        node_class = self._load_node_class()
        node = node_class.__new__(node_class)
        history = [{"role": "assistant", "content": "orphan"}]
        for index in range(10):
            history.append({"role": "user", "content": f"u{index}"})
            history.append({"role": "assistant", "content": f"a{index}"})
        node.chat_history = deque(history, maxlen=40)

        selected = node._history_for_request()

        self.assertLessEqual(len(selected), node_class.CHAT_HISTORY_MESSAGES)
        self.assertEqual(selected[0]["role"], "user")
        self.assertEqual(selected[-1]["content"], "a9")
        sys.modules.pop("nodes.llm_ros_node", None)

    def test_request_history_keeps_text_but_drops_image_blocks(self):
        node_class = self._load_node_class()
        node = node_class.__new__(node_class)
        node.chat_history = deque([
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "上一轮看到了什么"},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/jpeg;base64,old-image",
                    }},
                ],
            },
            {"role": "assistant", "content": "一个杯子。"},
        ], maxlen=40)

        selected = node._history_for_request()

        self.assertEqual(selected[0]["content"], "上一轮看到了什么")
        self.assertNotIn("old-image", repr(selected))
        sys.modules.pop("nodes.llm_ros_node", None)

    def test_game_mode_ignores_microphone_text_but_accepts_game_vision(self):
        node_class = self._load_node_class()
        node = node_class.__new__(node_class)
        node._game_mode = "playing"
        node._request_queue = queue.Queue()
        node.get_logger = lambda: MagicMock()

        node.voice_callback(types.SimpleNamespace(data="这句不应进入大模型"))

        self.assertTrue(node._request_queue.empty())

        node.llm = MagicMock()
        node.llm.chat_stream.return_value = iter([
            {"type": "text", "content": "小心前面的敌人！"}
        ])
        node.busy_publisher = MagicMock()
        node.full_ai_publisher = MagicMock()
        node._publish_tts = MagicMock()
        node._finish_tts_turn = MagicMock()
        node._process_game_vision_task({
            "turn_id": "game-1",
            "jpeg": b"jpeg-data",
        })

        node._publish_tts.assert_called_once_with("小心前面的敌人！", "game-1")
        self.assertEqual(
            node.llm.chat_stream.call_args.kwargs["image_base64"],
            base64.b64encode(b"jpeg-data").decode("ascii"),
        )
        node._finish_tts_turn.assert_called_once_with("game-1")
        sys.modules.pop("nodes.llm_ros_node", None)


if __name__ == "__main__":
    unittest.main()
