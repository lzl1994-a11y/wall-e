import importlib
import json
import sys
import threading
import types
import unittest
from collections import deque
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
    def test_turn_complete_runs_after_preceding_audio_finishes(self):
        events = []
        player = PlaybackService.__new__(PlaybackService)
        player.sample_rate = 48000
        player._device = 2
        player.on_turn_complete = lambda: events.append("complete")
        samples = np.array([1, -2, 3], dtype=np.int16)

        with (
            patch.object(player, "_refresh_device", return_value=True),
            patch("services.playback_service.sd.play"),
            patch(
                "services.playback_service.sd.wait",
                side_effect=lambda: events.append("played"),
            ),
        ):
            player._play_item(samples)
            player._play_item(PlaybackService._TURN_END)

        self.assertEqual(events, ["played", "complete"])


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

        fake_std_msgs = types.ModuleType("std_msgs")
        fake_std_msgs_msg = types.ModuleType("std_msgs.msg")
        fake_std_msgs_msg.String = String

        fake_pypinyin = types.ModuleType("pypinyin")
        fake_pypinyin.Style = types.SimpleNamespace(NORMAL="normal")
        fake_pypinyin.pinyin = lambda text, style=None: [[text]]

        fake_llm_service = types.ModuleType("services.llm_service")
        fake_llm_service.LLMService = object
        fake_camera = types.ModuleType("services.camera_frame")
        fake_camera.CameraFrameProvider = object
        fake_camera.is_camera_inspection_request = lambda _text: False

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
        self.assertEqual(
            node.llm.chat_stream.call_args_list[1].kwargs["max_tokens_override"],
            512,
        )
        logger.warning.assert_called_once()
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


if __name__ == "__main__":
    unittest.main()
