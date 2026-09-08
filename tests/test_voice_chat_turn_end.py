import importlib
import json
import sys
import threading
import types
import unittest
from unittest.mock import MagicMock, patch

from services.tts_protocol import decode_turn_end
from services.wake_audio_protocol import decode_wake_audio, encode_wake_audio


class VoiceChatTurnEndTests(unittest.TestCase):
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

        fake_service = types.ModuleType("services.voice_chat_service")
        fake_service.VoiceChatService = object
        fake_audio_output = types.ModuleType("services.audio_output")
        fake_audio_output.OUTPUT_CHANNELS = 1
        fake_audio_output.OUTPUT_SAMPLE_RATE = 48000
        fake_audio_output.OUTPUT_SAMPLE_WIDTH = 2
        fake_tools = types.ModuleType("services.tool_dispatcher")
        fake_tools.build_action_cmd = lambda name, arguments: ""
        fake_usb = types.ModuleType("services.usb_devices")
        fake_usb.resolve_audio_device = lambda *args, **kwargs: None

        modules = {
            "rclpy": fake_rclpy,
            "rclpy.node": fake_rclpy_node,
            "std_msgs": fake_std_msgs,
            "std_msgs.msg": fake_std_msgs_msg,
            "services.voice_chat_service": fake_service,
            "services.audio_output": fake_audio_output,
            "services.tool_dispatcher": fake_tools,
            "services.usb_devices": fake_usb,
        }
        sys.modules.pop("nodes.voice_chat_ros_node", None)
        with patch.dict(sys.modules, modules):
            module = importlib.import_module("nodes.voice_chat_ros_node")
        return module.VoiceChatNode

    def _make_wake_node(self):
        node_class = self._load_node_class()
        node = node_class.__new__(node_class)
        node._output_state_lock = threading.Lock()
        node._wake_play_lock = threading.Lock()
        node._awaiting_tts_playback = False
        node._wake_response_active = True
        node._wake_request_id = "wake-current"
        node._wake_watchdog = None
        node._shutting_down = False
        node._resume_timer = None
        node._wake_wav = "wake_response.wav"
        node._schedule_capture_resume = MagicMock()
        node._schedule_wake_watchdog = MagicMock()
        node.wake_audio_pub = MagicMock()
        node.wake_audio_pub.get_subscription_count.return_value = 1
        node.dialog_pub = MagicMock()
        node.vc = MagicMock()
        node.get_logger = lambda: MagicMock()
        self.addCleanup(sys.modules.pop, "nodes.voice_chat_ros_node", None)
        return node

    def test_wake_pcm_uses_shared_speaker_and_waits_for_matching_completion(self):
        node = self._make_wake_node()
        audio = MagicMock()
        audio.set_frame_rate.return_value = audio
        audio.set_channels.return_value = audio
        audio.set_sample_width.return_value = audio
        audio.__len__.return_value = 1000
        audio.raw_data = b"\x01\x00\xfe\xff"
        fake_pydub = types.ModuleType("pydub")
        fake_pydub.AudioSegment = MagicMock()
        fake_pydub.AudioSegment.from_wav.return_value = audio
        with (
            patch.dict(sys.modules, {"pydub": fake_pydub}),
            patch("os.path.exists", return_value=True),
        ):
            node._play_wake_response("wake-current")

        request = node.wake_audio_pub.publish.call_args.args[0].data
        self.assertEqual(decode_wake_audio(request), ("wake-current", audio.raw_data))
        audio.set_frame_rate.assert_called_once_with(48000)
        audio.set_channels.assert_called_once_with(1)
        audio.set_sample_width.assert_called_once_with(2)
        self.assertTrue(node._wake_response_active)
        node._schedule_capture_resume.assert_not_called()

        node._on_wake_audio_done(types.SimpleNamespace(data="wake-current"))

        self.assertFalse(node._wake_response_active)
        node._schedule_capture_resume.assert_called_once_with()

    def test_wake_start_preserves_pending_tts_completion(self):
        node = self._make_wake_node()
        node._wake_response_active = False
        node._wake_request_id = None
        node._awaiting_tts_playback = True
        with patch("threading.Thread"):
            node._on_wake_word()

        self.assertTrue(node._awaiting_tts_playback)
        self.assertTrue(node._wake_response_active)
        self.assertIsNotNone(node._wake_request_id)
        node.vc.begin_output_playback.assert_called_once_with()

    def test_wake_completion_cannot_finish_pending_tts(self):
        node = self._make_wake_node()
        node._awaiting_tts_playback = True

        node._on_wake_audio_done(types.SimpleNamespace(data="wake-current"))

        self.assertFalse(node._wake_response_active)
        self.assertTrue(node._awaiting_tts_playback)
        node._schedule_capture_resume.assert_not_called()
        node._on_playback_state(types.SimpleNamespace(data="idle"))
        node._schedule_capture_resume.assert_called_once_with()

    def test_tts_completion_cannot_finish_pending_wake(self):
        node = self._make_wake_node()
        node._awaiting_tts_playback = True

        node._on_playback_state(types.SimpleNamespace(data="idle"))

        self.assertFalse(node._awaiting_tts_playback)
        self.assertTrue(node._wake_response_active)
        node._schedule_capture_resume.assert_not_called()
        node._on_wake_audio_done(types.SimpleNamespace(data="wake-current"))
        node._schedule_capture_resume.assert_called_once_with()

    def test_stale_wake_completion_does_not_reopen_capture(self):
        node = self._make_wake_node()

        node._on_wake_audio_done(types.SimpleNamespace(data="wake-previous"))

        self.assertTrue(node._wake_response_active)
        self.assertEqual(node._wake_request_id, "wake-current")
        node._schedule_capture_resume.assert_not_called()

    def test_missing_playback_node_skips_wake_and_releases_capture(self):
        node = self._make_wake_node()
        node.wake_audio_pub.get_subscription_count.return_value = 0

        node._play_wake_response("wake-current")

        node.wake_audio_pub.publish.assert_not_called()
        self.assertFalse(node._wake_response_active)
        node._schedule_capture_resume.assert_called_once_with()

    def test_failed_wake_decode_releases_capture(self):
        node = self._make_wake_node()
        with patch("os.path.exists", return_value=False):
            node._play_wake_response("wake-current")

        node.wake_audio_pub.publish.assert_not_called()
        self.assertFalse(node._wake_response_active)
        node._schedule_capture_resume.assert_called_once_with()

    def test_connected_speaker_watchdog_waits_for_actual_completion(self):
        node = self._make_wake_node()

        node._check_wake_playback_connection("wake-current")

        self.assertTrue(node._wake_response_active)
        node._schedule_capture_resume.assert_not_called()
        node._schedule_wake_watchdog.assert_called_once_with("wake-current")

    def test_disappeared_speaker_releases_wake_wait(self):
        node = self._make_wake_node()
        node.wake_audio_pub.get_subscription_count.return_value = 0

        node._check_wake_playback_connection("wake-current")

        self.assertFalse(node._wake_response_active)
        node._schedule_capture_resume.assert_called_once_with()

    def test_wake_protocol_rejects_invalid_pcm_and_missing_correlation(self):
        for message in (
            "null", "[]", "{}", '{"request_id": "x", "audio_base64": "!"}',
            '{"request_id": "x", "audio_base64": "AA=="}',
            '{"request_id": "", "audio_base64": "AAA="}',
        ):
            with self.subTest(message=message):
                with self.assertRaises(ValueError):
                    decode_wake_audio(message)
        with self.assertRaises(ValueError):
            encode_wake_audio("wake", b"\x01")

    def test_completed_actions_are_forwarded_with_the_screen_dialog(self):
        node_class = self._load_node_class()
        node = node_class.__new__(node_class)
        node._active_turn_id = "turn-music"
        node._sentence_buffer = ""
        node._punc_count = 0
        node._correction_done = True
        node.punctuations = {"。"}
        node.tts_pub = MagicMock()
        node.dialog_pub = MagicMock()
        node.get_logger = lambda: MagicMock()
        node.vc = types.SimpleNamespace(last_action_results=[{
            "name": "control_music",
            "arguments": {"action": "play", "track": ""},
            "status": "completed",
        }])

        node._on_llm_reply("开始播放音乐。")

        dialog = json.loads(node.dialog_pub.publish.call_args.args[0].data)
        self.assertEqual(dialog["actions"], node.vc.last_action_results)

    def test_llm_done_publishes_turn_end_and_resets_turn_state(self):
        node_class = self._load_node_class()
        node = node_class.__new__(node_class)
        node.tts_pub = MagicMock()
        node.get_logger = lambda: MagicMock()
        node._active_turn_id = "turn-multimodal"
        node._sentence_buffer = ""
        node._punc_count = 0
        node._correction_done = True
        node._output_state_lock = threading.Lock()
        node._awaiting_tts_playback = False

        node._on_llm_done()

        marker = node.tts_pub.publish.call_args.args[0].data
        self.assertEqual(decode_turn_end(marker), "turn-multimodal")
        self.assertIsNone(node._active_turn_id)
        self.assertEqual(node._sentence_buffer, "")
        self.assertEqual(node._punc_count, 0)
        self.assertFalse(node._correction_done)
        self.assertTrue(node._awaiting_tts_playback)
        sys.modules.pop("nodes.voice_chat_ros_node", None)

    def test_playback_idle_schedules_capture_resume(self):
        node_class = self._load_node_class()
        node = node_class.__new__(node_class)
        node._output_state_lock = threading.Lock()
        node._awaiting_tts_playback = True
        node._wake_response_active = False
        node._schedule_capture_resume = MagicMock()

        node._on_playback_state(types.SimpleNamespace(data="idle"))

        self.assertFalse(node._awaiting_tts_playback)
        node._schedule_capture_resume.assert_called_once_with()
        sys.modules.pop("nodes.voice_chat_ros_node", None)

    def test_semantic_camera_tool_captures_then_calls_visual_model(self):
        node_class = self._load_node_class()
        node = node_class.__new__(node_class)
        node.tts_pub = MagicMock()
        node.camera_frames = MagicMock()
        node.tft_preview_settings = types.SimpleNamespace(
            recognition_duration_ms=1500,
            hold_ms=3000,
            fps=10,
        )
        node.tft_preview = MagicMock()
        node.tft_preview.send_camera_preview.return_value = types.SimpleNamespace(
            busy=False,
            last_frame=b"\xff\xd8vision\xff\xd9",
            error="",
        )
        node.vc = MagicMock()
        node.vc.analyze_image.return_value = "前面有一只杯子。"
        node.get_logger = lambda: MagicMock()

        answer = node._on_tool_call(
            "inspect_camera", {"question": "前面有什么"}
        )

        self.assertEqual(answer, "前面有一只杯子。")
        self.assertEqual(node.tts_pub.publish.call_args.args[0].data, "好的，我看一下。")
        node.vc.analyze_image.assert_called_once_with(
            "前面有什么", "/9h2aXNpb27/2Q=="
        )
        sys.modules.pop("nodes.voice_chat_ros_node", None)

    def test_photo_request_captures_and_saves_last_frame(self):
        node_class = self._load_node_class()
        node = node_class.__new__(node_class)
        node.tts_pub = MagicMock()
        node.camera_frames = MagicMock()
        node.tft_preview_settings = types.SimpleNamespace(
            photo_duration_ms=3000,
            hold_ms=3000,
            fps=10,
            photo_directory="/tmp/wali-photos",
        )
        node.tft_preview = MagicMock()
        node.tft_preview.send_camera_preview.return_value = types.SimpleNamespace(
            busy=False,
            last_frame=b"\xff\xd8photo\xff\xd9",
            error="",
        )
        node.get_logger = lambda: MagicMock()
        save = MagicMock(return_value="/tmp/wali-photos/photo.jpg")

        with patch.dict(
            node_class._process_camera_photo.__globals__,
            {"save_camera_photo": save},
        ):
            answer = node._process_camera_photo()

        self.assertEqual(answer, "拍好了，照片已经保存。")
        save.assert_called_once_with(b"\xff\xd8photo\xff\xd9", "/tmp/wali-photos")
        sys.modules.pop("nodes.voice_chat_ros_node", None)


if __name__ == "__main__":
    unittest.main()
