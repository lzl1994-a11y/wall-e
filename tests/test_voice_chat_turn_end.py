import importlib
import sys
import threading
import types
import unittest
from unittest.mock import MagicMock, patch

from services.tts_protocol import decode_turn_end


class VoiceChatTurnEndTests(unittest.TestCase):
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
