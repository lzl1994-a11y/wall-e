import importlib
import json
import sys
import threading
import types
import unittest
from unittest.mock import MagicMock, patch

from services.game_protocol import encode_game_frame
from services.tft_preview_protocol import decode_preview_result
from services.tft_preview_server import PreviewResult


class _String:
    def __init__(self, data=""):
        self.data = data


class _UInt8MultiArray(_String):
    pass


class TftTcpServiceNodeTests(unittest.TestCase):
    @staticmethod
    def _load_node_class():
        fake_rclpy = types.ModuleType("rclpy")
        fake_rclpy_node = types.ModuleType("rclpy.node")
        fake_rclpy_node.Node = object
        fake_std_msgs = types.ModuleType("std_msgs")
        fake_std_msgs_msg = types.ModuleType("std_msgs.msg")
        fake_std_msgs_msg.String = _String
        fake_std_msgs_msg.UInt8MultiArray = _UInt8MultiArray
        sys.modules.pop("nodes.tft_tcp_service_node", None)
        with patch.dict(sys.modules, {
            "rclpy": fake_rclpy,
            "rclpy.node": fake_rclpy_node,
            "std_msgs": fake_std_msgs,
            "std_msgs.msg": fake_std_msgs_msg,
        }):
            module = importlib.import_module("nodes.tft_tcp_service_node")
        return module.TftTcpServiceNode

    def test_game_frames_are_forwarded_by_display_owner(self):
        node_class = self._load_node_class()
        node = node_class.__new__(node_class)
        node._game_mode = "playing"
        node._game_frame_adapter = MagicMock()
        raw = bytes([1, 2, 3, 4] * 4)

        node._on_game_frame(_UInt8MultiArray(data=encode_game_frame(raw, 2, 2, 8)))

        node._game_frame_adapter.submit_frame.assert_called_once_with(raw, 2, 2, 8)

    def test_camera_request_returns_correlated_last_frame(self):
        node_class = self._load_node_class()
        node = node_class.__new__(node_class)
        node._stop_event = threading.Event()
        node._preview_threads = set()
        node._preview_threads_lock = threading.Lock()
        node.tracking_preview = MagicMock()
        node.tracking_preview.pause.return_value = True
        node.camera_frames = object()
        node.server = MagicMock()
        node.server.send_camera_preview.return_value = PreviewResult(
            last_frame=b"\xff\xd8frame\xff\xd9"
        )
        node._result_publisher = MagicMock()
        node.get_logger = lambda: MagicMock()
        request = {
            "request_id": "preview-1",
            "duration_ms": 1500,
            "hold_ms": 3000,
            "fps": 10,
        }

        node._run_preview_request(request)

        message = node._result_publisher.publish.call_args.args[0]
        request_id, result = decode_preview_result(message.data)
        self.assertEqual(request_id, "preview-1")
        self.assertEqual(result.last_frame, b"\xff\xd8frame\xff\xd9")
        node.tracking_preview.resume.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
