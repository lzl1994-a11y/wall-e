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


class _Float32MultiArray(_String):
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
        fake_std_msgs_msg.Float32MultiArray = _Float32MultiArray
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
        node._music_state = "stopped"
        node._music_tracking_was_enabled = False
        node._music_stream = None
        node._music_frame_adapter = None
        node._surface_lock = threading.RLock()
        node._active_previews = 0
        node._game_mode = "robot"
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

    def test_music_spectrum_is_rendered_by_the_tft_owner(self):
        node_class = self._load_node_class()
        node = node_class.__new__(node_class)
        node._game_mode = "robot"
        node._music_state = "playing"
        node._music_track = "Artist+-+Track"
        node._music_frame_adapter = MagicMock()
        frame = (b"raw", 1, 1, 4)

        globals_ = node_class._on_music_spectrum.__globals__
        renderer = MagicMock(return_value=frame)
        with patch.dict(globals_, {"render_spectrum_frame": renderer}):
            node._on_music_spectrum(_Float32MultiArray(data=[0.1, 0.8]))

        node._music_frame_adapter.submit_frame.assert_called_once_with(*frame)
        renderer.assert_called_once_with(
            [0.1, 0.8], title="Artist+-+Track"
        )

    def test_game_handoff_preserves_tracking_restore_for_music(self):
        node_class = self._load_node_class()
        node = node_class.__new__(node_class)
        node._game_mode = "robot"
        node._music_state = "playing"
        node._music_track = "Track"
        node._music_stream = MagicMock()
        node._music_frame_adapter = MagicMock()
        node._music_tracking_was_enabled = True
        node._game_stream = None
        node._game_frame_adapter = None
        node._tracking_was_enabled = False
        node._surface_lock = threading.RLock()
        node._active_previews = 0
        node._stop_event = threading.Event()
        node.tracking_preview = MagicMock()
        node.server = MagicMock()
        node.server.open_jpeg_stream.side_effect = [MagicMock(), MagicMock()]
        node._game_request_publisher = MagicMock()
        node.get_logger = lambda: MagicMock()

        globals_ = node_class._ensure_game_stream.__globals__
        with patch.dict(globals_, {"GameFrameAdapter": MagicMock(side_effect=lambda *_a, **_k: MagicMock())}):
            node._on_game_state(_String(data=json.dumps({"mode": "playing"})))
            self.assertTrue(node._tracking_was_enabled)
            node._on_game_state(_String(data=json.dumps({"mode": "robot"})))

        self.assertTrue(node._music_tracking_was_enabled)
        node._close_music_stream()
        node.tracking_preview.resume.assert_called_once_with()

    def _music_node(self):
        node_class = self._load_node_class()
        node = node_class.__new__(node_class)
        node._game_mode = "robot"
        node._music_state = "playing"
        node._music_track = "Track"
        node._music_stream = None
        node._music_frame_adapter = None
        node._music_tracking_was_enabled = False
        node._surface_lock = threading.RLock()
        node._active_previews = 0
        node._stop_event = threading.Event()
        node.tracking_preview = MagicMock()
        node.tracking_preview.pause.return_value = False
        node.server = MagicMock()
        node.get_logger = lambda: MagicMock()
        return node

    def test_music_retries_when_screen_connects_after_play_request(self):
        node = self._music_node()
        stream = MagicMock(closed=False)
        node.server.open_jpeg_stream.side_effect = [None, stream]
        adapter = MagicMock()
        globals_ = node._ensure_music_stream.__func__.__globals__
        with patch.dict(globals_, {"GameFrameAdapter": MagicMock(return_value=adapter)}):
            node._maintain_music_stream()
            self.assertIsNone(node._music_frame_adapter)
            node._maintain_music_stream()

        self.assertIs(node._music_stream, stream)
        self.assertEqual(node.server.open_jpeg_stream.call_count, 2)
        adapter.submit_frame.assert_called_once()
        self.assertEqual(adapter.submit_frame.call_args.args[1:], (240, 240, 960))

    def test_music_reopens_disconnected_stream_and_preserves_tracking(self):
        node = self._music_node()
        old_stream = MagicMock(closed=True)
        old_adapter = MagicMock()
        node._music_stream = old_stream
        node._music_frame_adapter = old_adapter
        node._music_tracking_was_enabled = True
        new_stream = MagicMock(closed=False)
        node.server.open_jpeg_stream.return_value = new_stream
        globals_ = node._ensure_music_stream.__func__.__globals__
        with patch.dict(globals_, {"GameFrameAdapter": MagicMock()}):
            node._maintain_music_stream()

        old_adapter.close.assert_called_once()
        old_stream.close.assert_called_once()
        self.assertIs(node._music_stream, new_stream)
        self.assertTrue(node._music_tracking_was_enabled)
        node.tracking_preview.resume.assert_not_called()

    def test_music_does_not_steal_active_preview_or_game_surface(self):
        for preview_count, game_mode in ((1, "robot"), (0, "playing")):
            with self.subTest(previews=preview_count, game=game_mode):
                node = self._music_node()
                node._active_previews = preview_count
                node._game_mode = game_mode
                node._maintain_music_stream()
                node.server.open_jpeg_stream.assert_not_called()

    def test_music_started_during_camera_preview_is_shown_afterwards(self):
        node = self._music_node()
        node._music_state = "stopped"
        node.camera_frames = object()
        node._result_publisher = MagicMock()
        node._preview_threads = set()
        node._preview_threads_lock = threading.Lock()
        def camera_preview(*_args, **_kwargs):
            node._music_state = "playing"
            node._maintain_music_stream()
            node.server.open_jpeg_stream.assert_not_called()
            return PreviewResult(last_frame=b"frame")
        node.server.send_camera_preview.side_effect = camera_preview
        globals_ = node._ensure_music_stream.__func__.__globals__
        with patch.dict(globals_, {"GameFrameAdapter": MagicMock()}):
            node._run_preview_request({
                "request_id": "preview-music", "duration_ms": 1000,
                "hold_ms": 1000, "fps": 10,
            })

        self.assertEqual(node._active_previews, 0)
        node.server.open_jpeg_stream.assert_called_once()

    def test_overlapping_previews_restore_music_tracking_when_first_finishes_last(self):
        node = self._music_node()
        node._music_stream = MagicMock(closed=False)
        node._music_frame_adapter = MagicMock()
        node._music_tracking_was_enabled = True
        node._result_publisher = MagicMock()
        node._preview_threads = set()
        node._preview_threads_lock = threading.Lock()
        node._preview_tracking_was_enabled = False
        node.camera_frames = object()
        first_started = threading.Event()
        release_first = threading.Event()
        call_lock = threading.Lock()
        calls = 0

        def preview(*_args, **_kwargs):
            nonlocal calls
            with call_lock:
                calls += 1
                call_number = calls
            if call_number == 1:
                first_started.set()
                self.assertTrue(release_first.wait(2))
                return PreviewResult(last_frame=b"first")
            return PreviewResult(busy=True)

        node.server.send_camera_preview.side_effect = preview
        node.server.open_jpeg_stream.return_value = MagicMock(closed=False)
        request = {"duration_ms": 100, "hold_ms": 0, "fps": 10}
        first = threading.Thread(
            target=node._run_preview_request,
            args=({**request, "request_id": "first"},),
        )
        second = threading.Thread(
            target=node._run_preview_request,
            args=({**request, "request_id": "second"},),
        )
        globals_ = node._ensure_music_stream.__func__.__globals__
        with patch.dict(globals_, {"GameFrameAdapter": MagicMock()}):
            first.start()
            self.assertTrue(first_started.wait(2))
            second.start()
            second.join(2)
            self.assertFalse(second.is_alive())
            self.assertEqual(node._active_previews, 1)
            release_first.set()
            first.join(2)

        self.assertFalse(first.is_alive())
        self.assertEqual(node._active_previews, 0)
        self.assertTrue(node._music_tracking_was_enabled)
        node.tracking_preview.resume.assert_not_called()


if __name__ == "__main__":
    unittest.main()
