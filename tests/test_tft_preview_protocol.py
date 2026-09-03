import base64
import json
import threading
import time
import unittest
from unittest.mock import patch

from services.tft_preview_protocol import (
    decode_preview_request,
    decode_preview_result,
    encode_camera_preview_request,
    encode_preview_result,
)
from services.tft_preview_server import PreviewResult
import services.tft_preview_client as client_module


class TftPreviewProtocolTests(unittest.TestCase):
    def test_camera_request_round_trip_clamps_limits(self):
        request_id, raw = encode_camera_preview_request(
            request_id="camera-1", duration_ms=1, hold_ms=-2, fps=90
        )

        self.assertEqual(request_id, "camera-1")
        self.assertEqual(decode_preview_request(raw), {
            "request_id": "camera-1",
            "kind": "camera_preview",
            "duration_ms": 100,
            "hold_ms": 0,
            "fps": 30,
        })

    def test_result_round_trip_keeps_original_camera_jpeg(self):
        source = PreviewResult(
            last_frame=b"\xff\xd8camera\xff\xd9",
            source_frames=12,
            sent_frames=7,
            connected=True,
        )

        request_id, result = decode_preview_result(
            encode_preview_result("camera-2", source)
        )

        self.assertEqual(request_id, "camera-2")
        self.assertEqual(result.last_frame, source.last_frame)
        self.assertEqual(result.source_frames, 12)
        self.assertEqual(result.sent_frames, 7)
        self.assertTrue(result.connected)

    def test_invalid_base64_result_is_rejected(self):
        raw = json.dumps({"request_id": "bad", "last_frame_base64": "%%%"})
        self.assertIsNone(decode_preview_result(raw))


class _String:
    def __init__(self, data=""):
        self.data = data


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _Node:
    def __init__(self):
        self.publisher = _Publisher()
        self.callback = None

    def create_publisher(self, *_args):
        return self.publisher

    def create_subscription(self, _type, _topic, callback, _depth):
        self.callback = callback
        return object()


class TftPreviewClientTests(unittest.TestCase):
    def test_waits_for_matching_correlated_result(self):
        node = _Node()
        with patch.object(client_module, "String", _String):
            client = client_module.TftPreviewClient(node)
            received = []

            def request_preview():
                received.append(client.send_camera_preview(
                    duration_ms=100, hold_ms=0, fps=10, timeout=1.0
                ))

            worker = threading.Thread(target=request_preview)
            worker.start()
            deadline = time.monotonic() + 1.0
            while not node.publisher.messages and time.monotonic() < deadline:
                time.sleep(0.01)
            request = decode_preview_request(node.publisher.messages[0].data)
            node.callback(_String(data=encode_preview_result(
                request["request_id"], PreviewResult(last_frame=b"jpeg")
            )))
            worker.join(timeout=1.0)

        self.assertEqual(received[0].last_frame, b"jpeg")


if __name__ == "__main__":
    unittest.main()
