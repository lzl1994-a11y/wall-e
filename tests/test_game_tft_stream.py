import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from services.game_tft_stream import GameTftStreamServer, prepare_game_bgr, prepare_game_jpeg
from services.tft_preview_server import (
    JPEG_FRAME,
    STREAM_END,
    STREAM_START,
    STREAM_START_MESSAGE,
)


def _quadrant_jpeg():
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[:20, :30] = (255, 0, 0)
    image[20:, 30:] = (0, 0, 255)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


class GameTftStreamTests(unittest.TestCase):
    def test_game_frame_is_not_camera_rotated(self):
        result = prepare_game_jpeg(_quadrant_jpeg(), quality=100)
        image = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR)
        self.assertGreater(int(image[5, 5, 0]), int(image[5, 5, 2]))
        self.assertGreater(int(image[-5, -5, 2]), int(image[-5, -5, 0]))

    def test_raw_bgr_path_preserves_aspect_ratio(self):
        source = cv2.imdecode(
            np.frombuffer(_quadrant_jpeg(), np.uint8), cv2.IMREAD_COLOR
        )
        result = prepare_game_bgr(source, quality=100)
        image = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape[:2], (160, 240))

    def test_one_start_many_frames_one_end(self):
        server = GameTftStreamServer()
        server._verified_client = Mock(return_value=object())
        server._send_packet = Mock()
        stream = server.open_jpeg_stream(fps=15)
        self.assertIsNotNone(stream)
        stream.send_jpeg(_quadrant_jpeg())
        stream.send_jpeg(_quadrant_jpeg())
        stream.close()
        message_types = [call.args[1] for call in server._send_packet.call_args_list]
        self.assertEqual(
            message_types,
            [STREAM_START_MESSAGE, JPEG_FRAME, JPEG_FRAME, STREAM_END],
        )
        start_payload = server._send_packet.call_args_list[0].args[3]
        self.assertEqual(
            STREAM_START.unpack(start_payload), (0xFFFFFFFF, 0, 15, 0)
        )


if __name__ == "__main__":
    unittest.main()
