import unittest

from services.camera_capture_protocol import (
    CAMERA_FRAME_TOPIC,
    CameraLeaseBook,
    build_hobot_camera_command,
    decode_camera_command,
    encode_camera_command,
    jpeg_from_ros_image,
)


def _valid_jpeg():
    try:
        import cv2
        import numpy as np
    except ImportError:
        # Structural fallback for environments where the optional decoder is
        # unavailable. Production RDK images are additionally decoded by cv2.
        return b"\xff\xd8jpeg-frame\xff\xd9"
    ok, encoded = cv2.imencode(".jpg", np.zeros((2, 2, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


class _JpegImage:
    encoding = "jpeg"
    data = _valid_jpeg()


class CameraCaptureProtocolTests(unittest.TestCase):
    def test_command_round_trip_and_lease_clamping(self):
        command = decode_camera_command(encode_camera_command("acquire", "client-1", 99))
        self.assertEqual(command["action"], "acquire")
        self.assertEqual(command["client_id"], "client-1")
        self.assertEqual(command["lease_sec"], 30.0)
        self.assertIsNone(decode_camera_command("{}"))

    def test_lease_book_expires_and_releases_clients(self):
        leases = CameraLeaseBook()
        leases.acquire("llm", 2.0, now=10.0)
        leases.acquire("web", 5.0, now=10.0)
        leases.purge(now=12.1)
        self.assertTrue(leases.active)
        self.assertEqual(leases.count, 1)
        leases.release("web")
        self.assertFalse(leases.active)

    def test_hobot_camera_is_remapped_to_dedicated_topic(self):
        command = build_hobot_camera_command("/dev/video2", ros_setup=None)
        self.assertIn("video_device:=/dev/video2", command)
        self.assertIn(f"/image:={CAMERA_FRAME_TOPIC}", command)

    def test_jpeg_ros_image_is_forwarded_without_opening_a_device(self):
        self.assertEqual(jpeg_from_ros_image(_JpegImage()), _JpegImage.data)

    def test_incomplete_or_mislabeled_jpeg_is_rejected(self):
        incomplete = type("Image", (), {
            "encoding": "jpeg",
            "data": b"\xff\xd8truncated",
        })()
        mislabeled = type("Image", (), {
            "encoding": "jpeg",
            "data": b"not-a-jpeg",
        })()

        self.assertIsNone(jpeg_from_ros_image(incomplete))
        self.assertIsNone(jpeg_from_ros_image(mislabeled))

    def test_bytes_after_jpeg_end_marker_are_removed(self):
        message = type("Image", (), {
            "encoding": "jpeg",
            "data": _JpegImage.data + b"trailing-camera-bytes",
        })()

        self.assertEqual(jpeg_from_ros_image(message), _JpegImage.data)


if __name__ == "__main__":
    unittest.main()
