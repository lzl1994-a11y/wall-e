import unittest

from services.camera_capture_protocol import (
    CAMERA_FRAME_TOPIC,
    CameraLeaseBook,
    build_hobot_camera_command,
    decode_camera_command,
    encode_camera_command,
    jpeg_from_ros_image,
)


class _JpegImage:
    encoding = "jpeg"
    data = b"jpeg-frame"


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
        self.assertEqual(jpeg_from_ros_image(_JpegImage()), b"jpeg-frame")


if __name__ == "__main__":
    unittest.main()
