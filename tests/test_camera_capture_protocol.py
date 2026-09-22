import unittest

from services.vision.camera_capture_protocol import (
    CAMERA_FRAME_TOPIC,
    CAMERA_SOURCE_TOPIC,
    CameraLeaseBook,
    CameraWatchdogAction,
    build_hobot_camera_command,
    decode_camera_command,
    encode_camera_command,
    evaluate_camera_watchdog,
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

    def test_hobot_camera_keeps_the_canonical_image_topic(self):
        command = build_hobot_camera_command("/dev/video2", ros_setup=None)
        self.assertIn("video_device:=/dev/video2", command)
        self.assertIn("framerate:=15", command)
        self.assertEqual(CAMERA_SOURCE_TOPIC, "/image")
        self.assertNotIn("/image:=/camera_frame", command)

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

    def test_trusted_internal_jpeg_can_skip_full_decode(self):
        message = type("Image", (), {
            "encoding": "jpeg",
            "data": b"\xff\xd8structurally-complete\xff\xd9",
        })()
        self.assertEqual(
            jpeg_from_ros_image(message, validate_decode=False),
            message.data,
        )

    def test_watchdog_first_frame_timeout_boundaries(self):
        started = 100.0
        # 1. Before timeout (waited 44.9s < 45.0s)
        decision_before = evaluate_camera_watchdog(
            process_alive=True,
            now=144.9,
            process_started_at=started,
            last_source_frame=0.0,
            has_active_leases=False,
        )
        self.assertEqual(decision_before.action, CameraWatchdogAction.PUBLISH_STATUS)
        self.assertEqual(decision_before.state, "starting")

        # 2. At boundary (waited 45.0s == 45.0s) -> strictly > 45.0 required, so no timeout
        decision_equal = evaluate_camera_watchdog(
            process_alive=True,
            now=145.0,
            process_started_at=started,
            last_source_frame=0.0,
            has_active_leases=False,
        )
        self.assertEqual(decision_equal.action, CameraWatchdogAction.PUBLISH_STATUS)
        self.assertEqual(decision_equal.state, "starting")

        # 3. Exceeded timeout (waited 45.01s > 45.0s)
        decision_after = evaluate_camera_watchdog(
            process_alive=True,
            now=145.01,
            process_started_at=started,
            last_source_frame=0.0,
            has_active_leases=False,
        )
        self.assertEqual(decision_after.action, CameraWatchdogAction.FIRST_FRAME_TIMEOUT)
        self.assertAlmostEqual(decision_after.elapsed_sec, 45.01)

    def test_watchdog_frame_timeout_boundaries(self):
        started = 100.0
        last_frame = 105.0
        # 1. Before timeout (stalled 2.9s < 3.0s)
        decision_before = evaluate_camera_watchdog(
            process_alive=True,
            now=107.9,
            process_started_at=started,
            last_source_frame=last_frame,
            has_active_leases=False,
        )
        self.assertEqual(decision_before.action, CameraWatchdogAction.PUBLISH_STATUS)
        self.assertEqual(decision_before.state, "standby")

        # 2. At boundary (stalled 3.0s == 3.0s) -> strictly > 3.0 required, so no timeout
        decision_equal = evaluate_camera_watchdog(
            process_alive=True,
            now=108.0,
            process_started_at=started,
            last_source_frame=last_frame,
            has_active_leases=False,
        )
        self.assertEqual(decision_equal.action, CameraWatchdogAction.PUBLISH_STATUS)
        self.assertEqual(decision_equal.state, "standby")

        # 3. Exceeded timeout (stalled 3.01s > 3.0s)
        decision_after = evaluate_camera_watchdog(
            process_alive=True,
            now=108.01,
            process_started_at=started,
            last_source_frame=last_frame,
            has_active_leases=False,
        )
        self.assertEqual(decision_after.action, CameraWatchdogAction.FRAME_TIMEOUT)
        self.assertAlmostEqual(decision_after.elapsed_sec, 3.01)

    def test_watchdog_process_restart_boundaries(self):
        retry_after = 200.0
        # 1. Before retry delay (now 199.9 < 200.0) -> no action
        decision_before = evaluate_camera_watchdog(
            process_alive=False,
            now=199.9,
            retry_after=retry_after,
            has_active_leases=False,
        )
        self.assertEqual(decision_before.action, CameraWatchdogAction.NONE)

        # 2. At boundary (now 200.0 == retry_after) -> start process even without leases
        decision_equal = evaluate_camera_watchdog(
            process_alive=False,
            now=200.0,
            retry_after=retry_after,
            has_active_leases=False,
        )
        self.assertEqual(decision_equal.action, CameraWatchdogAction.START_PROCESS)

        # 3. After retry delay (now 201.0 > 200.0) -> start process
        decision_after = evaluate_camera_watchdog(
            process_alive=False,
            now=201.0,
            retry_after=retry_after,
            has_active_leases=True,
        )
        self.assertEqual(decision_after.action, CameraWatchdogAction.START_PROCESS)

    def test_watchdog_normal_state_selection_with_active_leases(self):
        started = 100.0
        last_source = 105.0
        # With active lease and recent output (<= 1.0s) -> streaming
        decision_streaming = evaluate_camera_watchdog(
            process_alive=True,
            now=106.0,
            process_started_at=started,
            last_source_frame=last_source,
            last_output_frame=105.0,
            has_active_leases=True,
        )
        self.assertEqual(decision_streaming.action, CameraWatchdogAction.PUBLISH_STATUS)
        self.assertEqual(decision_streaming.state, "streaming")

        # With active lease but stale output (> 1.0s) -> starting
        decision_starting = evaluate_camera_watchdog(
            process_alive=True,
            now=106.01,
            process_started_at=started,
            last_source_frame=last_source,
            last_output_frame=105.0,
            has_active_leases=True,
        )
        self.assertEqual(decision_starting.action, CameraWatchdogAction.PUBLISH_STATUS)
        self.assertEqual(decision_starting.state, "starting")

    def test_watchdog_normal_state_selection_without_active_leases(self):
        started = 100.0
        # No lease and no frame yet from current process -> starting
        decision_starting = evaluate_camera_watchdog(
            process_alive=True,
            now=110.0,
            process_started_at=started,
            last_source_frame=90.0,
            has_active_leases=False,
        )
        self.assertEqual(decision_starting.action, CameraWatchdogAction.PUBLISH_STATUS)
        self.assertEqual(decision_starting.state, "starting")

        # No lease and frame received -> standby
        decision_standby = evaluate_camera_watchdog(
            process_alive=True,
            now=102.0,
            process_started_at=started,
            last_source_frame=101.0,
            has_active_leases=False,
        )
        self.assertEqual(decision_standby.action, CameraWatchdogAction.PUBLISH_STATUS)
        self.assertEqual(decision_standby.state, "standby")

    def test_watchdog_custom_thresholds(self):
        d1 = evaluate_camera_watchdog(
            process_alive=True,
            now=110.5,
            process_started_at=100.0,
            last_source_frame=0.0,
            first_frame_timeout_sec=10.0,
        )
        self.assertEqual(d1.action, CameraWatchdogAction.FIRST_FRAME_TIMEOUT)

        d2 = evaluate_camera_watchdog(
            process_alive=True,
            now=102.5,
            process_started_at=100.0,
            last_source_frame=100.1,
            frame_timeout_sec=2.0,
        )
        self.assertEqual(d2.action, CameraWatchdogAction.FRAME_TIMEOUT)

        d3 = evaluate_camera_watchdog(
            process_alive=True,
            now=100.8,
            process_started_at=100.0,
            last_source_frame=100.7,
            last_output_frame=100.2,
            has_active_leases=True,
            output_active_window_sec=0.5,
        )
        self.assertEqual(d3.state, "starting")


if __name__ == "__main__":
    unittest.main()
