import importlib
import sys
import time
import types
import unittest
from unittest.mock import patch

from services.vision.camera_capture_protocol import encode_camera_command


class _FakeString:
    def __init__(self, data=""):
        self.data = data


class _FakeImage:
    encoding = "jpeg"
    data = b"frame"
    width = 640
    height = 480
    header = object()


class _FakeCompressedImage:
    def __init__(self, *, header=None, format="", data=b""):
        self.header = header
        self.format = format
        self.data = data


class _FakePublisher:
    def __init__(self, topic):
        self.topic = topic
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _FakeLogger:
    def info(self, _message):
        pass

    def warn(self, _message):
        pass


class _FakeSetBool:
    class Request:
        def __init__(self):
            self.data = False


class _FakeServiceResponse:
    def __init__(self, enabled):
        # Match the deployed hobot_usb_cam bug: the operation completes but
        # SetBool.success remains at its default false value.
        self.success = False
        self.message = "Start Capturing" if enabled else "Stop Capturing"


class _FakeFuture:
    def __init__(self, response):
        self._response = response

    def result(self):
        return self._response

    def add_done_callback(self, callback):
        callback(self)


class _FakeClient:
    def __init__(self):
        self.requests = []
        self.ready = True
        self.removed_requests = []

    def remove_pending_request(self, future):
        self.removed_requests.append(future)

    def service_is_ready(self):
        return self.ready

    def call_async(self, request):
        self.requests.append(request.data)
        return _FakeFuture(_FakeServiceResponse(request.data))


class _FakeNodeBase:
    def __init__(self, _name):
        self.publishers = {}
        self.publisher_types = {}
        self.publisher_qos = {}
        self.subscriptions = {}
        self.subscription_types = {}
        self.clients = {}

    def create_publisher(self, message_type, topic, qos):
        publisher = _FakePublisher(topic)
        self.publishers[topic] = publisher
        self.publisher_types[topic] = message_type
        self.publisher_qos[topic] = qos
        return publisher

    def create_subscription(self, message_type, topic, callback, _qos):
        self.subscriptions.setdefault(topic, []).append(callback)
        self.subscription_types.setdefault(topic, []).append(message_type)
        return object()

    def create_client(self, _service_type, service_name):
        client = _FakeClient()
        self.clients[service_name] = client
        return client

    def create_timer(self, _period, _callback):
        return object()

    def count_publishers(self, _topic):
        return 0

    def get_topic_names_and_types(self):
        return {}

    def get_logger(self):
        return _FakeLogger()

    def destroy_node(self):
        pass


class _FakeProcess:
    def __init__(self):
        self.pid = 123
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.terminate()

    def wait(self, timeout=None):
        del timeout
        return self.returncode


def _load_camera_capture_module():
    fake_rclpy = types.ModuleType("rclpy")
    fake_rclpy.init = lambda: None
    fake_rclpy.spin = lambda _node: None
    fake_rclpy.shutdown = lambda: None
    fake_node_module = types.ModuleType("rclpy.node")
    fake_node_module.Node = _FakeNodeBase
    fake_qos_module = types.ModuleType("rclpy.qos")
    fake_qos_module.qos_profile_sensor_data = object()
    fake_sensor_module = types.ModuleType("sensor_msgs.msg")
    fake_sensor_module.Image = _FakeImage
    fake_sensor_module.CompressedImage = _FakeCompressedImage
    fake_std_module = types.ModuleType("std_msgs.msg")
    fake_std_module.String = _FakeString
    fake_std_srvs_module = types.ModuleType("std_srvs.srv")
    fake_std_srvs_module.SetBool = _FakeSetBool
    modules = {
        "rclpy": fake_rclpy,
        "rclpy.node": fake_node_module,
        "rclpy.qos": fake_qos_module,
        "sensor_msgs.msg": fake_sensor_module,
        "std_msgs.msg": fake_std_module,
        "std_srvs.srv": fake_std_srvs_module,
    }
    sys.modules.pop("nodes.camera_capture_node", None)
    with patch.dict(sys.modules, modules):
        return importlib.import_module("nodes.camera_capture_node")


class CameraCaptureNodeTests(unittest.TestCase):
    def test_status_ack_identifies_current_request_and_requested_resolution(self):
        import json
        module = _load_camera_capture_module()
        with (
            patch.object(module, "resolve_camera_device", return_value="/dev/video8"),
            patch.object(module.subprocess, "Popen", side_effect=[_FakeProcess(), _FakeProcess()]),
        ):
            node = module.CameraCaptureNode()
            node._on_command(_FakeString(encode_camera_command("acquire", "llm-test", 10,
                                                              resolution=(1280, 720))))
            payload = json.loads(node._status_pub.messages[-1].data)
            self.assertEqual(payload["request"], {"client_id": "llm-test", "width": 1280, "height": 720})
            self.assertTrue(payload["reconfiguring"])
            self.assertEqual((payload["width"], payload["height"]), node._camera_size)
            node._on_command(_FakeString(encode_camera_command("renew", "llm-test", 10,
                                                              resolution=(1280, 720))))
            self.assertEqual(json.loads(node._status_pub.messages[-1].data)["request"], payload["request"])

    def test_lost_capture_response_is_replaced_with_latest_lease_request(self):
        module = _load_camera_capture_module()
        from unittest.mock import Mock
        process = _FakeProcess()
        with (
            patch.object(module, "resolve_camera_device", return_value="/dev/video8"),
            patch.object(module.subprocess, "Popen", return_value=process) as popen,
        ):
            node = module.CameraCaptureNode()
            client = node._capture_client
            lost = Mock()
            with patch.object(client, "call_async", return_value=lost):
                node._tick()  # Stop request to the driver that has no reply.
            self.assertFalse(node._capture_request_target)
            node._leases.acquire("webrtc-h264:test", 60.0)
            node._tick()
            self.assertIs(node._capture_request, lost)
            self.assertEqual(client.requests, [])
            node._capture_request_started_at = time.monotonic() - 6.0
            node._tick()
            self.assertEqual(client.removed_requests, [lost])
            lost.cancel.assert_called_once()
            self.assertEqual(client.requests, [True])
            self.assertTrue(node._capture_enabled)
            self.assertIsNone(node._capture_request)
            # A late old stop response cannot undo the new start response.
            node._on_capture_response(lost)
            self.assertTrue(node._capture_enabled)
            self.assertEqual(popen.call_count, 1)

    def test_driver_restart_removes_pending_capture_future(self):
        module = _load_camera_capture_module()
        from unittest.mock import Mock
        with (
            patch.object(module, "resolve_camera_device", return_value="/dev/video8"),
            patch.object(module.subprocess, "Popen", side_effect=[_FakeProcess(), _FakeProcess()]),
        ):
            node = module.CameraCaptureNode()
            lost = Mock()
            with patch.object(node._capture_client, "call_async", return_value=lost):
                node._tick()
            node._leases.acquire("webrtc-h264:test", 60.0, resolution=(1280, 720))
            node._tick()
            self.assertEqual(node._capture_client.removed_requests, [lost])
            lost.cancel.assert_called_once()
            self.assertIsNone(node._capture_request)
            self.assertEqual(node._camera_size, (1280, 720))

    def test_capture_service_recovery_has_a_total_deadline(self):
        module = _load_camera_capture_module()
        from unittest.mock import Mock
        process = _FakeProcess()
        with (
            patch.object(module, "resolve_camera_device", return_value="/dev/video8"),
            patch.object(module.subprocess, "Popen", return_value=process),
        ):
            node = module.CameraCaptureNode()
            lost = Mock()
            with patch.object(node._capture_client, "call_async", return_value=lost):
                node._tick()
            node._capture_request_started_at -= node.CAPTURE_REQUEST_TIMEOUT_SEC + 1.0
            node._capture_transition_started_at -= node.FIRST_FRAME_TIMEOUT_SEC + 1.0
            node._tick()
            self.assertTrue(process.terminated)
            self.assertIsNone(node._camera_process)
            self.assertIsNone(node._capture_request)
            self.assertIn('"state":"error"', node._status_pub.messages[-1].data)
            self.assertIn("持续无响应", node._status_pub.messages[-1].data)

    def test_photo_resolution_changes_only_the_single_camera_owner_and_restores(self):
        module = _load_camera_capture_module()
        processes = [_FakeProcess(), _FakeProcess(), _FakeProcess()]
        with (
            patch.object(module, "resolve_camera_device", return_value="/dev/video8"),
            patch.object(module.subprocess, "Popen", side_effect=processes) as popen,
        ):
            node = module.CameraCaptureNode()
            node._on_command(_FakeString(encode_camera_command("acquire", "photo", 5, resolution=(1280, 720))))
            self.assertTrue(processes[0].terminated)
            self.assertIs(node._camera_process, processes[1])
            self.assertIn("image_width:=1280", popen.call_args.args[0])
            self.assertEqual(node._camera_size, (1280, 720))
            node._on_command(_FakeString(encode_camera_command("renew", "photo", 5, resolution=(1280, 720))))
            self.assertEqual(popen.call_count, 2)
            node._on_command(_FakeString(encode_camera_command("release", "photo")))
            self.assertTrue(processes[1].terminated)
            self.assertEqual(node._camera_size, (640, 480))
            self.assertEqual(popen.call_count, 3)

    def test_detection_frames_require_tracking_lease_without_stopping_preview(self):
        module = _load_camera_capture_module()
        with patch.object(module, "jpeg_from_ros_image", return_value=b"frame"):
            node = module.CameraCaptureNode()
            source = _FakeCompressedImage(header=object(), format="jpeg", data=b"frame")
            for client in ("tft-photo", "webrtc:WALLY-01", "webrtc-h264:WALLY-01"):
                node._on_command(_FakeString(encode_camera_command("acquire", client, 5)))
            node._on_source_image(source)
            self.assertEqual(node.publishers["/tracking_camera_frame"].messages, [])
            self.assertEqual(len(node.publishers["/camera_frame"].messages), 1)

            node._on_command(_FakeString(encode_camera_command("acquire", "tracking-vision", 5)))
            node._on_source_image(source)
            detection_frames = node.publishers["/tracking_camera_frame"].messages
            self.assertEqual(len(detection_frames), 1)
            self.assertIs(detection_frames[0], node.publishers["/camera_frame"].messages[-1])
            node._on_command(_FakeString(encode_camera_command("release", "tracking-vision")))
            node._on_source_image(source)
            self.assertEqual(len(detection_frames), 1)
            self.assertEqual(len(node.publishers["/camera_frame"].messages), 3)

            node._on_command(_FakeString(encode_camera_command("acquire", "tracking-vision", 5)))
            node._on_source_image(source)
            self.assertEqual(len(detection_frames), 2)
            node._leases.acquire("tracking-vision", 1, now=time.monotonic() - 2)
            node._on_source_image(source)
            self.assertEqual(len(detection_frames), 2)
            self.assertEqual(len(node.publishers["/camera_frame"].messages), 5)
            self.assertEqual(node.publisher_qos["/tracking_camera_frame"], 10)

    def test_camera_starts_on_node_init_and_release_keeps_it_warm(self):
        module = _load_camera_capture_module()
        process = _FakeProcess()
        with (
            patch.object(module, "resolve_camera_device", return_value="/dev/video2"),
            patch.object(module.subprocess, "Popen", return_value=process) as popen,
        ):
            node = module.CameraCaptureNode()
            self.assertIs(node.publisher_types["/camera_frame"], _FakeCompressedImage)
            self.assertEqual(node.publisher_qos["/camera_frame"], 10)
            self.assertEqual(
                node.subscription_types["/image"],
                [_FakeCompressedImage],
            )
            popen.assert_called_once()
            node._on_command(_FakeString(encode_camera_command("acquire", "llm", 5)))
            command = popen.call_args.args[0]
            self.assertIn("video_device:=/dev/video2", command)
            self.assertNotIn("/image:=/camera_frame", command)

            node._on_command(_FakeString(encode_camera_command("release", "llm")))

        self.assertFalse(process.terminated)
        self.assertIs(node._camera_process, process)
        self.assertEqual(node.clients["/set_capture"].requests, [False])
        self.assertFalse(node._capture_enabled)

        node.destroy_node()
        self.assertTrue(process.terminated)
        self.assertIsNone(node._camera_process)

    def test_source_image_is_relayed_without_starting_second_camera(self):
        module = _load_camera_capture_module()
        process = _FakeProcess()
        with (
            patch.object(module, "resolve_camera_device", return_value="/dev/video8"),
            patch.object(module.subprocess, "Popen", return_value=process) as popen,
            patch.object(module, "jpeg_from_ros_image", return_value=b"frame"),
        ):
            node = module.CameraCaptureNode()
            node._on_source_image(
                _FakeCompressedImage(header=object(), format="jpeg", data=b"frame")
            )
            node._on_command(_FakeString(encode_camera_command("acquire", "web", 5)))
            node._on_source_image(
                _FakeCompressedImage(header=object(), format="jpeg", data=b"frame")
            )

        popen.assert_called_once()
        frames = node.publishers["/camera_frame"].messages
        self.assertEqual(len(frames), 1)
        self.assertIsInstance(frames[0], _FakeCompressedImage)
        self.assertEqual(frames[0].format, "jpeg")
        self.assertEqual(frames[0].data, b"frame")

    def test_hardware_only_lease_skips_jpeg_normalization_but_tracks_source_health(self):
        module = _load_camera_capture_module()
        with patch.object(module, "jpeg_from_ros_image", return_value=b"frame") as convert:
            node = module.CameraCaptureNode()
            node._on_command(_FakeString(encode_camera_command("acquire", "webrtc-h264:WALLY-01", 5)))
            node._on_source_image(_FakeCompressedImage(format="jpeg", data=b"frame"))
            self.assertFalse(convert.call_args.kwargs["flip_vertical"])
            self.assertEqual(node.publishers["/camera_frame"].messages, [])
            self.assertGreater(node._last_source_frame, 0)
            self.assertGreater(node._last_output_frame, 0)
            node._on_command(_FakeString(encode_camera_command("acquire", "tft", 5)))
            node._on_source_image(_FakeCompressedImage(format="jpeg", data=b"frame"))
            self.assertTrue(convert.call_args.kwargs["flip_vertical"])
            self.assertEqual(len(node.publishers["/camera_frame"].messages), 1)
            node._on_command(_FakeString(encode_camera_command("release", "tft")))
            node._on_source_image(_FakeCompressedImage(format="jpeg", data=b"frame"))
            self.assertFalse(convert.call_args.kwargs["flip_vertical"])

    def test_compressed_source_image_is_relayed(self):
        module = _load_camera_capture_module()
        source = _FakeCompressedImage(
            header=object(),
            format="jpeg",
            data=b"compressed-frame",
        )
        with patch.object(module, "jpeg_from_ros_image", return_value=b"frame"):
            node = module.CameraCaptureNode()
            node._on_command(_FakeString(encode_camera_command("acquire", "web", 5)))
            node._on_source_image(source)

        frames = node.publishers["/camera_frame"].messages
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].header, source.header)
        self.assertEqual(frames[0].format, "jpeg")
        self.assertEqual(frames[0].data, b"frame")

    def test_acquire_starts_the_single_camera_without_external_source_probe(self):
        module = _load_camera_capture_module()
        process = _FakeProcess()
        with (
            patch.object(module, "resolve_camera_device", return_value="/dev/video2"),
            patch.object(module.subprocess, "Popen", return_value=process) as popen,
        ):
            node = module.CameraCaptureNode()
            node._on_command(_FakeString(encode_camera_command("acquire", "tracking", 5)))
        popen.assert_called_once()

    def test_idle_source_frames_are_discarded_until_a_client_acquires(self):
        module = _load_camera_capture_module()
        process = _FakeProcess()
        with (
            patch.object(module, "resolve_camera_device", return_value="/dev/video2"),
            patch.object(module.subprocess, "Popen", return_value=process),
            patch.object(module, "jpeg_from_ros_image", return_value=b"frame"),
        ):
            node = module.CameraCaptureNode()
            node._tick()
            node._on_source_image(
                _FakeCompressedImage(header=object(), format="jpeg", data=b"frame")
            )

        self.assertEqual(node.publishers["/camera_frame"].messages, [])
        status = node.publishers["/camera_capture_status"].messages[-1]
        self.assertIn('"state":"standby"', status.data)

    def test_standby_resumes_capture_without_restarting_process(self):
        module = _load_camera_capture_module()
        process = _FakeProcess()
        with (
            patch.object(module, "resolve_camera_device", return_value="/dev/video2"),
            patch.object(module.subprocess, "Popen", return_value=process) as popen,
        ):
            node = module.CameraCaptureNode()
            node._tick()
            node._on_command(_FakeString(encode_camera_command("acquire", "web", 5)))

        self.assertEqual(node.clients["/set_capture"].requests, [False, True])
        self.assertTrue(node._capture_enabled)
        self.assertEqual(popen.call_count, 1)
        self.assertIs(node._camera_process, process)

    def test_expired_lease_keeps_camera_in_standby(self):
        module = _load_camera_capture_module()
        process = _FakeProcess()
        with (
            patch.object(module, "resolve_camera_device", return_value="/dev/video2"),
            patch.object(module.subprocess, "Popen", return_value=process),
            patch.object(module, "jpeg_from_ros_image", return_value=b"frame"),
        ):
            node = module.CameraCaptureNode()
            node._on_source_image(
                _FakeCompressedImage(header=object(), format="jpeg", data=b"frame")
            )
            node._leases.acquire("web", 1.0, now=time.monotonic() - 2.0)
            node._tick()

        self.assertFalse(process.terminated)
        self.assertIs(node._camera_process, process)
        self.assertEqual(node.clients["/set_capture"].requests, [False])
        self.assertFalse(node._capture_enabled)
        status = node.publishers["/camera_capture_status"].messages[-1]
        self.assertIn('"state":"standby"', status.data)

    def test_hot_camera_restarts_after_exit_without_active_clients(self):
        module = _load_camera_capture_module()
        first = _FakeProcess()
        second = _FakeProcess()
        with (
            patch.object(module, "resolve_camera_device", return_value="/dev/video2"),
            patch.object(module.subprocess, "Popen", side_effect=[first, second]) as popen,
        ):
            node = module.CameraCaptureNode()
            first.returncode = 1
            node._tick()
            node._retry_after = 0.0
            node._tick()

        self.assertEqual(popen.call_count, 2)
        self.assertIs(node._camera_process, second)

    def test_camera_without_first_frame_is_reaped(self):
        module = _load_camera_capture_module()
        process = _FakeProcess()
        with (
            patch.object(module, "resolve_camera_device", return_value="/dev/video0"),
            patch.object(module.subprocess, "Popen", return_value=process),
        ):
            node = module.CameraCaptureNode()
            node._on_command(_FakeString(encode_camera_command("acquire", "llm", 10)))
            node._process_started_at = time.monotonic() - node.FIRST_FRAME_TIMEOUT_SEC - 1.0
            node._tick()

        self.assertTrue(process.terminated)
        self.assertIsNone(node._camera_process)
        status = node.publishers["/camera_capture_status"].messages[-1]
        self.assertIn("首帧等待超时", status.data)

    def test_camera_that_stalls_after_first_frame_is_restarted(self):
        module = _load_camera_capture_module()
        process = _FakeProcess()
        with (
            patch.object(module, "resolve_camera_device", return_value="/dev/video0"),
            patch.object(module.subprocess, "Popen", return_value=process),
            patch.object(module, "jpeg_from_ros_image", return_value=b"frame"),
        ):
            node = module.CameraCaptureNode()
            node._on_command(_FakeString(encode_camera_command("acquire", "tracking", 10)))
            node._on_source_image(
                _FakeCompressedImage(header=object(), format="jpeg", data=b"frame")
            )
            now = time.monotonic()
            node._process_started_at = now - node.FRAME_TIMEOUT_SEC - 5.0
            node._last_source_frame = now - node.FRAME_TIMEOUT_SEC - 1.0
            node._tick()

        self.assertTrue(process.terminated)
        self.assertIsNone(node._camera_process)
        status = node.publishers["/camera_capture_status"].messages[-1]
        self.assertIn("画面中断", status.data)

    def test_process_exit_honors_retry_delay_before_restarting(self):
        module = _load_camera_capture_module()
        first = _FakeProcess()
        second = _FakeProcess()
        with (
            patch.object(module, "resolve_camera_device", return_value="/dev/video2"),
            patch.object(module.subprocess, "Popen", side_effect=[first, second]) as popen,
        ):
            node = module.CameraCaptureNode()
            self.assertEqual(popen.call_count, 1)
            first.returncode = 1
            node._tick()
            self.assertIsNone(node._camera_process)
            status = node.publishers["/camera_capture_status"].messages[-1]
            self.assertIn("已退出，退出码 1", status.data)

            # Next tick while now < retry_after should NOT restart process
            node._tick()
            self.assertEqual(popen.call_count, 1)
            self.assertIsNone(node._camera_process)

            # Once retry delay passed, next tick restarts process
            node._retry_after = 0.0
            node._tick()
            self.assertEqual(popen.call_count, 2)
            self.assertIs(node._camera_process, second)

    def test_first_frame_timeout_diagnostic_and_retry_delay(self):
        module = _load_camera_capture_module()
        first = _FakeProcess()
        second = _FakeProcess()
        with (
            patch.object(module, "resolve_camera_device", return_value="/dev/video0"),
            patch.object(module.subprocess, "Popen", side_effect=[first, second]) as popen,
        ):
            node = module.CameraCaptureNode()
            node._on_command(_FakeString(encode_camera_command("acquire", "llm", 10)))
            node._process_started_at = time.monotonic() - node.FIRST_FRAME_TIMEOUT_SEC - 1.0
            node._tick()

            self.assertTrue(first.terminated)
            self.assertIsNone(node._camera_process)
            status = node.publishers["/camera_capture_status"].messages[-1]
            self.assertIn("首帧等待超时", status.data)

            # Still in retry delay: should not restart immediately on next tick
            node._tick()
            self.assertEqual(popen.call_count, 1)

            # After retry delay: should restart
            node._retry_after = 0.0
            node._tick()
            self.assertEqual(popen.call_count, 2)
            self.assertIs(node._camera_process, second)

    def test_frame_timeout_and_retry_delay(self):
        module = _load_camera_capture_module()
        first = _FakeProcess()
        second = _FakeProcess()
        with (
            patch.object(module, "resolve_camera_device", return_value="/dev/video0"),
            patch.object(module.subprocess, "Popen", side_effect=[first, second]) as popen,
            patch.object(module, "jpeg_from_ros_image", return_value=b"frame"),
        ):
            node = module.CameraCaptureNode()
            node._on_command(_FakeString(encode_camera_command("acquire", "llm", 10)))
            node._on_source_image(
                _FakeCompressedImage(header=object(), format="jpeg", data=b"frame")
            )
            now = time.monotonic()
            node._process_started_at = now - 10.0
            node._last_source_frame = now - node.FRAME_TIMEOUT_SEC - 0.5
            node._tick()

            self.assertTrue(first.terminated)
            self.assertIsNone(node._camera_process)
            status = node.publishers["/camera_capture_status"].messages[-1]
            self.assertIn("画面中断", status.data)

            # Retry delay prevents immediate restart
            node._tick()
            self.assertEqual(popen.call_count, 1)

    def test_tick_publishes_streaming_when_lease_active_and_output_fresh(self):
        module = _load_camera_capture_module()
        process = _FakeProcess()
        with (
            patch.object(module, "resolve_camera_device", return_value="/dev/video0"),
            patch.object(module.subprocess, "Popen", return_value=process),
            patch.object(module, "jpeg_from_ros_image", return_value=b"frame"),
        ):
            node = module.CameraCaptureNode()
            node._on_command(_FakeString(encode_camera_command("acquire", "client", 10)))
            node._on_source_image(
                _FakeCompressedImage(header=object(), format="jpeg", data=b"frame")
            )
            now = time.monotonic()
            node._last_output_frame = now
            node._last_status_signature = None
            node._tick()
            status = node.publishers["/camera_capture_status"].messages[-1]
            self.assertIn('"state":"streaming"', status.data)


if __name__ == "__main__":
    unittest.main()
