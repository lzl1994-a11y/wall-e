import importlib
import sys
import time
import types
import unittest
from unittest.mock import patch

from services.camera_capture_protocol import encode_camera_command


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


class _FakeNodeBase:
    def __init__(self, _name):
        self.publishers = {}
        self.publisher_types = {}
        self.subscriptions = {}
        self.subscription_types = {}

    def create_publisher(self, message_type, topic, _qos):
        publisher = _FakePublisher(topic)
        self.publishers[topic] = publisher
        self.publisher_types[topic] = message_type
        return publisher

    def create_subscription(self, message_type, topic, callback, _qos):
        self.subscriptions[topic] = callback
        self.subscription_types[topic] = message_type
        return object()

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
    modules = {
        "rclpy": fake_rclpy,
        "rclpy.node": fake_node_module,
        "rclpy.qos": fake_qos_module,
        "sensor_msgs.msg": fake_sensor_module,
        "std_msgs.msg": fake_std_module,
    }
    sys.modules.pop("nodes.camera_capture_node", None)
    with patch.dict(sys.modules, modules):
        return importlib.import_module("nodes.camera_capture_node")


class CameraCaptureNodeTests(unittest.TestCase):
    def test_acquire_starts_remapped_hobot_camera_and_release_stops_it(self):
        module = _load_camera_capture_module()
        process = _FakeProcess()
        with (
            patch.object(module, "resolve_camera_device", return_value="/dev/video2"),
            patch.object(module.subprocess, "Popen", return_value=process) as popen,
        ):
            node = module.CameraCaptureNode()
            self.assertIs(node.publisher_types["/camera_frame"], _FakeCompressedImage)
            self.assertIs(node.subscription_types["/image"], _FakeImage)
            self.assertIs(
                node.subscription_types["/camera_frame"],
                _FakeCompressedImage,
            )
            node._on_command(_FakeString(encode_camera_command("acquire", "llm", 5)))
            command = popen.call_args.args[0]
            self.assertIn("video_device:=/dev/video2", command)
            self.assertIn("/image:=/camera_frame", command)

            node._on_command(_FakeString(encode_camera_command("release", "llm")))

        self.assertTrue(process.terminated)
        self.assertIsNone(node._camera_process)

    def test_active_tracking_image_is_relayed_without_starting_second_camera(self):
        module = _load_camera_capture_module()
        with (
            patch.object(module.subprocess, "Popen") as popen,
            patch.object(module, "jpeg_from_ros_image", return_value=b"frame"),
        ):
            node = module.CameraCaptureNode()
            node._on_tracking_image(_FakeImage())
            node._on_command(_FakeString(encode_camera_command("acquire", "web", 5)))
            node._on_tracking_image(_FakeImage())

        popen.assert_not_called()
        frames = node.publishers["/camera_frame"].messages
        self.assertEqual(len(frames), 1)
        self.assertIsInstance(frames[0], _FakeCompressedImage)
        self.assertEqual(frames[0].format, "jpeg")
        self.assertEqual(frames[0].data, b"frame")

    def test_tracking_publisher_prevents_a_second_camera_during_startup(self):
        module = _load_camera_capture_module()
        with (
            patch.object(module.CameraCaptureNode, "count_publishers", return_value=1),
            patch.object(module.subprocess, "Popen") as popen,
        ):
            node = module.CameraCaptureNode()
            node._on_command(_FakeString(encode_camera_command("acquire", "llm", 5)))

        popen.assert_not_called()
        status = node.publishers["/camera_capture_status"].messages[-1]
        self.assertIn('"source":"/image"', status.data)

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


if __name__ == "__main__":
    unittest.main()
