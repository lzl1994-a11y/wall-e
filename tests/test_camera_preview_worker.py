import json
import sys
import types
import unittest
from unittest.mock import patch

from services import camera_preview_worker


class _FakeRosImage:
    encoding = "jpeg"
    data = b"jpeg-frame"
    width = 640
    height = 480


class _FakeString:
    def __init__(self, data=""):
        self.data = data


class CameraPreviewWorkerTests(unittest.TestCase):
    def test_worker_leases_and_streams_only_camera_frame(self):
        state = {"initialized": False, "emitted": False}
        commands = []
        subscription_types = {}

        class FakePublisher:
            def __init__(self, topic):
                self.topic = topic

            def publish(self, message):
                if self.topic == "/camera_capture_cmd":
                    commands.append(json.loads(message.data))

        class FakeNode:
            def __init__(self, _name):
                self.callbacks = {}

            def create_publisher(self, _message_type, topic, _qos):
                return FakePublisher(topic)

            def create_subscription(self, message_type, topic, callback, _qos):
                subscription_types[topic] = message_type
                self.callbacks[topic] = callback
                return object()

            def destroy_node(self):
                pass

        fake_rclpy = types.ModuleType("rclpy")
        fake_rclpy.ok = lambda: state["initialized"] and not state["emitted"]

        def init(args=None):
            del args
            state["initialized"] = True

        def spin_once(node, timeout_sec=0.0):
            del timeout_sec
            if not state["emitted"]:
                node.callbacks["/camera_capture_status"](
                    _FakeString('{"state":"starting"}')
                )
                node.callbacks["/camera_capture_status"](
                    _FakeString('{"state":"error","error":"temporary"}')
                )
                node.callbacks["/camera_frame"](_FakeRosImage())
                state["emitted"] = True

        fake_rclpy.init = init
        fake_rclpy.spin_once = spin_once
        fake_rclpy.shutdown = lambda: None
        fake_node_module = types.ModuleType("rclpy.node")
        fake_node_module.Node = FakeNode
        fake_qos_module = types.ModuleType("rclpy.qos")
        fake_qos_module.qos_profile_sensor_data = object()
        fake_sensor_module = types.ModuleType("sensor_msgs.msg")
        fake_sensor_module.CompressedImage = _FakeRosImage
        fake_std_module = types.ModuleType("std_msgs.msg")
        fake_std_module.String = _FakeString
        emitted = []

        with (
            patch.dict(sys.modules, {
                "rclpy": fake_rclpy,
                "rclpy.node": fake_node_module,
                "rclpy.qos": fake_qos_module,
                "sensor_msgs.msg": fake_sensor_module,
                "std_msgs.msg": fake_std_module,
            }),
            patch.object(camera_preview_worker, "emit", side_effect=emitted.append),
        ):
            result = camera_preview_worker.stream_camera_frames(8.0)

        self.assertEqual(result, 0)
        self.assertEqual(commands[0]["action"], "acquire")
        self.assertEqual(commands[-1]["action"], "release")
        self.assertIs(subscription_types["/camera_frame"], _FakeRosImage)
        frames = [item for item in emitted if item.get("type") == "frame"]
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0]["source"], "/camera_frame")
        phases = [item["phase"] for item in emitted if item.get("type") == "status"]
        self.assertEqual(phases[-2:], ["waiting_frame", "waiting_frame"])

    def test_ros_import_error_is_reported_without_uvc_fallback(self):
        real_import = __import__
        emitted = []

        def blocked_ros_import(name, *args, **kwargs):
            if name == "rclpy":
                raise ImportError("rclpy not installed")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=blocked_ros_import),
            patch.object(camera_preview_worker, "emit", side_effect=emitted.append),
        ):
            result = camera_preview_worker.stream_camera_frames(8.0)

        self.assertEqual(result, 2)
        self.assertIn("ROS 摄像头环境不可用", emitted[0]["error"])
        self.assertFalse(hasattr(camera_preview_worker, "stream_uvc_frames"))


if __name__ == "__main__":
    unittest.main()
