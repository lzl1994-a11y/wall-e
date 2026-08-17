import sys
import types
import unittest
from unittest.mock import patch
from unittest.mock import patch

from services import camera_preview_worker
from services.camera_preview_worker import jpeg_from_ros_message


class _FakeRosImage:
    encoding = "jpeg"
    data = b"jpeg-frame"
    width = 640
    height = 480


class _FakeCompressedRosImage:
    format = "bgr8; jpeg compressed bgr8"
    data = b"jpeg-frame"


class CameraPreviewWorkerTests(unittest.TestCase):
    def test_encoded_ros_image_is_reused_without_decoding(self):
        self.assertEqual(jpeg_from_ros_message(_FakeRosImage()), b"jpeg-frame")

    def test_empty_encoded_ros_image_is_ignored(self):
        message = _FakeRosImage()
        message.data = b""
        self.assertIsNone(jpeg_from_ros_message(message))

    def test_compressed_ros_image_is_reused_without_decoding(self):
        self.assertEqual(jpeg_from_ros_message(_FakeCompressedRosImage()), b"jpeg-frame")

    def test_compressed_jpeg_with_empty_format_is_reused(self):
        message = _FakeCompressedRosImage()
        message.format = ""
        message.data = b"\xff\xd8jpeg-frame"
        self.assertEqual(jpeg_from_ros_message(message), message.data)

    def test_ros_topic_frame_is_forwarded_without_opening_uvc(self):
        state = {"initialized": False, "emitted": False}

        class FakeNode:
            def __init__(self, _name):
                self.callbacks = []

            def create_subscription(self, _message_type, _topic, callback, _qos):
                self.callbacks.append(callback)
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
            node.callbacks[0](_FakeRosImage())
            state["emitted"] = True

        fake_rclpy.init = init
        fake_rclpy.spin_once = spin_once
        fake_rclpy.shutdown = lambda: None
        fake_node_module = types.ModuleType("rclpy.node")
        fake_node_module.Node = FakeNode
        fake_qos_module = types.ModuleType("rclpy.qos")
        fake_qos_module.qos_profile_sensor_data = object()
        fake_sensor_module = types.ModuleType("sensor_msgs.msg")
        fake_sensor_module.Image = _FakeRosImage
        fake_sensor_module.CompressedImage = _FakeCompressedRosImage
        emitted = []

        with (
            patch.dict(sys.modules, {
                "rclpy": fake_rclpy,
                "rclpy.node": fake_node_module,
                "rclpy.qos": fake_qos_module,
                "sensor_msgs.msg": fake_sensor_module,
            }),
            patch.object(camera_preview_worker, "emit", side_effect=emitted.append),
        ):
            using_ros, diagnostic = camera_preview_worker.stream_ros_frames(0.2, 8.0)

        self.assertTrue(using_ros)
        self.assertEqual(diagnostic, "")

        frame_messages = [item for item in emitted if item.get("type") == "frame"]
        self.assertEqual(len(frame_messages), 1)
        self.assertEqual(frame_messages[0]["source"], "/image_padded_jpeg")

    def test_ros_import_error_is_reported_for_the_web_status(self):
        real_import = __import__

        def blocked_ros_import(name, *args, **kwargs):
            if name == "rclpy":
                raise ImportError("rclpy not installed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=blocked_ros_import):
            using_ros, diagnostic = camera_preview_worker.stream_ros_frames(0.2, 8.0)

        self.assertFalse(using_ros)
        self.assertIn("ROS Python 环境不可用", diagnostic)


if __name__ == "__main__":
    unittest.main()
