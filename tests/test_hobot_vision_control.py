import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class _FakeString:
    def __init__(self, data=""):
        self.data = data


class _FakeNode:
    def __init__(self, _name):
        self.callbacks = {}

    def create_subscription(self, _message_type, topic, callback, _qos):
        self.callbacks[topic] = callback
        return object()


class _FakeQoSProfile:
    def __init__(self, depth):
        self.depth = depth
        self.durability = None


def _load_module():
    fake_rclpy = types.ModuleType("rclpy")
    fake_node = types.ModuleType("rclpy.node")
    fake_node.Node = _FakeNode
    fake_qos = types.ModuleType("rclpy.qos")
    fake_qos.QoSProfile = _FakeQoSProfile
    fake_qos.DurabilityPolicy = types.SimpleNamespace(TRANSIENT_LOCAL="transient")
    fake_qos.qos_profile_sensor_data = object()
    fake_std = types.ModuleType("std_msgs.msg")
    fake_std.String = _FakeString
    fake_sensor = types.ModuleType("sensor_msgs.msg")
    fake_sensor.Image = type("Image", (), {})
    modules = {
        "rclpy": fake_rclpy,
        "rclpy.node": fake_node,
        "rclpy.qos": fake_qos,
        "std_msgs.msg": fake_std,
        "sensor_msgs.msg": fake_sensor,
    }
    sys.modules.pop("nodes.hobot_vision_node", None)
    with patch.dict(sys.modules, modules):
        return importlib.import_module("nodes.hobot_vision_node")


class VisionPipelineControlTests(unittest.TestCase):
    def test_start_and_stop_commands_toggle_pipeline(self):
        module = _load_module()
        control = module.VisionPipelineControl()
        self.assertFalse(control.enabled)
        control._on_command(_FakeString("start"))
        self.assertTrue(control.enabled)
        control._on_command(_FakeString("stop"))
        self.assertFalse(control.enabled)
        control._on_command(_FakeString("invalid"))
        self.assertFalse(control.enabled)

    def test_frame_health_counts_both_model_input_stages(self):
        module = _load_module()
        control = module.VisionPipelineControl()
        control._on_command(_FakeString("start"))

        control.callbacks["/image_nv12"](object())
        control.callbacks["/image_nv12"](object())
        control.callbacks["/image_padded_nv12"](object())

        self.assertEqual(control.decoded_frames, 2)
        self.assertEqual(control.padded_frames, 1)

    def test_stop_reaps_known_descendants_after_process_group(self):
        module = _load_module()
        process = Mock()
        process.poll.return_value = None

        with (
            patch.object(module.os, "getpgid", return_value=123, create=True),
            patch.object(module.os, "killpg", create=True),
            patch.object(module, "cleanup_old_processes") as cleanup,
        ):
            module._stop_pipeline(process)

        cleanup.assert_called_once_with()

    def test_every_pipeline_command_inherits_tros_environment(self):
        module = _load_module()
        process = Mock()

        with (
            patch.object(
                module,
                "_ensure_padder_binary",
                return_value=module.Path("/tmp/nv12_padder_node"),
            ),
            patch.object(module, "cleanup_old_processes"),
            patch.object(module.time, "sleep"),
            patch.object(module.os, "setsid", create=True),
            patch.object(module.subprocess, "Popen", return_value=process) as popen,
        ):
            self.assertIs(module._start_pipeline(), process)

        command = popen.call_args.args[0]
        self.assertEqual(command[:2], ["bash", "-c"])
        pipeline_script = command[2]
        self.assertTrue(
            pipeline_script.startswith(
                "source /opt/tros/humble/setup.bash && { "
            )
        )
        self.assertTrue(pipeline_script.endswith("wait -n; }"))
        self.assertIn("ros2 run hobot_codec", pipeline_script)
        self.assertNotIn("hobot_usb_cam", pipeline_script)
        self.assertIn("ros2 run mono2d_body_detection", pipeline_script)
        self.assertIn(
            "ai_msg_pub_topic_name:=/hobot_mono2d_body_detection",
            pipeline_script,
        )
        self.assertNotIn("hobot_mono2d_body_detection_raw", pipeline_script)
        self.assertNotIn("ai_msg_scaler_node.py", pipeline_script)
        self.assertNotIn("ros2 run websocket", pipeline_script)
        self.assertNotIn("/image_padded_jpeg", pipeline_script)
        self.assertIn("ros_img_topic_name:=/image_padded_nv12", pipeline_script)

    def test_padder_uses_sensor_qos_input_and_reliable_model_output(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "cpp_nodes"
            / "wali_nv12_padder"
            / "src"
            / "nv12_padder_node.cpp"
        )
        text = source.read_text(encoding="utf-8")

        self.assertIn("auto input_qos = rclcpp::SensorDataQoS()", text)
        self.assertIn(
            "create_publisher<sensor_msgs::msg::Image>(output_topic_, output_qos)",
            text,
        )

    def test_detector_cleanup_does_not_kill_the_camera_owner(self):
        module = _load_module()
        self.assertNotIn("hobot_usb_cam", module.cleanup_old_processes.__code__.co_consts)


if __name__ == "__main__":
    unittest.main()
