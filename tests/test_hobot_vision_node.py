import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class _FakeNode:
    def __init__(self, _name):
        self.subscriptions = {}

    def create_subscription(self, _message_type, topic, callback, _qos):
        self.subscriptions[topic] = callback
        return object()


class _FakeQoSProfile:
    def __init__(self, *, depth):
        self.depth = depth
        self.durability = None


class _FakeString:
    def __init__(self, data=""):
        self.data = data


def _load_module():
    fake_rclpy = types.ModuleType("rclpy")
    fake_node = types.ModuleType("rclpy.node")
    fake_node.Node = _FakeNode
    fake_qos = types.ModuleType("rclpy.qos")
    fake_qos.DurabilityPolicy = types.SimpleNamespace(TRANSIENT_LOCAL="transient")
    fake_qos.QoSProfile = _FakeQoSProfile
    fake_qos.qos_profile_sensor_data = object()
    fake_sensor = types.ModuleType("sensor_msgs.msg")
    fake_sensor.Image = type("Image", (), {})
    fake_std = types.ModuleType("std_msgs.msg")
    fake_std.String = _FakeString
    modules = {
        "rclpy": fake_rclpy,
        "rclpy.node": fake_node,
        "rclpy.qos": fake_qos,
        "sensor_msgs.msg": fake_sensor,
        "std_msgs.msg": fake_std,
    }
    sys.modules.pop("nodes.hobot_vision_node", None)
    with patch.dict(sys.modules, modules):
        return importlib.import_module("nodes.hobot_vision_node")


class HotStandbyVisionTests(unittest.TestCase):
    def test_detector_subscribes_to_leased_camera_frame_topic(self):
        module = _load_module()
        process = object()
        with (
            patch.object(module, "cleanup_old_processes"),
            patch.object(module.time, "sleep"),
            patch.object(module.os, "setsid", create=True),
            patch.object(module.subprocess, "Popen", return_value=process) as popen,
        ):
            result = module._start_pipeline(Path("/tmp/nv12_padder_node"))

        self.assertIs(result, process)
        command = popen.call_args.args[0]
        self.assertEqual(command[:2], ["bash", "-c"])
        self.assertIn("sub_topic:=/camera_frame", command[2])
        self.assertNotIn("sub_topic:=/image ", command[2])

    def test_stop_command_disables_result_use_without_owning_processes(self):
        module = _load_module()
        control = module.VisionPipelineControl()
        control._on_command(_FakeString("start"))
        self.assertTrue(control.enabled)
        control._on_command(_FakeString("stop"))
        self.assertFalse(control.enabled)


if __name__ == "__main__":
    unittest.main()
