import importlib
import sys
import types
import unittest
from unittest.mock import patch


class _FakeString:
    def __init__(self, data=""):
        self.data = data


class _FakeNode:
    def __init__(self, _name):
        self.callback = None

    def create_subscription(self, _message_type, _topic, callback, _qos):
        self.callback = callback
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
    fake_std = types.ModuleType("std_msgs.msg")
    fake_std.String = _FakeString
    modules = {
        "rclpy": fake_rclpy,
        "rclpy.node": fake_node,
        "rclpy.qos": fake_qos,
        "std_msgs.msg": fake_std,
    }
    sys.modules.pop("nodes.hobot_vision_node", None)
    with patch.dict(sys.modules, modules):
        return importlib.import_module("nodes.hobot_vision_node")


class VisionPipelineControlTests(unittest.TestCase):
    def test_start_and_stop_commands_toggle_pipeline(self):
        module = _load_module()
        control = module.VisionPipelineControl()
        self.assertTrue(control.enabled)
        control._on_command(_FakeString("stop"))
        self.assertFalse(control.enabled)
        control._on_command(_FakeString("start"))
        self.assertTrue(control.enabled)
        control._on_command(_FakeString("invalid"))
        self.assertTrue(control.enabled)


if __name__ == "__main__":
    unittest.main()
