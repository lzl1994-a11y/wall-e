import importlib
import sys
import types
import unittest
from unittest.mock import Mock, patch


class _ExternalShutdownException(Exception):
    pass


def _load_module():
    fake_rclpy = types.ModuleType("rclpy")
    fake_rclpy.init = Mock()
    fake_rclpy.spin = Mock(side_effect=_ExternalShutdownException)
    fake_rclpy.ok = Mock(return_value=False)
    fake_rclpy.shutdown = Mock()

    fake_node_module = types.ModuleType("rclpy.node")
    fake_node_module.Node = type("Node", (), {})
    fake_executors = types.ModuleType("rclpy.executors")
    fake_executors.ExternalShutdownException = _ExternalShutdownException
    fake_ai = types.ModuleType("ai_msgs.msg")
    fake_ai.PerceptionTargets = type("PerceptionTargets", (), {})
    fake_sensor = types.ModuleType("sensor_msgs.msg")
    fake_sensor.Image = type("Image", (), {})

    modules = {
        "rclpy": fake_rclpy,
        "rclpy.node": fake_node_module,
        "rclpy.executors": fake_executors,
        "ai_msgs.msg": fake_ai,
        "sensor_msgs.msg": fake_sensor,
    }
    sys.modules.pop("nodes.ai_msg_scaler_node", None)
    with patch.dict(sys.modules, modules):
        return importlib.import_module("nodes.ai_msg_scaler_node")


class AIMsgScalerShutdownTests(unittest.TestCase):
    def test_external_shutdown_does_not_shutdown_context_twice(self):
        module = _load_module()
        node = Mock()

        with patch.object(module, "AIMSgScaler", return_value=node):
            module.main()

        node.destroy_node.assert_called_once_with()
        module.rclpy.shutdown.assert_not_called()


if __name__ == "__main__":
    unittest.main()
