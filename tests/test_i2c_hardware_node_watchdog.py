import importlib
import json
import sys
import types
import unittest
from unittest.mock import Mock, patch

from services.motor_watchdog import MotorWatchdog


class _String:
    def __init__(self, data=""):
        self.data = data


def _load_module():
    fake_rclpy = types.ModuleType("rclpy")
    fake_node = types.ModuleType("rclpy.node")
    fake_node.Node = object
    fake_std = types.ModuleType("std_msgs.msg")
    fake_std.String = _String
    modules = {
        "rclpy": fake_rclpy,
        "rclpy.node": fake_node,
        "std_msgs.msg": fake_std,
    }
    sys.modules.pop("nodes.i2c_hardware_node", None)
    with patch.dict(sys.modules, modules):
        return importlib.import_module("nodes.i2c_hardware_node")


class I2CHardwareNodeWatchdogTests(unittest.TestCase):
    def test_watchdog_stops_both_tracks_after_heartbeat_loss(self):
        module = _load_module()
        now = [30.0]
        node = module.I2CHardwareNode.__new__(module.I2CHardwareNode)
        node.driver = Mock()
        node.get_logger = lambda: Mock()
        node._motor_watchdog = MotorWatchdog(clock=lambda: now[0])
        message = _String(
            data=json.dumps(
                {
                    "left": {"action": 1, "throttle": 40},
                    "right": {"action": 1, "throttle": 40},
                }
            )
        )

        node._on_motor_cmd(message)
        node.driver.set_motor.reset_mock()
        now[0] += 0.31
        node._check_motor_watchdog()

        self.assertEqual(
            [call.args for call in node.driver.set_motor.call_args_list],
            [("left", 0, 0), ("right", 0, 0)],
        )
        sys.modules.pop("nodes.i2c_hardware_node", None)


if __name__ == "__main__":
    unittest.main()
