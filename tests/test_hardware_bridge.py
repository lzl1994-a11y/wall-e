import importlib
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from services.motor_watchdog import MotorWatchdog


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeString:
    def __init__(self):
        self.data = ""


def import_hardware_bridge():
    fake_rclpy = types.ModuleType("rclpy")
    fake_rclpy_node = types.ModuleType("rclpy.node")
    fake_std_msgs = types.ModuleType("std_msgs")
    fake_std_msgs_msg = types.ModuleType("std_msgs.msg")
    fake_rclpy_node.Node = object
    fake_std_msgs_msg.String = FakeString

    with patch.dict(
        sys.modules,
        {
            "rclpy": fake_rclpy,
            "rclpy.node": fake_rclpy_node,
            "std_msgs": fake_std_msgs,
            "std_msgs.msg": fake_std_msgs_msg,
        },
    ):
        return importlib.import_module("nodes.hardware_bridge_node")


hardware_bridge = import_hardware_bridge()


class HardwareBridgeBatchingTests(unittest.TestCase):
    def make_node(self):
        node = hardware_bridge.HardwareBridgeNode.__new__(
            hardware_bridge.HardwareBridgeNode
        )
        node._DUTY_MIN = 1638
        node._DUTY_MAX = 8192
        node._MOTOR_HIGH = 65535
        node._MOTOR_LOW = 0
        node._state = [0] * 15
        node._name_to_ch = {"head_yaw": 4, "neck_top": 5, "neck_bottom": 6}
        node._motor_inverted = {"left": False, "right": False}
        node._state_dirty = False
        self.now = 10.0
        node._motor_watchdog = MotorWatchdog(clock=lambda: self.now)
        node._raw_pub = Mock()
        node.get_logger = Mock()
        return node

    def command(self, payload):
        msg = FakeString()
        msg.data = json.dumps(payload)
        return msg

    def test_multiple_servo_updates_are_coalesced_into_one_packet(self):
        node = self.make_node()

        node._on_servo_cmd(self.command({"name": "head_yaw", "pwm": 5100}))
        node._on_servo_cmd(self.command({"name": "neck_top", "pwm": 5200}))
        node._on_servo_cmd(self.command({"name": "head_yaw", "pwm": 5300}))

        node._raw_pub.publish.assert_not_called()
        node._flush_state()
        node._raw_pub.publish.assert_called_once()

        values = node._raw_pub.publish.call_args.args[0].data.split(":", 1)[1]
        values = [int(value) for value in values.split(",")]
        self.assertEqual(values[4], 5300)
        self.assertEqual(values[5], 5200)

        node._flush_state()
        node._raw_pub.publish.assert_called_once()

    def test_unchanged_state_is_not_published_again(self):
        node = self.make_node()
        command = self.command({"name": "neck_bottom", "pwm": 4000})

        node._on_servo_cmd(command)
        node._flush_state()
        node._on_servo_cmd(command)
        node._flush_state()

        node._raw_pub.publish.assert_called_once()

    def test_motor_and_servo_updates_share_the_same_packet(self):
        node = self.make_node()

        node._on_motor_cmd(
            self.command(
                {
                    "left": {"action": 1, "throttle": 50},
                    "right": {"action": 2, "throttle": 25},
                }
            )
        )
        node._on_servo_cmd(self.command({"name": "head_yaw", "pwm": 5000}))
        node._flush_state()

        node._raw_pub.publish.assert_called_once()
        values = node._raw_pub.publish.call_args.args[0].data.split(":", 1)[1]
        values = [int(value) for value in values.split(",")]
        self.assertEqual(values[4], 5000)
        self.assertEqual(values[9:15], [65535, 0, 32767, 0, 65535, 16383])

    def test_motor_watchdog_forces_stop_after_heartbeat_loss(self):
        node = self.make_node()
        node._on_motor_cmd(
            self.command(
                {
                    "left": {"action": 1, "throttle": 50},
                    "right": {"action": 1, "throttle": 50},
                }
            )
        )
        node._flush_state()
        node._raw_pub.reset_mock()

        self.now += 0.31
        node._flush_state()

        node._raw_pub.publish.assert_called_once()
        values = node._raw_pub.publish.call_args.args[0].data.split(":", 1)[1]
        values = [int(value) for value in values.split(",")]
        self.assertEqual(values[9:15], [0, 0, 0, 0, 0, 0])

    def test_invalid_motor_command_does_not_refresh_watchdog(self):
        node = self.make_node()
        node._on_motor_cmd(
            self.command(
                {
                    "left": {"action": 1, "throttle": 101},
                    "right": {"action": 1, "throttle": 50},
                }
            )
        )
        self.now += 0.31

        self.assertTrue(node._motor_watchdog.poll())


if __name__ == "__main__":
    unittest.main()
