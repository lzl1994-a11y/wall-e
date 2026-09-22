import importlib
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from services.motion.motor_watchdog import MotorWatchdog
from services.hardware.pca9685_output_state import Pca9685OutputState


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


class HardwareBridgeNodeBoundaryTests(unittest.TestCase):
    def make_node(self, servo_inits=None):
        node = hardware_bridge.HardwareBridgeNode.__new__(
            hardware_bridge.HardwareBridgeNode
        )
        node._name_to_ch = {"head_yaw": 4, "neck_top": 5, "neck_bottom": 6}
        node._motor_inverted = {"left": False, "right": False}
        node._state = Pca9685OutputState(servo_inits)
        node._state.mark_published()
        self.now = 10.0
        node._motor_watchdog = MotorWatchdog(clock=lambda: self.now)
        node._raw_pub = Mock()
        node.get_logger = Mock()
        return node

    def command(self, payload):
        msg = FakeString()
        msg.data = json.dumps(payload)
        return msg

    def test_json_ros_adapter_and_config_mapping(self):
        node = self.make_node()

        # 优先写有效 PWM
        node._on_servo_cmd(
            self.command({"name": "head_yaw", "pwm": 5100, "angle": 0})
        )
        self.assertIn("5100", node._state.encode().split(",")[4])

        # 没有有效 PWM 时按 angle 写入
        node._on_servo_cmd(self.command({"name": "head_yaw", "angle": 180}))
        self.assertIn("8192", node._state.encode().split(",")[4])

        # 未知名称与非法 JSON 不崩溃且不更新
        node._state.mark_published()
        node._on_servo_cmd(self.command({"name": "unknown_servo", "pwm": 4000}))
        bad_msg = FakeString()
        bad_msg.data = "invalid-json"
        node._on_servo_cmd(bad_msg)
        self.assertFalse(node._state.dirty)

    def test_multiple_commands_coalesce_into_single_publish(self):
        node = self.make_node()

        node._on_servo_cmd(self.command({"name": "head_yaw", "pwm": 5100}))
        node._on_servo_cmd(self.command({"name": "neck_top", "pwm": 5200}))
        node._on_servo_cmd(self.command({"name": "head_yaw", "pwm": 5300}))
        node._on_motor_cmd(
            self.command(
                {
                    "left": {"action": 1, "throttle": 50},
                    "right": {"action": 2, "throttle": 25},
                }
            )
        )

        node._raw_pub.publish.assert_not_called()
        node._flush_state()
        node._raw_pub.publish.assert_called_once()

        values = [
            int(v)
            for v in node._raw_pub.publish.call_args.args[0].data.split(":", 1)[1].split(",")
        ]
        self.assertEqual(values[4], 5300)
        self.assertEqual(values[5], 5200)
        self.assertEqual(values[9:15], [65535, 0, 32767, 0, 65535, 16383])

        # 无状态更新时不重复发布
        node._flush_state()
        node._raw_pub.publish.assert_called_once()

    def test_watchdog_timeout_forces_stop(self):
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
        values = [
            int(v)
            for v in node._raw_pub.publish.call_args.args[0].data.split(":", 1)[1].split(",")
        ]
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

    def test_node_payload_matches_state_object_and_dirty_preserved_on_failure(self):
        node = self.make_node()
        node._on_servo_cmd(self.command({"name": "neck_bottom", "pwm": 4200}))
        self.assertTrue(node._state.dirty)

        # 检查 payload 与 state.encode() 一致
        node._publish_state()
        self.assertEqual(node._raw_pub.publish.call_args.args[0].data, node._state.encode())

        # 当 publish 抛出异常时，dirty 必须保留
        node._raw_pub.publish.side_effect = RuntimeError("publish error")
        with self.assertRaises(RuntimeError):
            node._flush_state()
        self.assertTrue(node._state.dirty)


if __name__ == "__main__":
    unittest.main()
