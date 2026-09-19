import importlib
import json
import sys
import types
import unittest
from unittest.mock import patch


class _String:
    def __init__(self, data=""):
        self.data = data


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _Logger:
    def info(self, _message):
        pass


class _Node:
    def __init__(self, _name):
        self.publishers = {}
        self.subscriptions = {}
        self.timers = []

    def create_publisher(self, _message_type, topic, _qos):
        publisher = _Publisher()
        self.publishers[topic] = publisher
        return publisher

    def create_subscription(self, _message_type, topic, callback, _qos):
        self.subscriptions[topic] = callback
        return object()

    def create_timer(self, interval, callback):
        timer = (interval, callback)
        self.timers.append(timer)
        return timer

    def destroy_timer(self, timer):
        if timer in self.timers:
            self.timers.remove(timer)

    def get_logger(self):
        return _Logger()


def _load_module():
    fake_rclpy = types.ModuleType("rclpy")
    fake_node = types.ModuleType("rclpy.node")
    fake_node.Node = _Node
    fake_std = types.ModuleType("std_msgs.msg")
    fake_std.String = _String

    fake_evdev = types.ModuleType("evdev")
    fake_ecodes = types.ModuleType("evdev.ecodes")
    fake_evdev.ecodes = fake_ecodes
    fake_evdev.list_devices = lambda: []

    modules = {
        "rclpy": fake_rclpy,
        "rclpy.node": fake_node,
        "std_msgs.msg": fake_std,
        "evdev": fake_evdev,
        "evdev.ecodes": fake_ecodes,
    }
    sys.modules.pop("nodes.joy_control_node", None)
    with patch.dict(sys.modules, modules):
        return importlib.import_module("nodes.joy_control_node")


class JoyControlNodeContractTests(unittest.TestCase):
    def _create_node(self, module):
        with patch.object(module.JoyControlNode, "_start_scanning"):
            node = module.JoyControlNode()
        node.device = types.SimpleNamespace(path="/dev/input/event2", name="FakeJoy")
        return node

    def test_tick_publishes_manual_servo_payload(self):
        module = _load_module()
        node = self._create_node(module)

        # Set axes for right stick and triggers
        node._axes[module.AXIS_RX] = 0.5
        node._axes[module.AXIS_RY] = 0.0
        node._axes[module.AXIS_L2] = 0.4
        node._axes[module.AXIS_R2] = 0.6
        node._auto_timers["arm_l"] = 0.0
        node._auto_timers["arm_r"] = 0.0
        node._auto_timers["eyebrow_l"] = 0.0
        node._auto_timers["eyebrow_r"] = 0.0

        node._tick_loop()

        self.assertEqual(len(node.action_pub.messages), 1)
        payload = json.loads(node.action_pub.messages[0].data)

        self.assertEqual(payload["name"], "manual_servo")
        self.assertEqual(payload["source"], "joystick")
        self.assertEqual(payload["arguments"]["step_size"], node.servo_step_size)

        targets = payload["arguments"]["targets"]
        self.assertEqual(targets["head_yaw"], int(5000 - 0.5 * 2600))
        self.assertEqual(targets["eye_l"], int(7500 - 0.4 * 2500))
        self.assertEqual(targets["eye_r"], int(2000 + 0.6 * 2000))
        self.assertEqual(targets["arm_l"], 2000)
        self.assertEqual(targets["arm_r"], 8000)
        self.assertEqual(targets["eyebrow_l"], 8000)
        self.assertEqual(targets["eyebrow_r"], 1920)
        self.assertIn("neck_top", targets)
        self.assertIn("neck_bottom", targets)

        # Verify exact target keys order
        self.assertEqual(
            list(targets.keys()),
            [
                "head_yaw",
                "neck_top",
                "neck_bottom",
                "eye_l",
                "eye_r",
                "arm_l",
                "arm_r",
                "eyebrow_l",
                "eyebrow_r",
            ],
        )

    def test_motor_path_preserved_during_tick(self):
        module = _load_module()
        node = self._create_node(module)

        # Non-zero left stick drives differential mixing
        node._axes[module.AXIS_LY] = 0.6
        node._axes[module.AXIS_LX] = 0.2

        node._tick_loop()

        self.assertEqual(len(node.motor_pub.messages), 1)
        motor_cmd = json.loads(node.motor_pub.messages[0].data)
        self.assertIn("left", motor_cmd)
        self.assertIn("right", motor_cmd)
        self.assertTrue(node._was_moving)

        # Resetting axes stops motors
        node._axes[module.AXIS_LY] = 0.0
        node._axes[module.AXIS_LX] = 0.0

        node._tick_loop()

        self.assertEqual(len(node.motor_pub.messages), 2)
        stop_cmd = json.loads(node.motor_pub.messages[1].data)
        self.assertEqual(stop_cmd, module.STOP_COMMAND)
        self.assertFalse(node._was_moving)

    def test_game_mode_suppresses_servo_and_motor_output(self):
        module = _load_module()
        node = self._create_node(module)
        node._game_active = True

        node._axes[module.AXIS_RX] = 0.5
        node._axes[module.AXIS_LY] = 0.5

        node._tick_loop()

        self.assertEqual(node.action_pub.messages, [])
        self.assertEqual(node.motor_pub.messages, [])


if __name__ == "__main__":
    unittest.main()
