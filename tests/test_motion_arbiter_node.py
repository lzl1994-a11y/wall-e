import importlib
import json
import sys
import types
import unittest
from unittest.mock import patch

from services.motion_arbiter import MotionArbiter, STOP_COMMAND


class _String:
    def __init__(self, data=""):
        self.data = data


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _GuardCondition:
    def __init__(self, callback):
        self.callback = callback
        self.trigger_count = 0

    def trigger(self):
        self.trigger_count += 1


class _Logger:
    def info(self, _message):
        pass

    def warning(self, _message):
        pass


class _Node:
    def __init__(self, _name, **kwargs):
        self.publisher = None
        self.subscriptions = {}
        self.guard = None
        self.init_kwargs = kwargs

    def create_publisher(self, _message_type, _topic, _qos):
        self.publisher = _Publisher()
        return self.publisher

    def create_subscription(self, _message_type, topic, callback, _qos):
        self.subscriptions[topic] = callback
        return object()

    def create_guard_condition(self, callback):
        self.guard = _GuardCondition(callback)
        return self.guard

    def get_logger(self):
        return _Logger()

    def destroy_node(self):
        pass


def _load_module():
    fake_rclpy = types.ModuleType("rclpy")
    fake_executors = types.ModuleType("rclpy.executors")
    fake_executors.SingleThreadedExecutor = object
    fake_node = types.ModuleType("rclpy.node")
    fake_node.Node = _Node
    fake_std = types.ModuleType("std_msgs.msg")
    fake_std.String = _String
    modules = {
        "rclpy": fake_rclpy,
        "rclpy.executors": fake_executors,
        "rclpy.node": fake_node,
        "std_msgs.msg": fake_std,
    }
    sys.modules.pop("nodes.motion_arbiter_node", None)
    with patch.dict(sys.modules, modules):
        return importlib.import_module("nodes.motion_arbiter_node")


FORWARD = {
    "left": {"action": 1, "throttle": 40},
    "right": {"action": 1, "throttle": 40},
}
REVERSE = {
    "left": {"action": 2, "throttle": 30},
    "right": {"action": 2, "throttle": 30},
}


class MotionArbiterNodeTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.module = _load_module()
        arbiter = MotionArbiter(timeout_sec=0.3, clock=lambda: self.now)
        self.node = self.module.MotionArbiterNode(
            arbiter=arbiter,
            start_watchdog=False,
        )

    def tearDown(self):
        sys.modules.pop("nodes.motion_arbiter_node", None)

    def test_command_publishes_immediately_without_periodic_timer(self):
        self.node._on_command("autonomy", _String(json.dumps(FORWARD)))

        self.assertEqual(len(self.node.publisher.messages), 1)
        self.assertEqual(json.loads(self.node.publisher.messages[0].data), FORWARD)
        self.assertAlmostEqual(self.node._watchdog_deadline, 10.3)
        self.assertFalse(hasattr(self.node, "_timer"))
        self.assertIsNone(self.node.guard)
        self.assertEqual(
            self.node.init_kwargs,
            {"enable_rosout": False, "start_parameter_services": False},
        )

    def test_watchdog_publishes_one_stop_after_expiry(self):
        self.node._on_command("autonomy", _String(json.dumps(FORWARD)))
        self.now = 10.31

        self.node._publish_selected()
        self.node._publish_selected()

        payloads = [json.loads(message.data) for message in self.node.publisher.messages]
        self.assertEqual(payloads, [FORWARD, STOP_COMMAND])
        self.assertIsNone(self.node._watchdog_deadline)

    def test_lower_priority_traffic_does_not_refresh_selected_source(self):
        self.node._on_command("joystick", _String(json.dumps(FORWARD)))
        joystick_deadline = self.node._watchdog_deadline
        self.now = 10.2

        self.node._on_command("autonomy", _String(json.dumps(REVERSE)))

        self.assertEqual(len(self.node.publisher.messages), 1)
        self.assertEqual(self.node._watchdog_deadline, joystick_deadline)

        self.now = 10.31
        self.node._publish_selected()
        self.assertEqual(
            json.loads(self.node.publisher.messages[-1].data),
            REVERSE,
        )

    def test_repeated_inactive_game_state_does_not_publish(self):
        self.node._on_game_state(_String("inactive"))
        self.node._on_game_state(_String("inactive"))

        self.assertEqual(self.node.publisher.messages, [])


if __name__ == "__main__":
    unittest.main()
