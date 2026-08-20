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

    def warning(self, _message):
        pass

    def warn(self, _message):
        pass

    def error(self, _message):
        pass


class _Node:
    def __init__(self, _name):
        self.publishers = {}

    def create_publisher(self, _message_type, topic, _qos):
        publisher = _Publisher()
        self.publishers[topic] = publisher
        return publisher

    def create_subscription(self, *_args):
        return object()

    def create_timer(self, *_args):
        return object()

    def destroy_timer(self, _timer):
        pass

    def get_logger(self):
        return _Logger()


def _load_module():
    fake_rclpy = types.ModuleType("rclpy")
    fake_node = types.ModuleType("rclpy.node")
    fake_node.Node = _Node
    fake_std = types.ModuleType("std_msgs.msg")
    fake_std.String = _String
    modules = {
        "rclpy": fake_rclpy,
        "rclpy.node": fake_node,
        "std_msgs.msg": fake_std,
    }
    sys.modules.pop("nodes.sequence_ros_node", None)
    with patch.dict(sys.modules, modules):
        return importlib.import_module("nodes.sequence_ros_node")


class SequenceMotorHeartbeatTests(unittest.TestCase):
    def test_motor_action_is_refreshed_until_deadline_then_stopped(self):
        module = _load_module()
        with patch.object(module.time, "monotonic", return_value=10.0):
            node = module.SequenceRosNode()
            node._dispatch_action({"type": "motor", "direction": "forward", "duration": 1.0})

        publisher = node.publishers["/motor_cmd/autonomy"]
        self.assertEqual(len(publisher.messages), 1)

        with patch.object(module.time, "monotonic", return_value=10.5):
            node._tick()
        self.assertEqual(len(publisher.messages), 2)
        self.assertEqual(json.loads(publisher.messages[-1].data)["left"]["action"], 1)

        with patch.object(module.time, "monotonic", return_value=11.1):
            node._tick()
        stop = json.loads(publisher.messages[-1].data)
        self.assertEqual(stop["left"]["action"], 0)
        self.assertEqual(stop["right"]["action"], 0)
        self.assertIsNone(node._active_motor_cmd)
        sys.modules.pop("nodes.sequence_ros_node", None)


if __name__ == "__main__":
    unittest.main()
