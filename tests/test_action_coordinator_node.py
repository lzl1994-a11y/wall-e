import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from services.action_arbitration import ActionArbiter


ROOT = Path(__file__).resolve().parents[1]


class _String:
    def __init__(self, *, data=""):
        self.data = data


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(json.loads(message.data))


class _Logger:
    def info(self, _message):
        pass

    def warning(self, _message):
        pass


def _load_node_class():
    fake_rclpy = types.ModuleType("rclpy")
    fake_node_module = types.ModuleType("rclpy.node")
    fake_node_module.Node = object
    fake_std_msgs = types.ModuleType("std_msgs")
    fake_std_msgs_msg = types.ModuleType("std_msgs.msg")
    fake_std_msgs_msg.String = _String
    modules = {
        "rclpy": fake_rclpy,
        "rclpy.node": fake_node_module,
        "std_msgs": fake_std_msgs,
        "std_msgs.msg": fake_std_msgs_msg,
    }
    module_path = ROOT / "nodes" / "action_coordinator_node.py"
    spec = importlib.util.spec_from_file_location(
        "action_coordinator_node_for_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module.ActionCoordinatorNode


class ActionCoordinatorNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.node_class = _load_node_class()

    def setUp(self):
        self.node = self.node_class.__new__(self.node_class)
        self.node._arbiter = ActionArbiter()
        self.node._command_pub = _Publisher()
        self.node._cancel_pub = _Publisher()
        self.node._status_pub = _Publisher()
        self.node.get_logger = lambda: _Logger()

    def request(self, request_id, name, source):
        self.node._on_request(_String(data=json.dumps({
            "request_id": request_id,
            "name": name,
            "arguments": {},
            "source": source,
        })))

    def test_forwards_independent_resources_and_rejects_lower_priority_conflict(self):
        self.request("joy", "manual_servo", "joystick")
        self.request("music", "control_music", "llm_dialog")
        self.request("llm", "play_sequence", "llm_dialog")

        self.assertEqual(
            [message["request_id"] for message in self.node._command_pub.messages],
            ["joy", "music"],
        )
        self.assertEqual(self.node._status_pub.messages[0]["request_id"], "llm")
        self.assertEqual(self.node._status_pub.messages[0]["status"], "rejected")
        self.assertEqual(
            self.node._status_pub.messages[0]["detail"],
            "resource_busy:joystick:manual_servo",
        )

    def test_terminal_status_releases_resource(self):
        self.request("first", "play_sequence", "joystick")
        self.node._on_status(_String(data=json.dumps({
            "request_id": "first",
            "name": "play_sequence",
            "status": "completed",
        })))
        self.request("second", "play_sequence", "llm_dialog")

        self.assertEqual(
            [message["request_id"] for message in self.node._command_pub.messages],
            ["first", "second"],
        )

    def test_preemption_emits_targeted_cancel_and_interrupted_status(self):
        self.request("old", "play_sequence", "llm_dialog")
        self.request("new", "move_chassis", "joystick")

        self.assertEqual(self.node._cancel_pub.messages[0]["request_id"], "old")
        self.assertEqual(
            self.node._cancel_pub.messages[0]["replacement_request_id"], "new"
        )
        interrupted = next(
            message for message in self.node._status_pub.messages
            if message["request_id"] == "old"
        )
        self.assertEqual(interrupted["status"], "interrupted")
        self.assertEqual(
            interrupted["detail"], "preempted_by:joystick:move_chassis"
        )

    def test_expired_cancellable_lease_is_cancelled_and_reported(self):
        self.node._arbiter.submit(
            "stale", "move_chassis", "llm_dialog", now=0
        )
        self.node._expire_leases()

        self.assertEqual(self.node._cancel_pub.messages[0]["request_id"], "stale")
        self.assertTrue(any(
            message["request_id"] == "stale"
            and message["status"] == "interrupted"
            and message["detail"] == "action_lease_expired"
            for message in self.node._status_pub.messages
        ))


if __name__ == "__main__":
    unittest.main()
