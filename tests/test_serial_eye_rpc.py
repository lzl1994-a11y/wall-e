import importlib
import json
import sys
import threading
import types
import unittest
from collections import deque
from unittest.mock import MagicMock, patch


class _String:
    def __init__(self, data=""):
        self.data = data


class SerialEyeRpcTests(unittest.TestCase):
    def setUp(self):
        fake_node = types.ModuleType("rclpy.node")
        fake_node.Node = object
        fake_qos = types.ModuleType("rclpy.qos")
        fake_qos.QoSProfile = MagicMock()
        fake_qos.DurabilityPolicy = MagicMock()
        fake_qos.ReliabilityPolicy = MagicMock()
        fake_messages = types.ModuleType("std_msgs.msg")
        fake_messages.String = _String
        sys.modules.pop("nodes.serial_ros_node", None)
        with patch.dict(
            sys.modules,
            {
                "rclpy": types.ModuleType("rclpy"),
                "rclpy.node": fake_node,
                "rclpy.qos": fake_qos,
                "std_msgs": types.ModuleType("std_msgs"),
                "std_msgs.msg": fake_messages,
            },
        ):
            self.module = importlib.import_module("nodes.serial_ros_node")

        node = self.module.SerialNode.__new__(self.module.SerialNode)
        node.bridge = MagicMock()
        node.bridge.send_raw.return_value = True
        node._shutdown_event = threading.Event()
        node._eye_request_lock = threading.Lock()
        node._eye_response_condition = threading.Condition()
        node._eye_events = deque(maxlen=64)
        node._eye_event_sequence = 0
        node._last_eye_state = {}
        node._eye_response_publisher = MagicMock()
        node._eye_status_publisher = MagicMock()
        self.node = node

    def _response_body(self):
        message = self.node._eye_response_publisher.publish.call_args.args[0]
        return json.loads(message.data)

    def _run_and_ack(self, request_id, command, expected, line):
        self.node._eye_request_lock.acquire()
        worker = threading.Thread(
            target=self.node._run_eyeconfig_request,
            args=(request_id, command, expected),
        )
        worker.start()
        for _ in range(100):
            if self.node.bridge.send_raw.called:
                break
            threading.Event().wait(0.01)
        self.assertTrue(self.node.bridge.send_raw.called)
        self.node._on_serial_line(line)
        worker.join(1.0)
        self.assertFalse(worker.is_alive())

    def test_command_uses_shared_bridge_and_correlates_eye_ok(self):
        self._run_and_ack("request-1", "eyeconfig:ring=1", "ack", "EYE:OK")
        self.node.bridge.send_raw.assert_called_once_with(
            "eyeconfig:ring=1\n", wake_screen=False
        )
        body = self._response_body()
        self.assertTrue(body["ok"])
        self.assertEqual(body["event"]["kind"], "ok")

    def test_query_returns_accumulated_state_fields(self):
        self._run_and_ack(
            "request-2",
            "eyeconfig:query",
            "state",
            "EYE:STATE color=00e5ff mood=heart dots=48",
        )
        body = self._response_body()
        self.assertEqual(
            body["state"],
            {"color": "00E5FF", "mood": "heart", "dots": 48},
        )


if __name__ == "__main__":
    unittest.main()
