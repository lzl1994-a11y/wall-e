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
        node._eye_blink_ms = None
        node._eye_effective_blink_ms = None
        node._eye_speaking = False
        node._eye_blink_revision = 0
        node._eye_blink_paused = False
        node.get_logger = MagicMock()
        node._eye_response_publisher = MagicMock()
        node._eye_status_publisher = MagicMock()
        self.node = node

    def _response_body(self):
        message = self.node._eye_response_publisher.publish.call_args.args[0]
        return json.loads(message.data)

    def _run_and_ack(self, request_id, command, expected, line):
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

    def test_speaking_temporarily_disables_and_restores_blink(self):
        self.node._eye_blink_ms = 6000
        self.node._eye_effective_blink_ms = 6000
        self.node._eye_speaking = True
        self.node._eye_blink_revision = 1
        with patch.object(self.node, "_exchange_eye_command", return_value={"ok": True}) as exchange:
            self.node._sync_eye_blink(1)
            exchange.assert_called_once_with("eyeconfig:blinkMs=0", "ack")
            self.assertEqual(self.node._eye_blink_ms, 6000)
            self.node._eye_speaking = False
            self.node._eye_blink_revision = 2
            self.node._on_serial_line("EYE:STATE:blinkMs=0")
            self.assertEqual(self.node._eye_blink_ms, 6000)
            self.node._sync_eye_blink(2)
            self.assertEqual(exchange.call_args.args, ("eyeconfig:blinkMs=6000", "ack"))

    def test_blink_configuration_while_speaking_is_not_reenabled_on_wire(self):
        self.node._eye_speaking = True
        self._run_and_ack("blink", "eyeconfig:blinkMs=7000", "ack", "EYE:OK")
        self.node.bridge.send_raw.assert_called_once_with("eyeconfig:blinkMs=0\n", wake_screen=False)
        self.assertEqual(self._response_body()["state"], {"blinkMs": 7000})
        self.assertEqual(self.node._eye_blink_ms, 7000)

    def test_user_disabled_blink_stays_disabled_after_speaking(self):
        self.node._eye_blink_ms = self.node._eye_effective_blink_ms = 0
        self.node._eye_blink_revision = 2
        with patch.object(self.node, "_exchange_eye_command") as exchange:
            self.node._sync_eye_blink(2)
            exchange.assert_not_called()

    def test_stale_speaking_transition_does_not_send_commands(self):
        self.node._eye_blink_revision = 2
        with patch.object(self.node, "_exchange_eye_command") as exchange:
            self.node._sync_eye_blink(1)
            exchange.assert_not_called()

    def test_failed_blink_ack_does_not_change_user_configuration(self):
        self.node._eye_blink_ms = 5000
        self._run_and_ack("blink-error", "eyeconfig:blinkMs=0", "ack", "EYE:ERR:invalid_config")
        self.assertFalse(self._response_body()["ok"])
        self.assertEqual(self.node._eye_blink_ms, 5000)

    def test_tts_chunks_and_turn_end_do_not_repeat_pause_or_resume_early(self):
        from services.speech.tts_protocol import encode_turn_end

        with patch.object(self.module.threading, "Thread") as worker:
            self.node._on_eye_tts(_String("hello"))
            self.node._on_eye_tts(_String("world"))
            self.node._on_eye_tts(_String(encode_turn_end("turn")))
            self.assertTrue(self.node._eye_speaking)
            self.assertEqual(worker.call_count, 1)
            self.node._on_eye_playback(_String("idle"))
            self.assertFalse(self.node._eye_speaking)
            self.assertEqual(worker.call_count, 2)

    def test_pause_failure_is_reported_without_recording_unconfirmed_device_state(self):
        self.node._eye_blink_ms = self.node._eye_effective_blink_ms = 4500
        self.node._eye_speaking = True
        with patch.object(self.node, "_exchange_eye_command", return_value={"ok": False, "error": "not_ready"}):
            self.node._sync_eye_blink(0)
        self.assertEqual(self.node._eye_effective_blink_ms, 4500)
        self.assertFalse(self.node._eye_blink_paused)
        self.node.get_logger().error.assert_called_once()

    def test_scale_cannot_exceed_device_current_constraints(self):
        self.node._last_eye_state = {"minScale": 0.6, "maxScale": 1.2}
        with patch.object(self.node, "_exchange_eye_command") as exchange:
            self.node._run_eyeconfig_request("scale", "eyeconfig:scale=1.3", "ack")
            exchange.assert_not_called()
        self.assertFalse(self._response_body()["ok"])
        self.assertIn("minScale/maxScale", self._response_body()["error"])


if __name__ == "__main__":
    unittest.main()
