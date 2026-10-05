import importlib
import json
import sys
import threading
import types
import unittest
from unittest.mock import MagicMock, patch

from services.hardware.esp32_netcfg import validate_network_payload


class _String:
    def __init__(self, data=""):
        self.data = data


class SerialNetcfgStartupTests(unittest.TestCase):
    def _load_module(self):
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
            return importlib.import_module("nodes.serial_ros_node")

    def test_startup_queries_then_always_pushes_ram_session_and_statuses(self):
        module = self._load_module()
        node = module.SerialNode.__new__(module.SerialNode)
        node._tft_preview_ready = threading.Event()
        node._tft_preview_ready.set()
        node._shutdown_event = threading.Event()
        node._netcfg_request_lock = threading.Lock()
        node._netcfg_status_publisher = MagicMock()
        node.get_logger = MagicMock(return_value=MagicMock())
        node.bridge = MagicMock()
        node.bridge.run_exclusive.side_effect = lambda operation: operation(object())
        node.netcfg = MagicMock()
        node.netcfg.query.return_value = {
            "wifi": [{"ssid": "same"}, {"ssid": ""}, {"ssid": ""}],
            "host": "192.168.1.20",
            "port": 9000,
        }
        node.netcfg.save_and_apply.side_effect = lambda settings, stream, on_phase: (
            on_phase("configuring"),
            on_phase("connected"),
            {"set_seq": 1, "apply_seq": 2},
        )[-1]
        saved = validate_network_payload(
            {"wifi": [{"ssid": "same", "password": "secret"}] + [{"ssid": "", "password": ""}] * 2, "host": "192.168.1.20", "port": 9000}
        )

        with patch.object(module, "load_saved_network_settings", return_value=saved), patch.object(
            module,
            "resolve_session_network_settings",
            return_value=(saved, {"ssid_source": "detected", "host_source": "detected"}),
        ):
            node._apply_saved_network_on_start()

        node.netcfg.query.assert_called_once()
        node.netcfg.save_and_apply.assert_called_once()
        statuses = [
            json.loads(call.args[0].data)
            for call in node._netcfg_status_publisher.publish.call_args_list
        ]
        self.assertEqual([item["state"] for item in statuses], ["configuring", "connected"])
        self.assertTrue(statuses[0]["request_id"])
        self.assertEqual(statuses[0]["request_id"], statuses[1]["request_id"])
        self.assertTrue(all("wifi" not in item for item in statuses))
        self.assertTrue(all("password" not in repr(item) for item in statuses))

    def test_startup_replays_host_eye_config_through_existing_serial_owner(self):
        module = self._load_module()
        node = module.SerialNode.__new__(module.SerialNode)
        node._shutdown_event = threading.Event()
        node._eye_request_lock = threading.Lock()
        node._eye_response_condition = threading.Condition()
        node._eye_speaking = False
        node._last_eye_state = {}
        node._eye_blink_ms = None
        node._eye_effective_blink_ms = None
        node._eye_auto_blink_enabled = True
        node._eye_blink_paused = False
        node.get_logger = MagicMock(return_value=MagicMock())
        node._exchange_eye_command = MagicMock(
            return_value={"ok": True, "state": {"blinkMs": 4500}}
        )
        settings = {
            "color": "FF7A45",
            "ringColor": "FFB347",
            "dotColor": "FFE08A",
            "brightness": 0.85,
            "ringBrightness": 0.65,
            "dotBrightness": 0.45,
            "scale": 1.0,
            "glow": 22,
            "lookX": 15,
            "lookY": -8,
            "breathMs": 2400,
            "autoBlink": True,
            "blinkMs": 6000,
            "ring": True,
            "dots": 48,
            "mood": "heart",
        }
        commands = ["eyeconfig:color=FF7A45", "eyeconfig:mood=heart", "eyeaction:look:x=15,y=-8"]
        with patch.object(module, "load_saved_eye_config", return_value=settings), patch.object(
            module, "eye_config_commands", return_value=commands
        ):
            node._apply_saved_eye_on_start()

        self.assertEqual(
            [call.args for call in node._exchange_eye_command.call_args_list],
            [("eyeconfig:query", "state"),
             ("eyeconfig:color=FF7A45", "ack"),
             ("eyeconfig:mood=heart", "ack"),
             ("eyeaction:look:x=15,y=-8", "ack")],
        )
        self.assertEqual(node._eye_blink_ms, 6000)
        self.assertEqual(node._last_eye_state["mood"], "heart")


if __name__ == "__main__":
    unittest.main()
