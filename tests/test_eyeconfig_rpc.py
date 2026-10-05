import unittest
import threading
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from services.hardware.eyeconfig_rpc import (
    EYE_CONFIG_DEFAULTS,
    EyeConfigError,
    EyeConfigRpcClient,
    encode_eye_request,
    eye_config_commands,
    load_saved_eye_config,
    normalize_eye_config,
    normalize_eye_state,
    parse_eye_response,
    validate_eye_command,
)


class EyeConfigProtocolTests(unittest.TestCase):
    def test_firmware_boundaries_and_disabled_periods(self):
        for command in (
            "eyeconfig:scale=0.4", "eyeconfig:scale=1.5", "eyeconfig:glow=8",
            "eyeconfig:glow=30", "eyeconfig:dots=64", "eyeconfig:breathMs=0",
            "eyeconfig:breathMs=500", "eyeconfig:breathMs=10000",
            "eyeconfig:blinkMs=0", "eyeconfig:blinkMs=1000", "eyeconfig:blinkMs=15000",
            "eyeaction:look:x=-26,y=26",
        ):
            with self.subTest(command=command):
                self.assertEqual(validate_eye_command(command), command)
        for command in (
            "eyeconfig:scale=0.39", "eyeconfig:scale=1.51", "eyeconfig:glow=7",
            "eyeconfig:glow=31", "eyeconfig:dots=65", "eyeconfig:breathMs=499",
            "eyeconfig:breathMs=10001", "eyeconfig:blinkMs=999", "eyeconfig:blinkMs=15001",
            "eyeaction:look:x=27,y=0",
        ):
            with self.subTest(command=command):
                with self.assertRaises(EyeConfigError):
                    validate_eye_command(command)

    def test_firmware_echo_preserves_scale_constraints_and_fractional_position(self):
        fields = parse_eye_response("EYE:STATE:minScale=0.6,maxScale=1.2,x=1.5,y=-2.5,blinkMs=0")["fields"]
        self.assertEqual(fields["minScale"], 0.6)
        self.assertEqual(fields["maxScale"], 1.2)
        self.assertEqual(fields["lookX"], 1.5)
        self.assertEqual(fields["lookY"], -2.5)
    def test_validate_documented_commands_and_normalizes_values(self):
        self.assertEqual(
            validate_eye_command("eyeconfig:color=00e5ff"),
            "eyeconfig:color=00E5FF",
        )
        self.assertEqual(validate_eye_command("eyeconfig:brightness=0.850"), "eyeconfig:brightness=0.85")
        self.assertEqual(
            validate_eye_command("eyeaction:look:x=15,y=-8"),
            "eyeaction:look:x=15,y=-8",
        )
        self.assertEqual(validate_eye_command("eyeaction:heart"), "eyeaction:heart")

    def test_validate_rejects_out_of_range_or_arbitrary_serial_input(self):
        for command in (
            "eyeconfig:brightness=1.2",
            "eyeconfig:autoBlink=0",
            "eyeconfig:dots=257",
            "eyeaction:look:x=101,y=0",
            "screen_dialog:secret",
            "eyeconfig:color=00E5FF\nopenchat:1",
        ):
            with self.subTest(command=command):
                with self.assertRaises(EyeConfigError):
                    validate_eye_command(command)

    def test_parse_state_accepts_key_value_and_json_payloads(self):
        event = parse_eye_response(
            "EYE:STATE color=00e5ff ringColor=00CFE8 dotColor=00BCD0 "
            "brightness=0.85 autoBlink=1 ring=0 dots=48 mood=flame x=15 y=-8"
        )
        self.assertEqual(event["kind"], "state")
        self.assertEqual(event["fields"]["color"], "00E5FF")
        self.assertEqual(event["fields"]["autoBlink"], True)
        self.assertEqual(event["fields"]["ring"], False)
        self.assertEqual(event["fields"]["lookX"], 15)
        self.assertEqual(event["fields"]["lookY"], -8)
        self.assertEqual(event["fields"]["mood"], "flame")

        json_event = parse_eye_response('EYE:STATE:{"scale":1.25,"glow":22,"mood":"heart"}')
        self.assertEqual(json_event["fields"], {"scale": 1.25, "glow": 22, "mood": "heart"})

    def test_parse_ack_and_error_preserves_device_message(self):
        self.assertEqual(parse_eye_response("EYE:OK")["kind"], "ok")
        error = parse_eye_response("EYE:ERR:unsupported dots")
        self.assertEqual(error["kind"], "err")
        self.assertEqual(error["message"], "unsupported dots")
        self.assertIsNone(parse_eye_response("EYE:STATEFUL nope"))

    def test_malformed_numeric_state_is_ignored_without_breaking_parser(self):
        event = parse_eye_response("EYE:STATE dots=inf mood=heart")
        self.assertEqual(event["fields"], {"mood": "heart"})

    def test_normalize_state_ignores_unknown_keys(self):
        self.assertEqual(
            normalize_eye_state({"color": "#ffffff", "dots": "12", "unknown": "x"}),
            {"color": "FFFFFF", "dots": 12},
        )

    def test_host_config_fills_defaults_and_replays_disabled_blink(self):
        config = normalize_eye_config({"autoBlink": False, "blinkMs": 6000, "mood": "heart"})
        self.assertEqual(config["color"], EYE_CONFIG_DEFAULTS["color"])
        self.assertFalse(config["autoBlink"])
        self.assertEqual(config["blinkMs"], 6000)
        self.assertEqual(config["mood"], "heart")
        commands = eye_config_commands(config)
        self.assertIn("eyeconfig:blinkMs=0", commands)
        self.assertIn("eyeconfig:mood=heart", commands)
        self.assertIn("eyeaction:look:x=0,y=0", commands)

    def test_startup_replay_encodes_every_host_owned_field(self):
        self.assertEqual(
            eye_config_commands(EYE_CONFIG_DEFAULTS),
            [
                "eyeconfig:color=00E5FF",
                "eyeconfig:ringColor=00CFE8",
                "eyeconfig:dotColor=00BCD0",
                "eyeconfig:brightness=0.85",
                "eyeconfig:ringBrightness=0.65",
                "eyeconfig:dotBrightness=0.45",
                "eyeconfig:scale=1",
                "eyeconfig:glow=22",
                "eyeconfig:breathMs=2400",
                "eyeconfig:blinkMs=4500",
                "eyeconfig:ring=1",
                "eyeconfig:dots=48",
                "eyeconfig:mood=dot",
                "eyeaction:look:x=0,y=0",
            ],
        )

    def test_host_config_rejects_unknown_and_boolean_numeric_values(self):
        with self.assertRaises(EyeConfigError):
            normalize_eye_config({"unexpected": True})
        with self.assertRaises(EyeConfigError):
            normalize_eye_config({"brightness": True})

    def test_missing_host_config_section_uses_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text("serial: {}\n", encoding="utf-8")
            self.assertEqual(load_saved_eye_config(config_path), EYE_CONFIG_DEFAULTS)
        with self.subTest("missing file remains backward compatible"):
            self.assertEqual(load_saved_eye_config("does-not-exist.yaml"), EYE_CONFIG_DEFAULTS)

    def test_encode_request_validates_before_publishing_contract(self):
        body = encode_eye_request("request-1", "eyeconfig:ring=1", "ack")
        self.assertIn('"request_id":"request-1"', body)
        self.assertIn('"command":"eyeconfig:ring=1"', body)
        self.assertIn('"expected":"ack"', body)


class EyeConfigRuntimeTests(unittest.TestCase):
    def client(self):
        client = object.__new__(EyeConfigRpcClient)
        client._lock = threading.RLock()
        client._status = {}
        client._speaking = False
        return client

    def test_serial_owner_speaking_status_is_authoritative(self):
        client = self.client()
        client._on_status(SimpleNamespace(data='{"speaking":false}'))
        self.assertFalse(client.status()["speaking"])
        client._on_status(SimpleNamespace(data='{"speaking":true,"event":{"kind":"state"}}'))
        self.assertTrue(client.status()["speaking"])
        client._on_status(SimpleNamespace(data='{"speaking":false}'))
        self.assertFalse(client.status()["speaking"])
        self.assertEqual(client.status()["event"]["kind"], "state")

    def test_query_via_command_waits_for_state_not_ack(self):
        client = self.client()
        client._call = MagicMock(return_value={"ok": True})
        client.command("eyeconfig:query")
        client._call.assert_called_once_with("eyeconfig:query", "state")

    def test_executor_failure_wakes_pending_requests_without_retry(self):
        client = self.client()
        client._closed = threading.Event()
        client._rclpy = MagicMock()
        client._rclpy.ok.return_value = True
        client._executor = MagicMock()
        client._executor.spin_once.side_effect = RuntimeError("executor failed")
        client._node = MagicMock()
        wake, result = threading.Event(), {}
        client._pending = {"request": (wake, result)}
        client._spin()
        self.assertTrue(wake.is_set())
        self.assertFalse(result["ok"])
        self.assertIn("接收线程异常", result["error"])
        client._executor.spin_once.assert_called_once()

    def test_one_client_closing_does_not_shutdown_other_client_context(self):
        from services.hardware import esp32_netcfg_rpc as transport

        clients = [self.client(), self.client()]
        runtime = MagicMock()
        for client in clients:
            client._closed = threading.Event()
            client._pending = {}
            client._context_registered = True
            client._rclpy = runtime
        with patch.object(transport, "_context_users", 2), patch.object(transport, "_context_owned", True):
            clients[0].close()
            runtime.shutdown.assert_not_called()
            clients[1].close()
            runtime.shutdown.assert_called_once()
            clients[1].close()
            runtime.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
