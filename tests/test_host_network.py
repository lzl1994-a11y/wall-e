import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

from services.hardware.esp32_netcfg import Esp32NetworkConfigurator, NetworkConfigError
from services.hardware.host_network import (
    AP_ADDRESS, HostNetworkManager, NetworkManagerBackend, split_nmcli, validate_wifi,
)
from tests.test_esp32_netcfg import FakeSerial
from services.hardware.serial_bridge import _RoutedNetcfgStream


class HostNetworkTests(unittest.TestCase):
    def setUp(self):
        self.store = MagicMock()
        self.config = {"esp32_network": {"wifi": [{"ssid": "old", "password": "old-secret"}]},
                       "tft_preview": {"port": 9000}}
        self.store.load.side_effect = lambda: copy.deepcopy(self.config)
        self.backend = MagicMock()
        self.backend.connect.return_value = {"ip": "192.168.1.20", "connected": True, "hotspot": False}
        self.backend.hotspot.return_value = {"ssid": "WALL-E-1234", "password": "ap-secret", "ip": AP_ADDRESS}
        self.rpc = MagicMock()
        self.manager = HostNetworkManager(self.store, lambda: self.rpc, 8080,
                                          backend=self.backend, switch_delay=0)

    def tearDown(self):
        self.manager.close()

    def finish(self):
        self.manager._worker.join(2)
        self.assertFalse(self.manager._worker.is_alive())

    def connect(self):
        self.manager.connect({"ssid": "new", "password": "new-secret"})
        self.finish()

    def test_connect_persists_before_esp_sync_and_keeps_old_network(self):
        def sync(payload):
            self.store.save_patch.assert_called_once()
            self.assertEqual(payload["host"], "192.168.1.20")
            self.assertEqual([item["ssid"] for item in payload["wifi"]], ["new", "old", ""])
        self.rpc.save_and_apply.side_effect = sync
        self.connect()
        self.assertEqual(self.manager.snapshot()["state"], "ready")
        self.assertNotIn("secret", repr(self.manager.snapshot()))
        self.assertEqual([call.args[0]["state"] for call in self.rpc.host_screen.call_args_list],
                         ["connecting", "connected", "done"])

    def test_invalid_wifi_cannot_switch_network(self):
        for ssid, password in [("", "12345678"), ("new\n", "12345678"),
                               ("new", "abc\n12345"), ("中" * 11, "12345678"), ("new", "short")]:
            with self.assertRaises(NetworkConfigError):
                self.manager.connect({"ssid": ssid, "password": password})
        self.backend.connect.assert_not_called()
        self.assertEqual(validate_wifi("open", "").password, "")

    def test_failure_restores_hotspot_without_overwriting_saved_wifi(self):
        self.backend.connect.side_effect = NetworkConfigError("new-secret")
        self.connect()
        self.assertEqual(self.manager.snapshot()["state"], "setup")
        self.assertNotIn("new-secret", repr(self.manager.snapshot()))
        self.store.save_patch.assert_not_called()
        self.backend.hotspot.assert_called_once()
        self.assertEqual(self.rpc.host_screen.call_args.args[0]["state"], "setup")

    def test_esp_failure_keeps_host_wifi_and_can_retry(self):
        self.rpc.save_and_apply.side_effect = NetworkConfigError("serial offline")
        self.connect()
        self.assertEqual(self.manager.snapshot()["state"], "partial")
        self.backend.hotspot.assert_not_called()
        self.rpc.save_and_apply.side_effect = None
        self.manager.retry_sync()
        self.finish()
        self.assertEqual(self.manager.snapshot()["state"], "ready")
        self.backend.connect.assert_called_once()

    def test_concurrent_submission_is_rejected(self):
        import threading
        gate = threading.Event()
        self.backend.connect.side_effect = lambda *_: (gate.wait(2), {"ip": "192.168.1.20"})[1]
        self.manager.connect({"ssid": "new", "password": "new-secret"})
        try:
            with self.assertRaises(NetworkConfigError):
                self.manager.enter_setup()
        finally:
            gate.set()
        self.finish()

    def test_screen_protocol_matches_ack_and_rejects_injection(self):
        client = Esp32NetworkConfigurator()
        client._next_seq = lambda: 42
        serial = FakeSerial(b"NETCFG:RESULT:41|HOSTSCREEN|0|0\nNETCFG:RESULT:42|HOSTSCREEN|0|0\n")
        payload = {"state": "setup", "ip": AP_ADDRESS, "port": 8080,
                   "ssid": "WALL-E-1234", "password": "ap-secret"}
        result = client.host_screen(payload, stream=serial)
        self.assertEqual(result["seq"], 42)
        self.assertEqual(serial.writes[0], b"hostnet:42|1|setup|10.42.0.1|8080|WALL-E-1234|ap-secret\r\n")
        for field, value in [("ip", "invalid"), ("password", "injected|value"), ("port", 0)]:
            with self.assertRaises(NetworkConfigError):
                client.host_screen({**payload, field: value}, stream=serial)

    def test_nmcli_escaping_and_keyfile_credentials(self):
        self.assertEqual(split_nmcli(r"shop\:wifi:80:WPA2"), ["shop:wifi", "80", "WPA2"])
        with tempfile.TemporaryDirectory() as directory:
            commands = []
            def run(command, **kwargs):
                commands.append(command)
                output = "wlan0:wifi:disconnected:--" if "DEVICE,TYPE,STATE,CONNECTION" in command else ""
                return SimpleNamespace(returncode=0, stdout=output)
            backend = NetworkManagerBackend(runner=run, profile_dir=directory)
            backend._profile("shop:wifi", "pass word")
            profile = next(Path(directory).glob("*.nmconnection")).read_text(encoding="utf-8")
            self.assertIn("psk=pass\\sword", profile)
            self.assertIn("ssid=115;104;111;112;58;119;105;102;105;", profile)
            self.assertNotIn("pass word", repr(commands))

    def test_host_screen_sequence_is_routed_by_shared_serial_stream(self):
        bridge = MagicMock()
        stream = _RoutedNetcfgStream(bridge)
        stream.write(b"hostnet:42|1|done|192.168.0.5|8080||\r\n")
        bridge._register_netcfg_sequence.assert_called_once_with(42)
        bridge.send_raw.assert_called_once_with("hostnet:42|1|done|192.168.0.5|8080||\r\n", wake_screen=False)
        stream.close()


if __name__ == "__main__":
    unittest.main()
