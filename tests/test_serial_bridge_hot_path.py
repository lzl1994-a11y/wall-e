import threading
import unittest
from unittest.mock import Mock, patch

from services.serial_bridge import SerialBridge
from services.esp32_netcfg import (
    network_settings_match_status,
    validate_network_payload,
)


class SerialBridgeHotPathTests(unittest.TestCase):
    def make_bridge(self):
        bridge = SerialBridge.__new__(SerialBridge)
        bridge.ser = Mock(is_open=True, port="/dev/ttyACM0")
        bridge.broker = Mock(config_path="/tmp/wali-config.yaml")
        bridge._next_selection_check_at = 0.0
        bridge._next_reconnect_at = float("inf")
        bridge._selection_config_mtime_ns = 10
        bridge._io_lock = threading.RLock()
        bridge.is_screen_awake = True
        bridge.last_send_time = 1.0
        bridge.timeout_seconds = 30.0
        return bridge

    @patch("services.serial_bridge.serial_ports_for_role")
    def test_unchanged_config_does_not_scan_usb_devices(self, resolve_ports):
        bridge = self.make_bridge()
        bridge._config_mtime_ns = Mock(return_value=10)

        self.assertTrue(bridge._ensure_connected())

        resolve_ports.assert_not_called()
        bridge.ser.close.assert_not_called()

    @patch(
        "services.serial_bridge.serial_ports_for_role",
        return_value=(["/dev/ttyACM1"], True),
    )
    def test_changed_selection_revalidates_open_port(self, resolve_ports):
        bridge = self.make_bridge()
        serial_port = bridge.ser
        bridge._config_mtime_ns = Mock(return_value=11)

        self.assertFalse(bridge._ensure_connected())

        resolve_ports.assert_called_once()
        serial_port.close.assert_called_once()

    def test_nonblocking_motion_send_drops_while_serial_is_owned(self):
        bridge = self.make_bridge()
        acquired = threading.Event()
        release = threading.Event()

        def owner():
            with bridge._io_lock:
                acquired.set()
                release.wait(1.0)

        thread = threading.Thread(target=owner)
        thread.start()
        acquired.wait(1.0)
        try:
            self.assertFalse(bridge.send_raw("pca9685:1\n", block=False))
            bridge.ser.write.assert_not_called()
        finally:
            release.set()
            thread.join(1.0)


class StartupNetworkSyncTests(unittest.TestCase):
    def test_matching_query_status_skips_reapply(self):
        settings = validate_network_payload(
            {
                "wifi": [
                    {"ssid": "shop", "password": "secret"},
                    {"ssid": "backup", "password": ""},
                    {"ssid": "", "password": ""},
                ],
                "host": "192.168.0.6",
                "port": 9000,
            }
        )
        status = {
            "wifi": [{"ssid": "shop"}, {"ssid": "backup"}, {"ssid": ""}],
            "host": "192.168.0.6",
            "port": 9000,
            "apply_running": False,
        }

        self.assertTrue(network_settings_match_status(settings, status))
        status["host"] = "192.168.0.7"
        self.assertFalse(network_settings_match_status(settings, status))


if __name__ == "__main__":
    unittest.main()
