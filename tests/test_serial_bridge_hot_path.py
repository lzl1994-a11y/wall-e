import threading
import unittest
from collections import defaultdict, deque
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
        bridge._reconnect_delay_sec = 1.0
        bridge._selection_config_mtime_ns = 10
        bridge._io_lock = threading.RLock()
        bridge._connection_lock = threading.RLock()
        bridge._response_condition = threading.Condition()
        bridge._netcfg_responses = defaultdict(deque)
        bridge._registered_netcfg_sequences = set()
        bridge._reader_stop = threading.Event()
        bridge.is_screen_awake = True
        bridge.last_send_time = 1.0
        bridge.timeout_seconds = 30.0
        return bridge

    @patch("services.serial_bridge.time.monotonic", side_effect=[100.0, 100.5, 101.0])
    def test_failed_reconnect_uses_exponential_backoff(self, _monotonic):
        bridge = self.make_bridge()
        bridge.ser = None
        bridge._next_reconnect_at = 0.0
        bridge._connect = Mock(return_value=False)

        self.assertFalse(bridge._ensure_connected())
        self.assertEqual(bridge._next_reconnect_at, 101.0)
        self.assertEqual(bridge._reconnect_delay_sec, 2.0)

        self.assertFalse(bridge._ensure_connected())
        bridge._connect.assert_called_once()

        self.assertFalse(bridge._ensure_connected())
        self.assertEqual(bridge._next_reconnect_at, 103.0)
        self.assertEqual(bridge._reconnect_delay_sec, 4.0)
        self.assertEqual(bridge._connect.call_count, 2)

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

    def test_persistent_surface_motion_does_not_inject_chat_wake(self):
        bridge = self.make_bridge()
        bridge._ensure_connected = Mock(return_value=True)
        bridge.is_screen_awake = False

        self.assertTrue(bridge.send_raw("pca9685:1\n", wake_screen=False))

        bridge.ser.write.assert_called_once_with(b"pca9685:1\n")
        self.assertFalse(bridge.is_screen_awake)

    def test_normal_dialog_can_wake_after_persistent_surface(self):
        bridge = self.make_bridge()
        bridge._ensure_connected = Mock(return_value=True)
        bridge.is_screen_awake = False
        bridge.send_raw("eyeaction:talk\n", wake_screen=False)

        bridge.send_raw("ai:hello\n")

        self.assertEqual(bridge.ser.write.call_args.args[0], b"openchat:1\nai:hello\n")

    def test_routed_netcfg_wait_does_not_hold_normal_write_lock(self):
        bridge = self.make_bridge()
        bridge._ensure_connected = Mock(return_value=True)
        transaction_started = threading.Event()
        release_response = threading.Event()

        def operation(stream):
            stream.write(b"netcfg:query:42|2\r\n")
            transaction_started.set()
            release_response.wait(1.0)
            return stream.read(1)

        worker = threading.Thread(target=lambda: bridge.run_exclusive(operation))
        worker.start()
        self.assertTrue(transaction_started.wait(1.0))
        self.assertTrue(bridge.send_raw("eyeaction:talk\n", wake_screen=False))
        with bridge._response_condition:
            bridge._netcfg_responses[42].append("NETCFG:STATUS:42|2|0|255|||||0")
            bridge._response_condition.notify_all()
        release_response.set()
        worker.join(1.0)
        self.assertFalse(worker.is_alive())


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
