import unittest
from types import SimpleNamespace
from unittest.mock import patch

from services.serial_broker import SerialBroker


class FakeSerial:
    def __init__(self, port, responses):
        self.port = port
        self.responses = responses
        self.dtr = False
        self.rts = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def reset_input_buffer(self):
        pass

    def write(self, _data):
        pass

    def readline(self):
        return self.responses.get(self.port, "").encode("utf-8")


class SerialBrokerTests(unittest.TestCase):
    @patch("services.serial_broker.time.sleep")
    @patch("services.serial_broker.serial_ports_for_role", return_value=([], True))
    @patch("services.serial_broker.serial.tools.list_ports.comports")
    def test_stale_screen_selector_falls_back_to_handshake(
        self, comports, _selected_ports, _sleep
    ):
        comports.return_value = [SimpleNamespace(device="/dev/ttyACM0")]
        responses = {"/dev/ttyACM0": "IAM:WALL_E_TFT\n"}

        with patch(
            "services.serial_broker.serial.Serial",
            side_effect=lambda port, *_args, **_kwargs: FakeSerial(port, responses),
        ):
            broker = SerialBroker(config_path="/tmp/config.yaml")
            result = broker.scan_and_identify(
                usb_role="screen_motion",
                fallback_device_name="WALL_E_TFT",
            )

        self.assertEqual(result["WALL_E_TFT"], "/dev/ttyACM0")

    @patch("services.serial_broker.time.sleep")
    @patch("services.serial_broker.serial_ports_for_role", return_value=([], True))
    @patch("services.serial_broker.serial.tools.list_ports.comports")
    def test_strict_role_does_not_probe_unselected_serial_ports(
        self, comports, _selected_ports, _sleep
    ):
        comports.return_value = [SimpleNamespace(device="/dev/ttyACM0")]

        with patch("services.serial_broker.serial.Serial") as serial_factory:
            broker = SerialBroker(config_path="/tmp/config.yaml")
            result = broker.scan_and_identify(usb_role="voice")

        self.assertEqual(result, {})
        serial_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
