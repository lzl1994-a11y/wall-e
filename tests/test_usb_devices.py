import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services import usb_devices


class FakeSoundDevice:
    @staticmethod
    def query_devices():
        return [
            {"name": "Built-in Audio", "max_input_channels": 2, "max_output_channels": 2},
            {"name": "Wali Voice USB (hw:3,0)", "max_input_channels": 2, "max_output_channels": 2},
        ]


class UsbDeviceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="wali-usb-")
        self.config_path = Path(self.temp_dir.name) / "config.yaml"

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_selector(self, role, selector):
        self.config_path.write_text(
            yaml.safe_dump({"usb_devices": {role: selector}}, sort_keys=False),
            encoding="utf-8",
        )

    @patch("services.usb_devices.list_usb_devices")
    def test_selected_serial_role_resolves_current_port(self, list_devices):
        selector = {"vendor_id": "303a", "product_id": "1001", "serial_number": "screen"}
        self.write_selector("screen_motion", selector)
        list_devices.return_value = [
            {
                **selector,
                "id": "303a:1001:screen",
                "port_path": "1-2",
                "interfaces": {"serial": ["/dev/ttyACM7"], "video": [], "audio_cards": []},
            }
        ]
        ports, configured = usb_devices.serial_ports_for_role("screen_motion", self.config_path)
        self.assertTrue(configured)
        self.assertEqual(ports, ["/dev/ttyACM7"])

    @patch("services.usb_devices.list_usb_devices", return_value=[])
    def test_selected_device_can_be_offline(self, _list_devices):
        self.write_selector("camera", {"vendor_id": "1234", "product_id": "5678", "port_path": "1-3"})
        device, configured = usb_devices.find_selected_usb_device("camera", self.config_path)
        self.assertTrue(configured)
        self.assertIsNone(device)

    @patch("services.usb_devices.list_usb_devices")
    def test_audio_role_maps_usb_card_to_portaudio_device(self, list_devices):
        selector = {"vendor_id": "1234", "product_id": "5678", "serial_number": "voice"}
        self.write_selector("voice", selector)
        list_devices.return_value = [
            {
                **selector,
                "id": "1234:5678:voice",
                "manufacturer": "Wali",
                "product": "Wali Voice USB",
                "port_path": "1-4",
                "interfaces": {"serial": ["/dev/ttyACM2"], "video": [], "audio_cards": [3]},
            }
        ]
        resolution = usb_devices.resolve_audio_device(
            "input", self.config_path, sounddevice_module=FakeSoundDevice
        )
        self.assertTrue(resolution.configured)
        self.assertTrue(resolution.available)
        self.assertEqual(resolution.index, 1)

    def test_linux_sysfs_groups_composite_usb_interfaces(self):
        sys_root = Path(self.temp_dir.name) / "sys"
        usb_root = sys_root / "bus" / "usb" / "devices"
        usb_device = usb_root / "1-2"
        usb_device.mkdir(parents=True)
        (usb_device / "idVendor").write_text("303a\n", encoding="utf-8")
        (usb_device / "idProduct").write_text("1001\n", encoding="utf-8")
        (usb_device / "serial").write_text("wali-composite\n", encoding="utf-8")
        (usb_device / "product").write_text("WALI Composite\n", encoding="utf-8")

        tty_target = usb_device / "interface0" / "tty" / "ttyACM5"
        video_target = usb_device / "interface1" / "video4linux" / "video7"
        audio_target = usb_device / "interface2" / "sound" / "card3"
        for target in (tty_target, video_target, audio_target):
            target.mkdir(parents=True)

        class_members = {
            "tty": [tty_target],
            "video4linux": [video_target],
            "sound": [audio_target],
        }

        def fake_class_members(class_name, pattern):
            return [
                path for path in class_members[class_name]
                if pattern.fullmatch(path.name)
            ]

        with patch.object(usb_devices, "USB_SYSFS", usb_root), patch.object(
            usb_devices, "_class_members", side_effect=fake_class_members
        ):
            devices = usb_devices._linux_usb_devices()

        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0]["interfaces"]["serial"], ["/dev/ttyACM5"])
        self.assertEqual(devices[0]["interfaces"]["video"], ["/dev/video7"])
        self.assertEqual(devices[0]["interfaces"]["audio_cards"], [3])

    def test_missing_role_uses_code_default(self):
        self.config_path.write_text("usb_devices: {}\n", encoding="utf-8")
        resolution = usb_devices.resolve_audio_device(
            "output", self.config_path, sounddevice_module=FakeSoundDevice
        )
        self.assertFalse(resolution.configured)
        self.assertTrue(resolution.available)
        self.assertIsNone(resolution.index)


if __name__ == "__main__":
    unittest.main()
