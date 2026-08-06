import tempfile
import unittest
from pathlib import Path

from services.remote_control_config import load_remote_control_config


class RemoteControlConfigTests(unittest.TestCase):
    def test_missing_section_uses_existing_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.yaml"
            path.write_text("launch:\n  serial: true\n", encoding="utf-8")
            self.assertEqual(
                load_remote_control_config(path),
                {"servo_step_size": 40.0, "update_rate_hz": 20.0},
            )

    def test_loads_configured_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.yaml"
            path.write_text(
                "remote_control:\n  servo_step_size: 55.5\n  update_rate_hz: 30\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_remote_control_config(path),
                {"servo_step_size": 55.5, "update_rate_hz": 30.0},
            )

    def test_bad_values_fall_back_to_existing_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.yaml"
            path.write_text(
                "remote_control:\n  servo_step_size: -1\n  update_rate_hz: fast\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_remote_control_config(path),
                {"servo_step_size": 40.0, "update_rate_hz": 20.0},
            )


if __name__ == "__main__":
    unittest.main()
