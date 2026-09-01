import tempfile
import unittest
from pathlib import Path

from services.servo_motion_config import load_neck_kinematics


class NeckKinematicsTests(unittest.TestCase):
    def _load(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        config_path = Path(temp_dir.name) / "config.yaml"
        config_path.write_text(
            """servos:
  - id: 5
    name: neck_top
    limit_1: 5000
    limit_2: 6000
    init: 5000
  - id: 6
    name: neck_bottom
    limit_1: 2000
    limit_2: 4800
    init: 3000
""",
            encoding="utf-8",
        )
        return load_neck_kinematics(config_path)

    def test_neutral_uses_configured_initial_positions(self):
        self.assertEqual(
            self._load().targets(0.0),
            {"neck_top": 5000, "neck_bottom": 3000},
        )

    def test_looking_up_extends_only_the_lower_neck(self):
        kinematics = self._load()
        self.assertEqual(
            kinematics.targets(1.0),
            {"neck_top": 5000, "neck_bottom": 4800},
        )
        self.assertEqual(
            kinematics.targets(0.5),
            {"neck_top": 5000, "neck_bottom": 3900},
        )

    def test_looking_down_increases_top_and_lowers_bottom(self):
        kinematics = self._load()
        self.assertEqual(
            kinematics.targets(-1.0),
            {"neck_top": 6000, "neck_bottom": 2000},
        )
        self.assertEqual(
            kinematics.targets(-0.5),
            {"neck_top": 5500, "neck_bottom": 2500},
        )

    def test_pitch_is_clamped_to_normalized_range(self):
        kinematics = self._load()
        self.assertEqual(kinematics.targets(2.0), kinematics.targets(1.0))
        self.assertEqual(kinematics.targets(-2.0), kinematics.targets(-1.0))

    def test_missing_neck_servo_fails_closed(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        config_path = Path(temp_dir.name) / "config.yaml"
        config_path.write_text("servos: []\n", encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "neck_top"):
            load_neck_kinematics(config_path)


if __name__ == "__main__":
    unittest.main()
