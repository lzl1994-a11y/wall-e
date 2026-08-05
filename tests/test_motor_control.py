import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.motor_control import (
    apply_direction_inversion,
    mix_differential_drive,
    motor_inversion_flags,
)


class DifferentialDriveTests(unittest.TestCase):
    def test_cardinal_joystick_directions(self):
        self.assertEqual(
            mix_differential_drive(1, 0),
            {"left": {"action": 1, "throttle": 100}, "right": {"action": 1, "throttle": 100}},
        )
        self.assertEqual(
            mix_differential_drive(-1, 0),
            {"left": {"action": 2, "throttle": 100}, "right": {"action": 2, "throttle": 100}},
        )
        self.assertEqual(
            mix_differential_drive(0, -1),
            {"left": {"action": 2, "throttle": 100}, "right": {"action": 1, "throttle": 100}},
        )
        self.assertEqual(
            mix_differential_drive(0, 1),
            {"left": {"action": 1, "throttle": 100}, "right": {"action": 2, "throttle": 100}},
        )

    def test_direction_inversion_swaps_forward_and_reverse_only(self):
        self.assertEqual(apply_direction_inversion(1, True), 2)
        self.assertEqual(apply_direction_inversion(2, True), 1)
        self.assertEqual(apply_direction_inversion(0, True), 0)
        self.assertEqual(apply_direction_inversion(1, False), 1)

    def test_motor_inversion_flags_use_track_names(self):
        flags = motor_inversion_flags([
            {"name": "track_r", "invert_direction": True},
            {"name": "track_l", "invert_direction": False},
        ])

        self.assertEqual(flags, {"left": False, "right": True})


if __name__ == "__main__":
    unittest.main()
