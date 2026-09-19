import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.motor_control import (
    apply_direction_inversion,
    compute_joystick_motor_decision,
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


class JoystickMotorDecisionTests(unittest.TestCase):
    def test_initial_idle_does_not_publish_or_stop(self):
        decision = compute_joystick_motor_decision(0.0, 0.0, was_moving=False)
        self.assertIsNone(decision.command)
        self.assertFalse(decision.should_stop)
        self.assertFalse(decision.is_moving)

    def test_first_motion_emits_command_and_sets_moving(self):
        decision = compute_joystick_motor_decision(0.5, 0.0, was_moving=False)
        self.assertIsNotNone(decision.command)
        self.assertFalse(decision.should_stop)
        self.assertTrue(decision.is_moving)
        self.assertEqual(decision.command, mix_differential_drive(0.5, 0.0))

    def test_continuous_motion_emits_command_each_tick_regardless_of_same_value(self):
        d1 = compute_joystick_motor_decision(0.6, 0.2, was_moving=False)
        self.assertIsNotNone(d1.command)
        self.assertTrue(d1.is_moving)

        # Same values on next tick
        d2 = compute_joystick_motor_decision(0.6, 0.2, was_moving=d1.is_moving)
        self.assertIsNotNone(d2.command)
        self.assertEqual(d1.command, d2.command)
        self.assertTrue(d2.is_moving)
        self.assertFalse(d2.should_stop)

    def test_zero_after_motion_stops_once(self):
        d_stop = compute_joystick_motor_decision(0.0, 0.0, was_moving=True)
        self.assertIsNone(d_stop.command)
        self.assertTrue(d_stop.should_stop)
        self.assertFalse(d_stop.is_moving)

    def test_continuous_idle_does_not_repeat_stop(self):
        d_idle = compute_joystick_motor_decision(0.0, 0.0, was_moving=False)
        self.assertIsNone(d_idle.command)
        self.assertFalse(d_idle.should_stop)
        self.assertFalse(d_idle.is_moving)

    def test_forward_and_turn_parameter_order(self):
        # Forward only: left and right tracks both driven forward
        d_fwd = compute_joystick_motor_decision(0.5, 0.0, was_moving=False)
        self.assertEqual(d_fwd.command, mix_differential_drive(0.5, 0.0))
        self.assertEqual(d_fwd.command["left"]["action"], 1)
        self.assertEqual(d_fwd.command["right"]["action"], 1)

        # Turn only (positive turn = right turn): left forward, right reverse
        d_turn = compute_joystick_motor_decision(0.0, 0.5, was_moving=False)
        self.assertEqual(d_turn.command, mix_differential_drive(0.0, 0.5))
        self.assertEqual(d_turn.command["left"]["action"], 1)
        self.assertEqual(d_turn.command["right"]["action"], 2)

        # Distinct forward and turn inputs
        d_combo = compute_joystick_motor_decision(0.7, -0.3, was_moving=False)
        self.assertEqual(d_combo.command, mix_differential_drive(0.7, -0.3))

if __name__ == "__main__":
    unittest.main()
