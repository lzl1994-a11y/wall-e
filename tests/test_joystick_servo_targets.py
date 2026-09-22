import copy
import unittest

from services.motion.joystick_servo_targets import (
    AXIS_L2,
    AXIS_R2,
    AXIS_RX,
    AXIS_RY,
    compute_joystick_servo_targets,
)
from services.motion.servo_motion_config import NeckKinematics, ServoCalibration


def _make_kinematics():
    return NeckKinematics(
        top=ServoCalibration(initial=5000, low=5000, high=6000),
        bottom=ServoCalibration(initial=3000, low=2000, high=4800),
    )


class JoystickServoTargetsTests(unittest.TestCase):
    def test_all_axes_centered_and_no_active_timers(self):
        kinematics = _make_kinematics()
        axes = {
            AXIS_RX: 0.0,
            AXIS_RY: 0.0,
            AXIS_L2: 0.0,
            AXIS_R2: 0.0,
        }
        auto_timers = {
            "arm_l": 0.0,
            "arm_r": 0.0,
            "eyebrow_l": 0.0,
            "eyebrow_r": 0.0,
        }
        now = 10.0

        targets = compute_joystick_servo_targets(axes, auto_timers, now, kinematics)

        expected = {
            "head_yaw": 5000,
            "neck_top": 5000,
            "neck_bottom": 3000,
            "eye_l": 7500,
            "eye_r": 2000,
            "arm_l": 2000,
            "arm_r": 8000,
            "eyebrow_l": 8000,
            "eyebrow_r": 1920,
        }
        self.assertEqual(targets, expected)
        # Verify exact key ordering
        self.assertEqual(
            list(targets.keys()),
            [
                "head_yaw",
                "neck_top",
                "neck_bottom",
                "eye_l",
                "eye_r",
                "arm_l",
                "arm_r",
                "eyebrow_l",
                "eyebrow_r",
            ],
        )

    def test_axes_endpoints_and_extreme_values(self):
        kinematics = _make_kinematics()
        auto_timers = {"arm_l": 0.0, "arm_r": 0.0, "eyebrow_l": 0.0, "eyebrow_r": 0.0}
        now = 1.0

        # Positive endpoints: rx=1.0, ry=1.0, l2=1.0, r2=1.0
        axes_pos = {AXIS_RX: 1.0, AXIS_RY: 1.0, AXIS_L2: 1.0, AXIS_R2: 1.0}
        targets_pos = compute_joystick_servo_targets(axes_pos, auto_timers, now, kinematics)
        self.assertEqual(targets_pos["head_yaw"], 2400)
        self.assertEqual(targets_pos["neck_top"], 5000)
        self.assertEqual(targets_pos["neck_bottom"], 4800)
        self.assertEqual(targets_pos["eye_l"], 5000)
        self.assertEqual(targets_pos["eye_r"], 4000)

        # Negative endpoints: rx=-1.0, ry=-1.0, l2=0.0, r2=0.0
        axes_neg = {AXIS_RX: -1.0, AXIS_RY: -1.0, AXIS_L2: 0.0, AXIS_R2: 0.0}
        targets_neg = compute_joystick_servo_targets(axes_neg, auto_timers, now, kinematics)
        self.assertEqual(targets_neg["head_yaw"], 7600)
        self.assertEqual(targets_neg["neck_top"], 6000)
        self.assertEqual(targets_neg["neck_bottom"], 2000)
        self.assertEqual(targets_neg["eye_l"], 7500)
        self.assertEqual(targets_neg["eye_r"], 2000)

    def test_neck_coupling_delegates_to_neck_kinematics(self):
        class MockNeckKinematics:
            def __init__(self):
                self.received_pitch = None

            def targets(self, pitch: float):
                self.received_pitch = pitch
                return {"neck_top": 5555, "neck_bottom": 3333}

        mock_kinematics = MockNeckKinematics()
        axes = {AXIS_RX: 0.0, AXIS_RY: 0.42, AXIS_L2: 0.0, AXIS_R2: 0.0}
        auto_timers = {"arm_l": 0.0, "arm_r": 0.0, "eyebrow_l": 0.0, "eyebrow_r": 0.0}
        targets = compute_joystick_servo_targets(axes, auto_timers, 0.0, mock_kinematics)

        self.assertEqual(mock_kinematics.received_pitch, 0.42)
        self.assertEqual(targets["neck_top"], 5555)
        self.assertEqual(targets["neck_bottom"], 3333)

    def test_auto_reset_timer_strict_deadline_boundary(self):
        kinematics = _make_kinematics()
        axes = {AXIS_RX: 0.0, AXIS_RY: 0.0, AXIS_L2: 0.0, AXIS_R2: 0.0}
        deadline = 100.0
        auto_timers = {
            "arm_l": deadline,
            "arm_r": deadline,
            "eyebrow_l": deadline,
            "eyebrow_r": deadline,
        }

        # Case 1: before deadline (now < deadline -> True)
        before = compute_joystick_servo_targets(axes, auto_timers, 99.9, kinematics)
        self.assertEqual(before["arm_l"], 6000)
        self.assertEqual(before["arm_r"], 4000)
        self.assertEqual(before["eyebrow_l"], 5700)
        self.assertEqual(before["eyebrow_r"], 4200)

        # Case 2: exactly at deadline (now < deadline -> False)
        at_deadline = compute_joystick_servo_targets(axes, auto_timers, 100.0, kinematics)
        self.assertEqual(at_deadline["arm_l"], 2000)
        self.assertEqual(at_deadline["arm_r"], 8000)
        self.assertEqual(at_deadline["eyebrow_l"], 8000)
        self.assertEqual(at_deadline["eyebrow_r"], 1920)

        # Case 3: after deadline (now < deadline -> False)
        after = compute_joystick_servo_targets(axes, auto_timers, 100.1, kinematics)
        self.assertEqual(after["arm_l"], 2000)
        self.assertEqual(after["arm_r"], 8000)
        self.assertEqual(after["eyebrow_l"], 8000)
        self.assertEqual(after["eyebrow_r"], 1920)

    def test_inputs_are_not_mutated(self):
        kinematics = _make_kinematics()
        axes = {AXIS_RX: 0.5, AXIS_RY: -0.5, AXIS_L2: 0.2, AXIS_R2: 0.8}
        auto_timers = {
            "arm_l": 5.0,
            "arm_r": 15.0,
            "eyebrow_l": 5.0,
            "eyebrow_r": 15.0,
        }
        axes_copy = copy.deepcopy(axes)
        timers_copy = copy.deepcopy(auto_timers)

        compute_joystick_servo_targets(axes, auto_timers, 10.0, kinematics)

        self.assertEqual(axes, axes_copy)
        self.assertEqual(auto_timers, timers_copy)

    def test_missing_axis_does_not_silently_publish_a_default_target(self):
        axes = {AXIS_RY: 0.0, AXIS_L2: 0.0, AXIS_R2: 0.0}
        timers = {"arm_l": 0.0, "arm_r": 0.0, "eyebrow_l": 0.0, "eyebrow_r": 0.0}

        with self.assertRaises(KeyError):
            compute_joystick_servo_targets(axes, timers, 1.0, _make_kinematics())

    def test_missing_timer_does_not_silently_publish_a_default_target(self):
        axes = {AXIS_RX: 0.0, AXIS_RY: 0.0, AXIS_L2: 0.0, AXIS_R2: 0.0}
        timers = {"arm_r": 0.0, "eyebrow_l": 0.0, "eyebrow_r": 0.0}

        with self.assertRaises(KeyError):
            compute_joystick_servo_targets(axes, timers, 1.0, _make_kinematics())


if __name__ == "__main__":
    unittest.main()
