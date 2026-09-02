import unittest

from services.motion_arbiter import MotionArbiter, STOP_COMMAND


FORWARD = {
    "left": {"action": 1, "throttle": 40},
    "right": {"action": 1, "throttle": 40},
}
REVERSE = {
    "left": {"action": 2, "throttle": 30},
    "right": {"action": 2, "throttle": 30},
}


class MotionArbiterTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.arbiter = MotionArbiter(timeout_sec=0.3, clock=lambda: self.now)

    def test_priority_is_joystick_then_tracking_then_autonomy(self):
        self.assertTrue(self.arbiter.update("autonomy", FORWARD))
        self.assertEqual(self.arbiter.select(), ("autonomy", FORWARD))

        self.assertTrue(self.arbiter.update("tracking", REVERSE))
        self.assertEqual(self.arbiter.select(), ("tracking", REVERSE))

        self.assertTrue(self.arbiter.update("joystick", STOP_COMMAND))
        self.assertEqual(self.arbiter.select(), ("joystick", STOP_COMMAND))

    def test_stale_commands_fail_safe_to_stop(self):
        self.arbiter.update("joystick", FORWARD)
        self.now += 0.31

        self.assertEqual(self.arbiter.select(), ("failsafe", STOP_COMMAND))

    def test_fresh_lower_priority_command_takes_over_after_manual_timeout(self):
        self.arbiter.update("joystick", FORWARD)
        self.now += 0.2
        self.arbiter.update("tracking", REVERSE)
        self.now += 0.11

        self.assertEqual(self.arbiter.select(), ("tracking", REVERSE))

    def test_selection_reports_exact_expiry_deadline(self):
        self.arbiter.update("autonomy", FORWARD)

        source, command, deadline = self.arbiter.select_with_deadline()

        self.assertEqual((source, command), ("autonomy", FORWARD))
        self.assertAlmostEqual(deadline, 10.3)

        self.now += 0.31
        self.assertEqual(
            self.arbiter.select_with_deadline(),
            ("failsafe", STOP_COMMAND, None),
        )

    def test_invalid_commands_are_rejected(self):
        invalid = {
            "left": {"action": 1, "throttle": 101},
            "right": {"action": 1, "throttle": 20},
        }
        self.assertFalse(self.arbiter.update("autonomy", invalid))
        self.assertFalse(self.arbiter.update("unknown", FORWARD))
        self.assertEqual(self.arbiter.select(), ("failsafe", STOP_COMMAND))


if __name__ == "__main__":
    unittest.main()
