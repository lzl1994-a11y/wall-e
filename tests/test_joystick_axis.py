import unittest

from services.joystick_axis import normalize_axis_value


class JoystickAxisNormalizationTests(unittest.TestCase):
    def test_trigger_normalization_and_non_zero_start(self):
        # 0 ~ 255 standard trigger
        self.assertEqual(normalize_axis_value(0, 0, 255, is_trigger=True), 0.0)
        self.assertEqual(normalize_axis_value(255, 0, 255, is_trigger=True), 1.0)
        self.assertAlmostEqual(normalize_axis_value(127.5, 0, 255, is_trigger=True), 0.5)

        # Non-zero start calibration: 100 ~ 300
        self.assertEqual(normalize_axis_value(100, 100, 300, is_trigger=True), 0.0)
        self.assertEqual(normalize_axis_value(300, 100, 300, is_trigger=True), 1.0)
        self.assertEqual(normalize_axis_value(200, 100, 300, is_trigger=True), 0.5)

    def test_trigger_out_of_bounds_behavior(self):
        # Below min clamps to 0
        self.assertEqual(normalize_axis_value(-10, 0, 255, is_trigger=True), 0.0)
        self.assertEqual(normalize_axis_value(50, 100, 200, is_trigger=True), 0.0)

        # Above max preserves value > 1.0 without upper clipping
        result = normalize_axis_value(300, 0, 255, is_trigger=True)
        self.assertGreater(result, 1.0)
        self.assertAlmostEqual(result, 300 / 255)

        # Trigger ignores deadzone and invert_y
        self.assertAlmostEqual(
            normalize_axis_value(10, 0, 255, deadzone=0.15, is_trigger=True, invert_y=True),
            10 / 255,
        )

    def test_stick_endpoints_and_asymmetric_midpoint(self):
        # Standard symmetric stick (-32768 ~ 32767)
        # mid = -0.5, span = 32767.5
        pos = normalize_axis_value(32767, -32768, 32767, deadzone=0.15)
        self.assertEqual(pos, 1.0)

        neg = normalize_axis_value(-32768, -32768, 32767, deadzone=0.15)
        self.assertEqual(neg, -1.0)

        # Asymmetric range: 0 ~ 100 -> mid = 50, span = 50
        mid_val = normalize_axis_value(50, 0, 100, deadzone=0.15)
        self.assertEqual(mid_val, 0.0)

        high_val = normalize_axis_value(100, 0, 100, deadzone=0.15)
        self.assertEqual(high_val, 1.0)

        low_val = normalize_axis_value(0, 0, 100, deadzone=0.15)
        self.assertEqual(low_val, -1.0)

    def test_stick_deadzone_boundaries(self):
        # Range -100 ~ 100 -> mid = 0, span = 100
        # abs(n_val) < deadzone sets 0.0; equal preserves value
        deadzone = 0.15

        # Strictly less than deadzone -> 0.0
        self.assertEqual(normalize_axis_value(14, -100, 100, deadzone=deadzone), 0.0)
        self.assertEqual(normalize_axis_value(-14, -100, 100, deadzone=deadzone), 0.0)

        # Exactly equal to deadzone -> preserved
        self.assertEqual(normalize_axis_value(15, -100, 100, deadzone=deadzone), 0.15)
        self.assertEqual(normalize_axis_value(-15, -100, 100, deadzone=deadzone), -0.15)

        # Greater than deadzone -> preserved
        self.assertEqual(normalize_axis_value(50, -100, 100, deadzone=deadzone), 0.5)
        self.assertEqual(normalize_axis_value(-50, -100, 100, deadzone=deadzone), -0.5)

    def test_stick_invert_y_applied_after_deadzone(self):
        deadzone = 0.15

        # Inside deadzone: 0.0 -> -0.0 -> 0.0
        self.assertEqual(
            normalize_axis_value(10, -100, 100, deadzone=deadzone, invert_y=True),
            0.0,
        )

        # Outside deadzone: 0.5 -> -0.5
        self.assertEqual(
            normalize_axis_value(50, -100, 100, deadzone=deadzone, invert_y=True),
            -0.5,
        )

        # Negative outside deadzone: -0.5 -> 0.5
        self.assertEqual(
            normalize_axis_value(-50, -100, 100, deadzone=deadzone, invert_y=True),
            0.5,
        )

if __name__ == "__main__":
    unittest.main()
