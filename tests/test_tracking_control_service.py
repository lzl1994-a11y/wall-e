import unittest

from services.tracking_control import PID, TargetSelector


class PIDTests(unittest.TestCase):
    def test_output_is_bounded_and_reset_clears_history(self):
        pid = PID(kp=2.0, ki=1.0, kd=1.0, out_min=-0.5, out_max=0.5)

        self.assertEqual(pid.update(1.0, 0.1), 0.5)
        pid.reset()

        self.assertEqual(pid.integral, 0.0)
        self.assertIsNone(pid.prev_error)
        self.assertEqual(pid.update(1.0, 0.0), 0.0)


class TargetSelectorTests(unittest.TestCase):
    def _selector(self):
        return TargetSelector(
            image_width=960,
            image_height=544,
            memory_seconds=1.5,
            filter_seconds=0.15,
        )

    def test_new_track_uses_largest_box(self):
        selector = self._selector()
        small = (100.0, 100.0, 0.1)
        large = (500.0, 250.0, 0.3)

        selected, reacquired = selector.select(
            [small, large], kind="body", dt=0.1, now=10.0
        )

        self.assertEqual(selected, large)
        self.assertTrue(reacquired)

    def test_live_track_prefers_spatial_continuity(self):
        selector = self._selector()
        selector.select([(200.0, 272.0, 0.2)], kind="body", dt=0.1, now=10.0)

        selected, reacquired = selector.select(
            [(205.0, 272.0, 0.18), (750.0, 272.0, 0.3)],
            kind="body",
            dt=0.1,
            now=10.1,
        )

        self.assertIsNotNone(selected)
        self.assertLess(selected[0], 210.0)
        self.assertFalse(reacquired)

    def test_expired_track_can_acquire_another_person(self):
        selector = self._selector()
        selector.select([(200.0, 272.0, 0.2)], kind="body", dt=0.1, now=10.0)

        selected, reacquired = selector.select(
            [(750.0, 272.0, 0.3)], kind="body", dt=0.1, now=12.0
        )

        self.assertEqual(selected, (750.0, 272.0, 0.3))
        self.assertTrue(reacquired)


if __name__ == "__main__":
    unittest.main()
