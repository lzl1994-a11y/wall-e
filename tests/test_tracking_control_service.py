import unittest

from services.tracking_control import PID, TargetSelector, TrackingController


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


class TrackingControllerTests(unittest.TestCase):
    def _controller(self):
        return TrackingController(
            image_width=960,
            image_height=544,
            body_target_ratio=0.35,
            target_memory_seconds=1.5,
            filter_seconds=0.15,
            gaze_start_pitch=0.18,
            gaze_min_pitch=-0.20,
            gaze_max_pitch=0.65,
            pitch_rate=0.35,
        )

    def test_body_follow_returns_motor_and_level_head_targets(self):
        controller = self._controller()

        decision = controller.follow_body(
            [(720.0, 272.0, 0.20)], dt=0.1, now=10.0
        )

        self.assertTrue(decision.target_seen)
        self.assertAlmostEqual(decision.motor.left, -0.2625)
        self.assertAlmostEqual(decision.motor.right, 0.3375)
        self.assertEqual(decision.head.x_error, -0.5)
        self.assertEqual(decision.head.pitch, 0.0)
        self.assertEqual(controller.last_horizontal_error, -0.5)

    def test_body_follow_with_no_target_holds_all_outputs(self):
        decision = self._controller().follow_body([], dt=0.1, now=10.0)

        self.assertFalse(decision.target_seen)
        self.assertIsNone(decision.motor)
        self.assertIsNone(decision.head)

    def test_face_gaze_stops_chassis_and_rate_limits_pitch(self):
        controller = self._controller()
        controller.reset(gaze=True)

        decision = controller.gaze_at_face(
            [(480.0, 400.0, 0.05)], [], dt=0.1, now=10.0
        )

        self.assertTrue(decision.target_seen)
        self.assertEqual((decision.motor.left, decision.motor.right), (0.0, 0.0))
        self.assertAlmostEqual(decision.head.pitch, 0.1668235294117647)

    def test_body_fallback_turns_head_without_changing_gaze_pitch(self):
        controller = self._controller()
        controller.reset(gaze=True)

        decision = controller.gaze_at_face(
            [], [(240.0, 450.0, 0.2)], dt=0.1, now=10.0
        )

        self.assertTrue(decision.target_seen)
        self.assertEqual(decision.head.x_error, 0.5)
        self.assertEqual(decision.head.pitch, 0.18)
        self.assertEqual(controller.current_neck_pitch, 0.18)

    def test_empty_gaze_result_still_returns_fail_safe_motor_stop(self):
        decision = self._controller().gaze_at_face([], [], dt=0.1, now=10.0)

        self.assertFalse(decision.target_seen)
        self.assertEqual((decision.motor.left, decision.motor.right), (0.0, 0.0))
        self.assertIsNone(decision.head)

if __name__ == "__main__":
    unittest.main()
