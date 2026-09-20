import unittest

from services.tracking_control import (
    HeadTarget,
    LossState,
    PID,
    TargetSelector,
    TrackingController,
    TrackingExitReason,
)


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
            search_rotate_speed=0.25,
            search_start_seconds=1.0,
            search_stop_seconds=5.0,
            tracking_shutdown_seconds=60.0,
            pipeline_startup_timeout_seconds=180.0,
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

    def test_body_loss_holds_then_searches_in_last_seen_direction(self):
        controller = self._controller()
        controller.last_horizontal_error = -0.5

        holding = controller.handle_target_loss(
            gaze=False,
            detector_ready=True,
            detector_stale=False,
            lost_seconds=0.5,
            mode_seconds=0.5,
        )
        searching = controller.handle_target_loss(
            gaze=False,
            detector_ready=True,
            detector_stale=False,
            lost_seconds=1.1,
            mode_seconds=1.1,
        )

        self.assertEqual(holding.state, LossState.HOLDING)
        self.assertEqual((holding.motor.left, holding.motor.right), (0.0, 0.0))
        self.assertEqual(searching.state, LossState.SEARCHING)
        self.assertEqual(
            (searching.motor.left, searching.motor.right), (-0.25, 0.25)
        )
        self.assertEqual(searching.head, HeadTarget(x_error=0.0, pitch=0.0))

    def test_search_stops_once_and_exits_after_long_loss(self):
        controller = self._controller()

        stopped = controller.handle_target_loss(
            gaze=False,
            detector_ready=True,
            detector_stale=False,
            lost_seconds=5.1,
            mode_seconds=5.1,
        )
        repeated = controller.handle_target_loss(
            gaze=False,
            detector_ready=True,
            detector_stale=False,
            lost_seconds=5.2,
            mode_seconds=5.2,
        )
        exited = controller.handle_target_loss(
            gaze=False,
            detector_ready=True,
            detector_stale=True,
            lost_seconds=60.1,
            mode_seconds=60.1,
        )

        self.assertEqual(stopped.state, LossState.SEARCH_STOPPED)
        self.assertIsNotNone(stopped.motor)
        self.assertIsNotNone(stopped.head)
        self.assertIsNone(repeated.motor)
        self.assertIsNone(repeated.head)
        self.assertEqual(exited.state, LossState.EXITED)
        self.assertEqual(exited.exit_reason, TrackingExitReason.TARGET_LOST)

    def test_stalled_detector_stops_without_forgetting_active_search(self):
        controller = self._controller()
        first_search = controller.handle_target_loss(
            gaze=False,
            detector_ready=True,
            detector_stale=False,
            lost_seconds=1.1,
            mode_seconds=1.1,
        )
        stalled = controller.handle_target_loss(
            gaze=False,
            detector_ready=True,
            detector_stale=True,
            lost_seconds=2.0,
            mode_seconds=2.0,
        )
        resumed = controller.handle_target_loss(
            gaze=False,
            detector_ready=True,
            detector_stale=False,
            lost_seconds=2.1,
            mode_seconds=2.1,
        )

        self.assertIsNotNone(first_search.head)
        self.assertEqual(stalled.state, LossState.DETECTOR_UNAVAILABLE)
        self.assertEqual((stalled.motor.left, stalled.motor.right), (0.0, 0.0))
        self.assertIsNone(resumed.head)

    def test_gaze_never_searches_and_pipeline_startup_can_time_out(self):
        controller = self._controller()
        controller.reset(gaze=True)

        gaze = controller.handle_target_loss(
            gaze=True,
            detector_ready=True,
            detector_stale=False,
            lost_seconds=2.0,
            mode_seconds=2.0,
        )
        startup_timeout = controller.handle_target_loss(
            gaze=True,
            detector_ready=False,
            detector_stale=True,
            lost_seconds=180.1,
            mode_seconds=180.1,
        )

        self.assertEqual(gaze.state, LossState.HOLDING)
        self.assertEqual((gaze.motor.left, gaze.motor.right), (0.0, 0.0))
        self.assertEqual(
            startup_timeout.exit_reason,
            TrackingExitReason.PIPELINE_STARTUP_TIMEOUT,
        )


if __name__ == "__main__":
    unittest.main()
