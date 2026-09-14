import unittest

from services.sequence_execution import (
    SequenceCommandController,
    SequenceLibrary,
    SequenceRuntime,
    ServoTrajectory,
)


SERVOS = {
    "eye_r": {
        "name": "eye_r", "init": 3000, "limit_1": 1900, "limit_2": 4400,
    },
    "eye_l": {
        "name": "eye_l", "init": 6500, "limit_1": 7400, "limit_2": 5000,
    },
    "head_yaw": {
        "name": "head_yaw", "init": 5000, "limit_1": 1900, "limit_2": 7600,
    },
}


class SequenceLibraryTests(unittest.TestCase):
    def test_nested_sequences_are_flattened_with_offsets(self):
        library = SequenceLibrary(
            sequences={
                "greeting": [{
                    "time": 0.5,
                    "actions": [
                        {"type": "sequence", "name": "wave"},
                        {"type": "motor", "direction": "forward"},
                    ],
                }],
                "wave": [{
                    "time": 0.25,
                    "actions": [{"type": "pose", "name": "hand_up"}],
                }],
            },
            poses={"hand_up": {"targets": {"arm_r": 4000}}},
        )

        self.assertEqual(library.flatten("greeting"), [
            {
                "time": 0.75,
                "actions": [{"type": "pose", "name": "hand_up"}],
            },
            {
                "time": 0.5,
                "actions": [{"type": "motor", "direction": "forward"}],
            },
        ])

    def test_pose_name_can_be_used_as_a_single_frame_sequence(self):
        library = SequenceLibrary({}, {"look_center": {"targets": {}}})

        self.assertEqual(library.flatten("look_center", offset_time=1.5), [{
            "time": 1.5,
            "actions": [{"type": "pose", "name": "look_center"}],
        }])


class ServoTrajectoryTests(unittest.TestCase):
    def test_targets_are_clamped_and_interpolated_without_ros(self):
        trajectory = ServoTrajectory(SERVOS)
        trajectory.tick()  # consume the initial full-state publication

        trajectory.apply_targets({"head_yaw": 9000}, step_size=100)
        changed = trajectory.tick()

        self.assertEqual(trajectory.targets["head_yaw"], 7600)
        self.assertEqual(changed["head_yaw"], 5100)

    def test_head_eye_collision_rule_is_applied_before_interpolation(self):
        trajectory = ServoTrajectory(SERVOS)
        trajectory.virtual_state["eye_r"] = 2500
        trajectory.targets["eye_r"] = 2500
        trajectory.targets["head_yaw"] = 6000
        trajectory.steps["head_yaw"] = 100

        changed = trajectory.tick()

        self.assertEqual(trajectory.targets["eye_r"], 3000)
        self.assertEqual(changed["head_yaw"], 5000)
        self.assertEqual(changed["eye_r"], 2530)


class SequenceRuntimeTests(unittest.TestCase):
    def _runtime(self, *, sequences=None, poses=None):
        trajectory = ServoTrajectory(SERVOS)
        library = SequenceLibrary(sequences or {}, poses or {})
        return SequenceRuntime(library, trajectory)

    def test_timeline_is_sorted_and_dispatches_one_frame_per_tick(self):
        runtime = self._runtime(sequences={
            "demo": [
                {"time": 0.5, "actions": [{"type": "express_emotion", "emotion": "happy"}]},
                {"time": 0.1, "actions": [{"type": "express_emotion", "emotion": "sad"}]},
            ],
        })
        self.assertEqual(runtime.start_sequence("demo", now=10.0), 2)

        first = runtime.tick(wall_now=11.0, monotonic_now=20.0)
        second = runtime.tick(wall_now=11.0, monotonic_now=20.02)

        self.assertEqual([(item.kind, item.payload) for item in first.effects], [
            ("emotion", "sad"),
        ])
        self.assertEqual([(item.kind, item.payload) for item in second.effects], [
            ("emotion", "happy"),
        ])

    def test_motor_is_refreshed_until_its_deadline_then_stopped(self):
        runtime = self._runtime()
        started = runtime.dispatch_action(
            {"type": "motor", "direction": "forward", "duration": 1.0},
            monotonic_now=5.0,
        )
        active = runtime.tick(wall_now=0.0, monotonic_now=5.5)
        stopped = runtime.tick(wall_now=0.0, monotonic_now=6.1)

        self.assertEqual(started[0].kind, "motor")
        self.assertEqual(active.effects[0].kind, "motor")
        self.assertEqual(stopped.effects[0].kind, "motor_stop")
        self.assertIsNone(runtime.active_motor_command)

    def test_pose_dispatch_uses_runtime_servo_calibration(self):
        runtime = self._runtime(poses={
            "look": {
                "default_step": 25,
                "targets": {"head_yaw": "max"},
            },
        })

        runtime.dispatch_action(
            {"type": "pose", "name": "look"},
            monotonic_now=0.0,
        )

        self.assertEqual(runtime.trajectory.targets["head_yaw"], 7600)
        self.assertEqual(runtime.trajectory.steps["head_yaw"], 25)


class SequenceCommandControllerTests(unittest.TestCase):
    def _controller(self, *, sequences=None):
        trajectory = ServoTrajectory(SERVOS)
        library = SequenceLibrary(sequences or {}, {})
        return SequenceCommandController(SequenceRuntime(library, trajectory))

    @staticmethod
    def _statuses(effects):
        return [
            effect.payload["status"]
            for effect in effects
            if effect.kind == "status"
        ]

    def test_move_request_completes_when_motor_deadline_expires(self):
        controller = self._controller()
        request = {
            "name": "move_chassis",
            "arguments": {"direction": "forward", "duration": 0.25},
            "request_id": "move",
        }

        started = controller.handle_action(
            request, wall_now=1.0, monotonic_now=10.0
        )
        finished = controller.tick(wall_now=1.5, monotonic_now=10.3)

        self.assertEqual(self._statuses(started), ["accepted"])
        self.assertIn("motor", [effect.kind for effect in started])
        self.assertEqual(self._statuses(finished.effects), ["completed"])
        self.assertIn("motor_stop", [effect.kind for effect in finished.effects])

    def test_targeted_cancel_stops_matching_sequence_without_duplicate_status(self):
        controller = self._controller(sequences={
            "dance": [{
                "time": 2.0,
                "actions": [{"type": "express_emotion", "emotion": "happy"}],
            }],
        })
        controller.handle_action({
            "name": "play_sequence",
            "arguments": {"sequence_name": "dance"},
            "request_id": "dance-request",
        }, wall_now=1.0, monotonic_now=1.0)

        ignored = controller.cancel({
            "request_id": "other",
            "reason": "preempted",
        })
        cancelled = controller.cancel({
            "request_id": "dance-request",
            "reason": "preempted",
        })

        self.assertEqual(ignored, ())
        self.assertEqual(self._statuses(cancelled), [])
        self.assertEqual(controller.runtime.timeline, [])

    def test_game_mode_interruption_still_reports_sequence_status(self):
        controller = self._controller(sequences={
            "dance": [{
                "time": 2.0,
                "actions": [{"type": "express_emotion", "emotion": "happy"}],
            }],
        })
        controller.handle_action({
            "name": "play_sequence",
            "arguments": {"sequence_name": "dance"},
            "request_id": "dance-request",
        }, wall_now=1.0, monotonic_now=1.0)

        effects = controller.set_game_active(True)

        self.assertEqual(self._statuses(effects), ["interrupted"])

    def test_replacement_command_does_not_repeat_preemption_status(self):
        controller = self._controller(sequences={
            "dance": [{"time": 2.0, "actions": []}],
        })
        controller.handle_action({
            "name": "play_sequence",
            "arguments": {"sequence_name": "dance"},
            "request_id": "old",
        }, wall_now=1.0, monotonic_now=1.0)

        effects = controller.handle_action({
            "name": "express_emotion",
            "arguments": {"emotion": "happy"},
            "request_id": "new",
        }, wall_now=1.1, monotonic_now=1.1)

        statuses = [
            effect.payload
            for effect in effects
            if effect.kind == "status"
        ]
        self.assertEqual(
            [
                (status["request"]["request_id"], status["status"])
                for status in statuses
            ],
            [("new", "accepted"), ("new", "completed")],
        )

    def test_dialog_expression_waits_for_explicit_motion(self):
        controller = self._controller()
        controller.handle_action({
            "name": "manual_servo",
            "arguments": {"targets": {"head_yaw": 5100}, "step_size": 100},
            "request_id": "manual",
        }, wall_now=1.0, monotonic_now=1.0)
        controller.apply_dialog_expression({"head_yaw": 6200}, 50)

        self.assertEqual(controller.runtime.trajectory.targets["head_yaw"], 5100)
        controller.tick(wall_now=1.1, monotonic_now=1.1)

        self.assertEqual(controller.runtime.trajectory.targets["head_yaw"], 6200)
        self.assertFalse(controller.runtime.explicit_motion_active)


if __name__ == "__main__":
    unittest.main()
