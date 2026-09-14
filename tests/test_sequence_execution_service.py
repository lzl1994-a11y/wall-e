import unittest

from services.sequence_execution import SequenceLibrary, ServoTrajectory


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


if __name__ == "__main__":
    unittest.main()
