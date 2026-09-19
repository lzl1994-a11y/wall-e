from __future__ import annotations

import copy
import unittest

from services.dialog_expression_pose import (
    DEFAULT_STEP_SIZE,
    DialogExpressionPose,
    resolve_dialog_expression_pose,
)


SYNTHETIC_SERVOS = {
    "eye_r": {"name": "eye_r", "init": 3000, "limit_1": 2000, "limit_2": 4000},
    "eye_l": {"name": "eye_l", "init": 6500, "limit_1": 5500, "limit_2": 7500},
    "eyebrow_r": {"name": "eyebrow_r", "init": 2000, "limit_1": 1000, "limit_2": 4000},
    "eyebrow_l": {"name": "eyebrow_l", "init": 8000, "limit_1": 6000, "limit_2": 9000},
    "head_yaw": {"name": "head_yaw", "init": 5000, "limit_1": 3000, "limit_2": 7000},
    "neck_top": {"name": "neck_top", "init": 5000, "limit_1": 5000, "limit_2": 6000},
    "neck_bottom": {"name": "neck_bottom", "init": 3000, "limit_1": 2000, "limit_2": 4800},
}

SYNTHETIC_EXPRESSION_POSES = {
    "neutral": {
        "default_step": 24.0,
        "targets": {
            "eye_r": 3000,
            "eye_l": 6500,
            "eyebrow_r": 2000,
            "eyebrow_l": 8000,
            "head_yaw": 5000,
            "neck_top": "init",
            "neck_bottom": "init",
        },
    },
    "happy": {
        "default_step": 20.0,
        "targets": {
            "eye_r": 3000,
            "eye_l": 6500,
            "eyebrow_r": 3000,  # neutral is 2000, diff is +1000
            "eyebrow_l": 7000,  # neutral is 8000, diff is -1000
            "head_yaw": 5000,
            "neck_top": "init",
            "neck_bottom": {"toward": "max", "fraction": 0.5},  # init 3000, max 4800 -> 3900, diff +900
        },
    },
    "surprised": {
        "targets": {
            "neck_top": "max",
            "eyebrow_r": 3500,
            "extra_servo": 3500,  # exists in servos, but not in neutral
            "unconfigured_servo": 3500,  # not in servos
            "bad_target_servo": "invalid_value",
        },
    },
}


class DialogExpressionPoseTests(unittest.TestCase):
    def test_neutral_pose_resolution(self):
        pose = resolve_dialog_expression_pose(
            SYNTHETIC_EXPRESSION_POSES,
            SYNTHETIC_SERVOS,
            "neutral",
            intensity="high",
            default_step=24.0,
        )
        self.assertIsInstance(pose, DialogExpressionPose)
        self.assertEqual(pose.step_size, 24.0)
        self.assertEqual(
            pose.targets,
            {
                "eye_r": 3000,
                "eye_l": 6500,
                "eyebrow_r": 2000,
                "eyebrow_l": 8000,
                "head_yaw": 5000,
                "neck_top": 5000,
                "neck_bottom": 3000,
            },
        )

    def test_happy_expression_intensities(self):
        # eyebrow_r: neutral=2000, happy=3000 -> diff = 1000
        # eyebrow_l: neutral=8000, happy=7000 -> diff = -1000
        # neck_bottom: neutral=3000, happy=3900 -> diff = 900
        # low = 0.6:
        # eyebrow_r = 2000 + 600 = 2600
        # eyebrow_l = 8000 - 600 = 7400
        # neck_bottom = 3000 + 540 = 3540
        pose_low = resolve_dialog_expression_pose(
            SYNTHETIC_EXPRESSION_POSES,
            SYNTHETIC_SERVOS,
            "happy",
            intensity="low",
        )
        self.assertEqual(pose_low.step_size, 20.0)
        self.assertEqual(pose_low.targets["eyebrow_r"], 2600)
        self.assertEqual(pose_low.targets["eyebrow_l"], 7400)
        self.assertEqual(pose_low.targets["neck_bottom"], 3540)

        # medium = 0.85:
        # eyebrow_r = 2000 + 850 = 2850
        # eyebrow_l = 8000 - 850 = 7150
        # neck_bottom = 3000 + int(round(900 * 0.85)) = 3000 + 765 = 3765
        pose_med = resolve_dialog_expression_pose(
            SYNTHETIC_EXPRESSION_POSES,
            SYNTHETIC_SERVOS,
            "happy",
            intensity="medium",
        )
        self.assertEqual(pose_med.targets["eyebrow_r"], 2850)
        self.assertEqual(pose_med.targets["eyebrow_l"], 7150)
        self.assertEqual(pose_med.targets["neck_bottom"], 3765)

        # high = 1.0:
        # eyebrow_r = 3000, eyebrow_l = 7000, neck_bottom = 3900
        pose_high = resolve_dialog_expression_pose(
            SYNTHETIC_EXPRESSION_POSES,
            SYNTHETIC_SERVOS,
            "happy",
            intensity="high",
        )
        self.assertEqual(pose_high.targets["eyebrow_r"], 3000)
        self.assertEqual(pose_high.targets["eyebrow_l"], 7000)
        self.assertEqual(pose_high.targets["neck_bottom"], 3900)

        # unknown intensity falls back to 0.6
        pose_other = resolve_dialog_expression_pose(
            SYNTHETIC_EXPRESSION_POSES,
            SYNTHETIC_SERVOS,
            "happy",
            intensity="unknown_val",
        )
        self.assertEqual(pose_other.targets["eyebrow_r"], 2600)

    def test_missing_or_empty_expression_falls_back_to_neutral(self):
        # Missing expression
        pose_missing = resolve_dialog_expression_pose(
            SYNTHETIC_EXPRESSION_POSES,
            SYNTHETIC_SERVOS,
            "non_existent",
            intensity="high",
        )
        expected_neutral = resolve_dialog_expression_pose(
            SYNTHETIC_EXPRESSION_POSES,
            SYNTHETIC_SERVOS,
            "neutral",
        )
        self.assertEqual(pose_missing.targets, expected_neutral.targets)
        self.assertEqual(pose_missing.step_size, expected_neutral.step_size)

        # Empty expression dict falls back to neutral
        poses_with_empty = dict(SYNTHETIC_EXPRESSION_POSES)
        poses_with_empty["empty_expr"] = {}
        pose_empty = resolve_dialog_expression_pose(
            poses_with_empty,
            SYNTHETIC_SERVOS,
            "empty_expr",
        )
        self.assertEqual(pose_empty.targets, expected_neutral.targets)

    def test_target_missing_in_neutral_uses_own_target_as_base(self):
        # Add extra servo to servos dict
        servos = dict(SYNTHETIC_SERVOS)
        servos["extra_servo"] = {
            "name": "extra_servo",
            "init": 2500,
            "limit_1": 1000,
            "limit_2": 4000,
        }
        # neutral does not have extra_servo, but surprised does (3500)
        pose = resolve_dialog_expression_pose(
            SYNTHETIC_EXPRESSION_POSES,
            servos,
            "surprised",
            intensity="low",
        )
        # diff from self is 0, so target stays 3500 regardless of factor
        self.assertEqual(pose.targets["extra_servo"], 3500)

    def test_invalid_servos_and_unresolvable_targets_ignored(self):
        servos = dict(SYNTHETIC_SERVOS)
        servos["bad_target_servo"] = {
            "name": "bad_target_servo",
            "init": 2000,
            "limit_1": 1000,
            "limit_2": 3000,
        }
        pose = resolve_dialog_expression_pose(
            SYNTHETIC_EXPRESSION_POSES,
            servos,
            "surprised",
        )
        self.assertNotIn("unconfigured_servo", pose.targets)
        self.assertNotIn("bad_target_servo", pose.targets)
        self.assertIn("neck_top", pose.targets)

    def test_symbolic_and_relative_targets_resolved(self):
        # Symbolic 'max'
        pose_surprised = resolve_dialog_expression_pose(
            SYNTHETIC_EXPRESSION_POSES,
            SYNTHETIC_SERVOS,
            "surprised",
            intensity="high",
        )
        # neck_top target was "max", limit_1=5000, limit_2=6000 -> 6000
        self.assertEqual(pose_surprised.targets["neck_top"], 6000)

        # Absolute, symbolic 'init', and relative {'toward': 'max', 'fraction': 0.5}
        pose_happy = resolve_dialog_expression_pose(
            SYNTHETIC_EXPRESSION_POSES,
            SYNTHETIC_SERVOS,
            "happy",
            intensity="high",
        )
        # eye_r was absolute 3000
        self.assertEqual(pose_happy.targets["eye_r"], 3000)
        # neck_top was symbolic 'init' -> 5000
        self.assertEqual(pose_happy.targets["neck_top"], 5000)
        # neck_bottom was relative {'toward': 'max', 'fraction': 0.5} -> 3900
        self.assertEqual(pose_happy.targets["neck_bottom"], 3900)

    def test_default_step_precedence_and_falsy_fallback(self):
        # 1. Pose specifies default_step (happy has 20.0)
        pose_happy = resolve_dialog_expression_pose(
            SYNTHETIC_EXPRESSION_POSES,
            SYNTHETIC_SERVOS,
            "happy",
            default_step=30.0,
        )
        self.assertEqual(pose_happy.step_size, 20.0)
        self.assertIsInstance(pose_happy.step_size, float)

        # 2. Pose lacks default_step (surprised has none) -> falls back to passed default_step
        pose_surprised = resolve_dialog_expression_pose(
            SYNTHETIC_EXPRESSION_POSES,
            SYNTHETIC_SERVOS,
            "surprised",
            default_step=16.0,
        )
        self.assertEqual(pose_surprised.step_size, 16.0)

        # 3. Pose has falsy default_step (None or 0)
        poses_falsy = {
            "neutral": {"targets": {}},
            "zero_step": {"default_step": 0, "targets": {}},
            "none_step": {"default_step": None, "targets": {}},
        }
        pose_zero = resolve_dialog_expression_pose(
            poses_falsy,
            SYNTHETIC_SERVOS,
            "zero_step",
            default_step=24.0,
        )
        self.assertEqual(pose_zero.step_size, 24.0)

        pose_none = resolve_dialog_expression_pose(
            poses_falsy,
            SYNTHETIC_SERVOS,
            "none_step",
            default_step=24.0,
        )
        self.assertEqual(pose_none.step_size, 24.0)

        # 4. If default_step parameter is also omitted/None, falls back to DEFAULT_STEP_SIZE
        pose_default = resolve_dialog_expression_pose(
            poses_falsy,
            SYNTHETIC_SERVOS,
            "none_step",
            default_step=None,
        )
        self.assertEqual(pose_default.step_size, DEFAULT_STEP_SIZE)

    def test_input_configurations_not_mutated(self):
        poses_copy = copy.deepcopy(SYNTHETIC_EXPRESSION_POSES)
        servos_copy = copy.deepcopy(SYNTHETIC_SERVOS)

        resolve_dialog_expression_pose(
            poses_copy,
            servos_copy,
            "happy",
            intensity="medium",
        )

        self.assertEqual(poses_copy, SYNTHETIC_EXPRESSION_POSES)
        self.assertEqual(servos_copy, SYNTHETIC_SERVOS)


if __name__ == "__main__":
    unittest.main()
