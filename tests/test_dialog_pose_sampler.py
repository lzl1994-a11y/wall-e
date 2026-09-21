import random
import unittest
from unittest.mock import Mock, patch

from services.dialog_pose_sampler import DialogPoseSampler


def _sample_servos():
    return {
        "eye_r": {"init": 2800, "limit_1": 1920, "limit_2": 4380},
        "eye_l": {"init": 6500, "limit_1": 7400, "limit_2": 5200},
        "eyebrow_r": {"init": 1920, "limit_1": 1920, "limit_2": 4200},
        "eyebrow_l": {"init": 8000, "limit_1": 8000, "limit_2": 5700},
        "head_yaw": {"init": 5000, "limit_1": 1920, "limit_2": 7600},
        "neck_top": {"init": 2000, "limit_1": 2000, "limit_2": 5000},
        "neck_bottom": {"init": 2000, "limit_1": 2000, "limit_2": 4800},
    }


class DialogPoseSamplerTests(unittest.TestCase):
    def test_deterministic_sampling_with_fixed_seed(self):
        servos = _sample_servos()
        sampler1 = DialogPoseSampler(servos, rng=random.Random(12345))
        sampler2 = DialogPoseSampler(servos, rng=random.Random(12345))

        for _ in range(20):
            self.assertEqual(sampler1.speaking_pose(), sampler2.speaking_pose())

    def test_coupled_eye_gap_preserved(self):
        servos = _sample_servos()
        sampler = DialogPoseSampler(servos, rng=random.Random(42))
        expected_gap = servos["eye_l"]["init"] - servos["eye_r"]["init"]

        for _ in range(50):
            pose = sampler.speaking_pose()
            self.assertEqual(pose["eye_l"] - pose["eye_r"], expected_gap)

    def test_mirrored_eyebrows_movement(self):
        servos = _sample_servos()
        sampler = DialogPoseSampler(servos, rng=random.Random(42))

        for _ in range(50):
            pose = sampler.speaking_pose()
            right_diff = pose["eyebrow_r"] - servos["eyebrow_r"]["init"]
            left_diff = servos["eyebrow_l"]["init"] - pose["eyebrow_l"]
            self.assertEqual(right_diff, left_diff)
            self.assertGreaterEqual(right_diff, 0)

    def test_all_outputs_inside_limits_repeatedly(self):
        servos = _sample_servos()
        sampler = DialogPoseSampler(servos, rng=random.Random(999))

        for _ in range(200):
            pose = sampler.speaking_pose()
            self.assertEqual(
                set(pose.keys()),
                {"eye_r", "eye_l", "eyebrow_r", "eyebrow_l", "head_yaw", "neck_top", "neck_bottom"},
            )
            for name, val in pose.items():
                self.assertIsInstance(val, int)
                cfg = servos[name]
                low = min(cfg["limit_1"], cfg["limit_2"])
                high = max(cfg["limit_1"], cfg["limit_2"])
                self.assertGreaterEqual(val, low)
                self.assertLessEqual(val, high)

    def test_neck_pitch_range_and_targets(self):
        servos = _sample_servos()
        mock_kinematics = Mock()
        mock_kinematics.targets.return_value = {"neck_top": 2100, "neck_bottom": 2600}

        with patch(
            "services.dialog_pose_sampler.neck_kinematics_from_servos",
            return_value=mock_kinematics,
        ):
            sampler = DialogPoseSampler(servos, rng=random.Random(42))
            for _ in range(50):
                mock_kinematics.targets.reset_mock()
                pose = sampler.speaking_pose()
                mock_kinematics.targets.assert_called_once()
                pitch = mock_kinematics.targets.call_args[0][0]
                self.assertGreaterEqual(pitch, 0.15)
                self.assertLessEqual(pitch, 0.30)
                self.assertEqual(pose["neck_top"], 2100)
                self.assertEqual(pose["neck_bottom"], 2600)

    def test_step_size(self):
        servos = _sample_servos()
        sampler = DialogPoseSampler(servos)
        self.assertEqual(sampler.step_size, 24.0)

    def test_different_valid_servo_limit_configurations(self):
        custom_servos = {
            "eye_r": {"init": 3000, "limit_1": 4000, "limit_2": 2000},
            "eye_l": {"init": 6000, "limit_1": 5000, "limit_2": 7000},
            "eyebrow_r": {"init": 2000, "limit_1": 3000, "limit_2": 1500},
            "eyebrow_l": {"init": 7000, "limit_1": 6000, "limit_2": 8000},
            "head_yaw": {"init": 4500, "limit_1": 6000, "limit_2": 3000},
            "neck_top": {"init": 2500, "limit_1": 2000, "limit_2": 4000},
            "neck_bottom": {"init": 2500, "limit_1": 2000, "limit_2": 4000},
        }
        sampler = DialogPoseSampler(custom_servos, rng=random.Random(42))
        expected_gap = custom_servos["eye_l"]["init"] - custom_servos["eye_r"]["init"]

        for _ in range(50):
            pose = sampler.speaking_pose()
            self.assertEqual(pose["eye_l"] - pose["eye_r"], expected_gap)
            for name, val in pose.items():
                cfg = custom_servos[name]
                low = min(cfg["limit_1"], cfg["limit_2"])
                high = max(cfg["limit_1"], cfg["limit_2"])
                self.assertGreaterEqual(val, low)
                self.assertLessEqual(val, high)

    def test_fixed_seed_matches_pre_migration_behavior(self):
        servos = _sample_servos()
        sampler = DialogPoseSampler(servos, rng=random.Random(42))
        first_pose = sampler.speaking_pose()
        expected_first = {
            "eye_r": 2934,
            "eye_l": 6634,
            "eyebrow_r": 1977,
            "eyebrow_l": 7943,
            "head_yaw": 4374,
            "neck_top": 2000,
            "neck_bottom": 2731,
        }
        self.assertEqual(first_pose, expected_first)

        second_pose = sampler.speaking_pose()
        expected_second = {
            "eye_r": 2732,
            "eye_l": 6432,
            "eyebrow_r": 2034,
            "eyebrow_l": 7886,
            "head_yaw": 4608,
            "neck_top": 2000,
            "neck_bottom": 2729,
        }
        self.assertEqual(second_pose, expected_second)

    def test_default_rng_creates_independent_instances(self):
        servos = _sample_servos()
        sampler1 = DialogPoseSampler(servos)
        sampler2 = DialogPoseSampler(servos)
        self.assertIsInstance(sampler1._rng, random.Random)
        self.assertIsInstance(sampler2._rng, random.Random)
        self.assertIsNot(sampler1._rng, sampler2._rng)


if __name__ == "__main__":
    unittest.main()
