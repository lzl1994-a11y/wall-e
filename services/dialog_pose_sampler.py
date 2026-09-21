"""Pure conversational pose sampling model for speaking motions."""

from __future__ import annotations

import random

from services.servo_motion_config import neck_kinematics_from_servos


class DialogPoseSampler:
    """Sample coupled face, head-yaw, and neck-pitch poses within limits."""

    _STEP_SIZE = 24.0
    _DIALOG_RANGE_FRACTION = 0.22
    _NECK_PITCH_MIN = 0.15
    _NECK_PITCH_MAX = 0.30

    def __init__(self, servos, rng=None):
        self._servos = servos
        self._rng = rng or random.Random()
        self._eye_range = self._coupled_eye_range()
        self._eyebrow_open_range = self._eyebrow_open_range()
        self._head_yaw_range = self._half_offset_range("head_yaw")
        self._neck_kinematics = neck_kinematics_from_servos(servos)

    @staticmethod
    def _clamp(value, cfg):
        return int(max(min(cfg["limit_1"], cfg["limit_2"]), min(
            max(cfg["limit_1"], cfg["limit_2"]), value
        )))

    def _coupled_eye_range(self):
        """Return half of the common safe range for equal eye offsets."""
        lower = max(
            min(self._servos[name]["limit_1"], self._servos[name]["limit_2"])
            - self._servos[name]["init"]
            for name in ("eye_r", "eye_l")
        )
        upper = min(
            max(self._servos[name]["limit_1"], self._servos[name]["limit_2"])
            - self._servos[name]["init"]
            for name in ("eye_r", "eye_l")
        )
        return (
            int(lower * self._DIALOG_RANGE_FRACTION),
            int(upper * self._DIALOG_RANGE_FRACTION),
        )

    def _eyebrow_open_range(self):
        """Return half of the common safe range for mirrored eyebrow opening."""
        right = self._servos["eyebrow_r"]
        left = self._servos["eyebrow_l"]
        right_max = max(right["limit_1"], right["limit_2"])
        left_min = min(left["limit_1"], left["limit_2"])
        safe_open = min(right_max - right["init"], left["init"] - left_min)
        return (0, int(safe_open * self._DIALOG_RANGE_FRACTION))

    def _half_offset_range(self, name):
        cfg = self._servos[name]
        return (
            int((min(cfg["limit_1"], cfg["limit_2"]) - cfg["init"])
                * self._DIALOG_RANGE_FRACTION),
            int((max(cfg["limit_1"], cfg["limit_2"]) - cfg["init"])
                * self._DIALOG_RANGE_FRACTION),
        )

    def _pose(self, eyebrow_range):
        eye_range = self._eye_range
        eye_offset = self._rng.randint(*eye_range)
        eyebrow_offset = self._rng.randint(*eyebrow_range)
        head_yaw_offset = self._rng.randint(*self._head_yaw_range)
        targets = {
            # Equal eye offsets preserve the installed eye-pair gap.
            "eye_r": self._servos["eye_r"]["init"] + eye_offset,
            "eye_l": self._servos["eye_l"]["init"] + eye_offset,
            # The eyebrow servos are mirrored, so opening is +/- PWM.
            "eyebrow_r": self._servos["eyebrow_r"]["init"] + eyebrow_offset,
            "eyebrow_l": self._servos["eyebrow_l"]["init"] - eyebrow_offset,
            "head_yaw": self._servos["head_yaw"]["init"] + head_yaw_offset,
        }
        neck_pitch = self._rng.uniform(self._NECK_PITCH_MIN, self._NECK_PITCH_MAX)
        targets.update(self._neck_kinematics.targets(neck_pitch))
        return {
            name: self._clamp(value, self._servos[name])
            for name, value in targets.items()
        }

    def speaking_pose(self):
        return self._pose(self._eyebrow_open_range)

    @property
    def step_size(self):
        return self._STEP_SIZE
