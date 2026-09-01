"""Configuration-driven mappings for coupled robot servos."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "core" / "config.yaml"


@dataclass(frozen=True)
class ServoCalibration:
    """Validated numeric range and neutral position for one servo."""

    initial: int
    low: int
    high: int


@dataclass(frozen=True)
class NeckKinematics:
    """Map normalized pitch onto WALL-E's mechanically coupled neck servos."""

    top: ServoCalibration
    bottom: ServoCalibration

    @staticmethod
    def _toward(start: int, end: int, amount: float) -> int:
        return int(round(start + (end - start) * amount))

    def targets(self, pitch: float) -> dict[str, int]:
        """Return neck targets for ``pitch`` in [-1, 1].

        Positive pitch raises the telescopic lower neck while leaving the top
        joint neutral. Negative pitch lowers the lower neck and increases the
        top joint, matching the linkage installed on this robot.
        """
        pitch = max(-1.0, min(1.0, float(pitch)))
        if pitch >= 0.0:
            return {
                "neck_top": self.top.initial,
                "neck_bottom": self._toward(
                    self.bottom.initial, self.bottom.high, pitch
                ),
            }

        amount = -pitch
        return {
            "neck_top": self._toward(self.top.initial, self.top.high, amount),
            "neck_bottom": self._toward(
                self.bottom.initial, self.bottom.low, amount
            ),
        }


def _calibration(item: Any, name: str) -> ServoCalibration:
    if not isinstance(item, dict):
        raise ValueError(f"舵机配置缺失: {name}")
    try:
        initial = int(item["init"])
        limit_1 = int(item["limit_1"])
        limit_2 = int(item["limit_2"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"舵机配置不完整: {name}") from exc

    low, high = sorted((limit_1, limit_2))
    if not low <= initial <= high:
        raise ValueError(f"舵机初始位置超出限位: {name}")
    return ServoCalibration(initial=initial, low=low, high=high)


def resolve_servo_target(servo: Any, target: Any) -> int | None:
    """Resolve a numeric or calibration-relative pose target.

    Relative targets use ``{"toward": "min|max", "fraction": 0..1}`` and
    are measured from the servo's configured initial position.
    """
    try:
        calibration = _calibration(
            servo, str(servo.get("name", "unknown")) if isinstance(servo, dict) else "unknown"
        )
    except ValueError:
        return None

    symbolic = {
        "init": calibration.initial,
        "min": calibration.low,
        "max": calibration.high,
    }
    if isinstance(target, str):
        target = symbolic.get(target.strip().lower())
    elif isinstance(target, dict):
        destination = symbolic.get(str(target.get("toward", "")).strip().lower())
        fraction = target.get("fraction")
        if (
            destination is None
            or isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not 0.0 <= float(fraction) <= 1.0
        ):
            return None
        target = calibration.initial + (
            destination - calibration.initial
        ) * float(fraction)

    try:
        target = float(target)
    except (TypeError, ValueError):
        return None
    target = max(calibration.low, min(calibration.high, target))
    return int(round(target))


def load_neck_kinematics(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> NeckKinematics:
    """Load and validate neck calibration from ``core/config.yaml``."""
    try:
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"无法读取舵机配置: {exc}") from exc
    if not isinstance(config, dict):
        raise RuntimeError("舵机配置根节点必须是对象")

    servos = {
        item.get("name"): item
        for item in config.get("servos", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    try:
        return neck_kinematics_from_servos(servos)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def neck_kinematics_from_servos(servos: Any) -> NeckKinematics:
    """Build neck kinematics from an already loaded name-to-config mapping."""
    if not isinstance(servos, dict):
        raise ValueError("舵机配置映射无效")
    return NeckKinematics(
        top=_calibration(servos.get("neck_top"), "neck_top"),
        bottom=_calibration(servos.get("neck_bottom"), "neck_bottom"),
    )
