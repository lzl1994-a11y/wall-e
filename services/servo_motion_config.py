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
        return NeckKinematics(
            top=_calibration(servos.get("neck_top"), "neck_top"),
            bottom=_calibration(servos.get("neck_bottom"), "neck_bottom"),
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
