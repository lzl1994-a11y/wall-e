"""Configuration loader for joystick-driven servo and motor updates."""

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "core" / "config.yaml"
DEFAULT_SERVO_STEP_SIZE = 40.0
DEFAULT_UPDATE_RATE_HZ = 20


def _number_in_range(value: Any, minimum: float, maximum: float, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value) if minimum <= value <= maximum else default


def load_remote_control_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, float]:
    """Load validated remote-control settings, falling back safely on bad files."""
    try:
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        config = {}

    remote = config.get("remote_control", {}) if isinstance(config, dict) else {}
    if not isinstance(remote, dict):
        remote = {}

    return {
        "servo_step_size": _number_in_range(
            remote.get("servo_step_size"), 0.1, 65535, DEFAULT_SERVO_STEP_SIZE
        ),
        "update_rate_hz": _number_in_range(
            remote.get("update_rate_hz"), 1, 100, DEFAULT_UPDATE_RATE_HZ
        ),
    }
