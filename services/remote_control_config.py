"""Configuration loader for joystick-driven servo and motor updates."""

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "core" / "config.yaml"
DEFAULT_SERVO_STEP_SIZE = 40.0
DEFAULT_UPDATE_RATE_HZ = 20
_UNSET = object()


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


class RemoteControlConfigWatcher:
    """Check the config file on every control tick and parse only after a change."""

    def __init__(self, config_path: Path | str = DEFAULT_CONFIG_PATH):
        self.path = Path(config_path)
        self._signature: object = _UNSET

    def _file_signature(self) -> tuple[int, int] | None:
        try:
            stat = self.path.stat()
        except OSError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def load_if_changed(self) -> dict[str, float] | None:
        """Return fresh settings after a file change, otherwise return ``None``."""
        signature = self._file_signature()
        if signature == self._signature:
            return None
        settings = load_remote_control_config(self.path)
        self._signature = signature
        return settings
