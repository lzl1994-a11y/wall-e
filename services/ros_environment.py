"""Conservative reuse of a previously sourced default TROS environment.

The service entry point sets the ready marker only after sourcing TROS.  The
launcher seals the environment it actually passes to each child, and children
may reuse it only while its ROS and library search settings still match.
Other launch paths keep their existing source/setup behavior.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, MutableMapping


DEFAULT_TROS_SETUP = "/opt/tros/humble/setup.bash"
TROS_ENV_READY_KEY = "WALI_TROS_ENV_READY"
TROS_ENV_SIGNATURE_KEY = "WALI_TROS_ENV_SIGNATURE"

_REQUIRED_PATH_KEYS = ("AMENT_PREFIX_PATH", "PATH", "PYTHONPATH", "LD_LIBRARY_PATH")
_REQUIRED_PREFIXES = ("/opt/tros/humble", "/opt/ros/humble")
_SEARCH_PATH_KEYS = {"PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "PKG_CONFIG_PATH"}
_ROS_VARIABLE_PREFIXES = (
    "ROS_", "RMW_", "AMENT_", "COLCON_", "CMAKE_", "HOBOT_", "TROS_",
    "CYCLONEDDS_", "FASTDDS_", "FASTRTPS_", "RCL_", "RCUTILS_", "LD_",
)


def _default_setup_exists() -> bool:
    try:
        return Path(DEFAULT_TROS_SETUP).is_file()
    except OSError:
        return False


def _valid_default_environment(env: Mapping[str, str]) -> bool:
    if env.get(TROS_ENV_READY_KEY) != DEFAULT_TROS_SETUP or not _default_setup_exists():
        return False
    for key in _REQUIRED_PATH_KEYS:
        parts = [
            part.replace("\\", "/").rstrip("/")
            for part in env.get(key, "").split(os.pathsep)
            if part
        ]
        for prefix in _REQUIRED_PREFIXES:
            if not any(part == prefix or part.startswith(prefix + "/") for part in parts):
                return False
    return True


def _environment_signature(env: Mapping[str, str]) -> str:
    keys = sorted(
        key for key in env
        if key in _SEARCH_PATH_KEYS or key.startswith(_ROS_VARIABLE_PREFIXES)
    )
    values = [(key, env[key]) for key in keys]
    encoded = json.dumps(values, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def seal_tros_env(env: MutableMapping[str, str]) -> None:
    """Seal launcher-owned child environment after its intentional changes."""
    env.pop(TROS_ENV_SIGNATURE_KEY, None)
    if _valid_default_environment(env):
        env[TROS_ENV_SIGNATURE_KEY] = _environment_signature(env)


def can_reuse_tros_env(env: Mapping[str, str], setup: str | Path) -> bool:
    """Return true only for the unchanged, trusted default TROS environment."""
    if str(setup).replace("\\", "/") != DEFAULT_TROS_SETUP:
        return False
    signature = env.get(TROS_ENV_SIGNATURE_KEY)
    return bool(
        signature
        and _valid_default_environment(env)
        and signature == _environment_signature(env)
    )
