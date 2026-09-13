#!/usr/bin/env python3
"""Replace this process with the repository-local BehaviorTree.ROS2 bridge."""

import os
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    executable = (
        root
        / "install"
        / "wali_bt_ros2_bridge"
        / "lib"
        / "wali_bt_ros2_bridge"
        / "wali_bt_ros2_bridge"
    )
    ros_setup = Path("/opt/ros") / (os.environ.get("ROS_DISTRO", "humble") or "humble") / "setup.bash"
    workspace_setup = root / "install" / "setup.bash"
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError(f"BehaviorTree.ROS2 bridge executable not found: {executable}")
    if not ros_setup.is_file() or not workspace_setup.is_file():
        raise RuntimeError("ROS or workspace setup file not found")
    os.execv(
        "/bin/bash",
        [
            "bash",
            "-c",
            'source "$1" && source "$2" && exec "$3"',
            "wali-bt-ros2-bridge",
            str(ros_setup),
            str(workspace_setup),
            str(executable),
        ],
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[behavior-tree-ros2-bridge] {exc}", file=sys.stderr)
        raise SystemExit(1)
