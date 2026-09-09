#!/usr/bin/env python3
"""Replace this process with the installed native BehaviorTree.CPP node."""

import os
import shutil
import sys
from pathlib import Path


def main():
    explicit = os.environ.get("WALI_BEHAVIOR_TREE_EXECUTABLE", "").strip()
    if explicit:
        os.execv(explicit, [explicit])
    root = Path(__file__).resolve().parent.parent
    local_binary = (
        root
        / "install"
        / "wali_behavior_tree"
        / "lib"
        / "wali_behavior_tree"
        / "behavior_tree_node"
    )
    if local_binary.is_file() and os.access(local_binary, os.X_OK):
        executable = str(local_binary)
        ros_distro = os.environ.get("ROS_DISTRO", "humble").strip() or "humble"
        ros_setup = Path("/opt/ros") / ros_distro / "setup.bash"
        if ros_setup.is_file():
            os.execv(
                "/bin/bash",
                [
                    "bash",
                    "-c",
                    'source "$1" && exec "$2"',
                    "wali-behavior-tree",
                    str(ros_setup),
                    executable,
                ],
            )
        os.execv(executable, [executable])
    ros2 = shutil.which("ros2")
    if not ros2:
        raise RuntimeError("ros2 executable not found")
    os.execv(ros2, [ros2, "run", "wali_behavior_tree", "behavior_tree_node"])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[behavior-tree-launcher] {exc}", file=sys.stderr)
        raise SystemExit(1)
