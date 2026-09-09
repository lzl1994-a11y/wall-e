#!/usr/bin/env python3
"""Replace this process with the installed native BehaviorTree.CPP node."""

import os
import shutil
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    registry = str(root / "core" / "action_skills.json")
    explicit = os.environ.get("WALI_BEHAVIOR_TREE_EXECUTABLE", "").strip()
    if explicit:
        os.execv(explicit, [
            explicit, "--ros-args", "-p", f"skill_registry_path:={registry}"
        ])
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
                    'source "$1" && exec "$2" --ros-args -p "skill_registry_path:=$3"',
                    "wali-behavior-tree",
                    str(ros_setup),
                    executable,
                    registry,
                ],
            )
        os.execv(executable, [
            executable, "--ros-args", "-p", f"skill_registry_path:={registry}"
        ])
    ros2 = shutil.which("ros2")
    if not ros2:
        raise RuntimeError("ros2 executable not found")
    os.execv(ros2, [
        ros2, "run", "wali_behavior_tree", "behavior_tree_node",
        "--ros-args", "-p", f"skill_registry_path:={registry}",
    ])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[behavior-tree-launcher] {exc}", file=sys.stderr)
        raise SystemExit(1)
