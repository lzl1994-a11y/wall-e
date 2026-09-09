#!/usr/bin/env python3
"""Replace this process with the installed native BehaviorTree.CPP node."""

import os
import shutil
import sys


def main():
    explicit = os.environ.get("WALI_BEHAVIOR_TREE_EXECUTABLE", "").strip()
    if explicit:
        os.execv(explicit, [explicit])
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
