#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_DISTRO_NAME="${ROS_DISTRO:-humble}"
ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"

if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "ROS setup not found: ${ROS_SETUP}" >&2
  exit 1
fi
if [[ ! -d "${ROOT_DIR}/cpp_nodes/BehaviorTree.ROS2/behaviortree_ros2" ]]; then
  echo "BehaviorTree.ROS2 submodule is missing; run git submodule update --init" >&2
  exit 1
fi

source "${ROS_SETUP}"
set -u
cd "${ROOT_DIR}"
colcon build \
  --base-paths \
    "${ROOT_DIR}/cpp_nodes/BehaviorTree.ROS2" \
    "${ROOT_DIR}/cpp_nodes/wali_bt_ros2_bridge" \
  --packages-select btcpp_ros2_interfaces behaviortree_ros2 wali_bt_ros2_bridge \
  --executor sequential \
  --parallel-workers 1 \
  --cmake-args \
    "-Dbehaviortree_cpp_DIR=${ROOT_DIR}/cmake/behaviortree_cpp"
