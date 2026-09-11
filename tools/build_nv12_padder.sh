#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/cpp_nodes/wali_nv12_padder"
BUILD="$ROOT/build/wali_nv12_padder"

# TROS/ROS setup scripts may read unset variables internally, so do not use
# `set -u` while sourcing them.
source /opt/tros/humble/setup.bash
mkdir -p "$BUILD"
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
BUILD_JOBS="${WALI_BUILD_JOBS:-1}"
case "$BUILD_JOBS" in
  ''|*[!0-9]*) echo "WALI_BUILD_JOBS must be an integer" >&2; exit 2 ;;
esac
if [ "$BUILD_JOBS" -lt 1 ]; then
  echo "WALI_BUILD_JOBS must be at least 1" >&2
  exit 2
fi
# The deployed X3 host has 2 GB RAM and no swap. Keep even explicit overrides
# bounded so a native compile cannot invoke the kernel OOM killer.
if [ "$BUILD_JOBS" -gt 2 ]; then BUILD_JOBS=2; fi
cmake --build "$BUILD" --parallel "$BUILD_JOBS"

echo "Built: $BUILD/nv12_padder_node"
