#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/cpp_nodes/wali_nv12_padder"
BUILD="$ROOT/build/wali_nv12_padder"

source /opt/tros/humble/setup.bash
mkdir -p "$BUILD"
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" --parallel "$(nproc)"

echo "Built: $BUILD/nv12_padder_node"
