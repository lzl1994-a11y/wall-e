#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cmake -S "$ROOT/cpp_nodes/wali_webrtc_codec" -B "$ROOT/build/wali_webrtc_codec" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/build/wali_webrtc_codec" --parallel 1
