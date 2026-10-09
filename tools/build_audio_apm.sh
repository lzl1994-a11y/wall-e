#!/usr/bin/env bash
set -eo pipefail
# Ubuntu: apt-get install --no-install-recommends libwebrtc-audio-processing-dev
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cmake -S "$ROOT/cpp_nodes/wali_audio_apm" -B "$ROOT/build/wali_audio_apm" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/build/wali_audio_apm" --parallel 1
