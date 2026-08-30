#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
g++ -std=c++17 -O3 -pthread "$root/tools/diagnostics/fc_libretro_tft_probe.cpp" \
  $(pkg-config --cflags --libs opencv4 alsa) -ldl -o "$root/build/fc_libretro_tft_probe_cpp"
