#!/usr/bin/env python3
"""Launch one GUI app on Xvfb and report whether frames can be captured."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.virtual_display import MjpegFrameSource, VirtualDisplay, VirtualDisplaySettings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rom")
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()

    settings = VirtualDisplaySettings()
    display = VirtualDisplay(settings)
    source = MjpegFrameSource(settings)
    try:
        display.start()
        source.start()
        display.launch(
            [
                "fceux", "--sound", "0", "--no-config", "1",
                "--xscale", "3", "--yscale", "3", "--noframe", "1",
                args.rom,
            ]
        )
        deadline = time.monotonic() + max(0.1, args.seconds)
        sequence = 0
        size = 0
        while time.monotonic() < deadline:
            sequence, frame = source.latest
            size = len(frame or b"")
            if sequence:
                break
            time.sleep(0.05)
        if not sequence:
            raise RuntimeError("no virtual-display frame captured")
        print(f"captured sequence={sequence} jpeg_bytes={size}")
        return 0
    finally:
        source.stop()
        display.stop()


if __name__ == "__main__":
    raise SystemExit(main())
