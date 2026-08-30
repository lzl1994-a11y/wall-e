#!/usr/bin/env python3
"""Run a bounded FC controller/display/TFT test outside the robot launcher."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.fc_input import FcControllerRelay, XTestKeySink
from services.tft_preview_server import TftPreviewServer, load_tft_preview_settings
from services.virtual_display import VirtualDisplaySettings
from services.virtual_display_bridge import VirtualDisplayTftBridge


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rom")
    parser.add_argument("--controller", default="/dev/input/event2")
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--connect-timeout", type=float, default=20.0)
    args = parser.parse_args()

    rom = Path(args.rom)
    if not rom.is_file():
        raise SystemExit(f"ROM not found: {rom}")

    settings = VirtualDisplaySettings(fps=10)
    server = TftPreviewServer(load_tft_preview_settings())
    bridge = VirtualDisplayTftBridge(server, settings=settings)
    relay = None
    server.start()
    try:
        deadline = time.monotonic() + max(0.1, args.connect_timeout)
        while not server.device_connected and time.monotonic() < deadline:
            time.sleep(0.1)
        if not server.device_connected:
            raise RuntimeError("chest TFT did not connect to the preview server")

        bridge.start([
            "fceux", "--sound", "0", "--no-config", "1",
            "--xscale", "3", "--yscale", "3", "--noframe", "1",
            str(rom),
        ])
        sink = XTestKeySink(settings.display)
        relay = FcControllerRelay(args.controller, sink)
        relay.start()
        print(
            f"FC TFT test running for {args.seconds:g}s; "
            "controller is exclusively captured. Press Ctrl-C to stop.",
            flush=True,
        )
        end = time.monotonic() + max(0.1, args.seconds)
        while time.monotonic() < end:
            if relay.error is not None:
                raise relay.error
            if not bridge.running:
                raise RuntimeError("game TFT bridge stopped unexpectedly")
            time.sleep(0.2)
        return 0
    finally:
        if relay is not None:
            relay.stop()
        bridge.stop()
        server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
