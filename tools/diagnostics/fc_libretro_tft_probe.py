#!/usr/bin/env python3
"""Direct FC-core-to-TFT diagnostic: no X11, Xvfb, ffmpeg, or screenshots."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.fc_input import FcControllerRelay
from services.libretro_audio import LibretroAudioPlayer
from services.game_tft_stream import GameTftStreamServer
from services.libretro_fc import LibretroFc
from services.tft_preview_server import load_tft_preview_settings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rom")
    parser.add_argument("--core", default="/root/libretro-fceumm/fceumm_libretro.so")
    parser.add_argument("--controller", default="/dev/input/event2")
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--audio", action="store_true", help="play FC PCM on the USB speaker")
    parser.add_argument("--audio-device", type=int, help="sounddevice output index override")
    args = parser.parse_args()

    server = GameTftStreamServer(load_tft_preview_settings())
    stream = None
    relay = None
    audio = None
    last_sent = 0.0

    def on_frame(raw: bytes, width: int, height: int, pitch: int) -> None:
        nonlocal last_sent
        now = time.monotonic()
        if now - last_sent < 1.0 / max(1.0, args.fps):
            return
        image = np.frombuffer(raw, dtype=np.uint8).reshape(height, pitch // 4, 4)
        ok, jpeg = cv2.imencode(".jpg", image[:, :width, :3], [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok and stream is not None and stream.send_jpeg(jpeg.tobytes()):
            last_sent = now

    server.start()
    if args.audio:
        audio = LibretroAudioPlayer(device=args.audio_device)
        audio.start()
        print(f"FC audio output enabled (sounddevice device {audio.device})")
    core = LibretroFc(args.core, on_frame=on_frame, audio_sink=audio)
    try:
        deadline = time.monotonic() + 20.0
        while not server.device_connected and time.monotonic() < deadline:
            time.sleep(0.1)
        if not server.device_connected:
            raise RuntimeError("chest TFT did not connect")
        stream = server.open_jpeg_stream(fps=int(args.fps))
        if stream is None:
            raise RuntimeError("chest TFT stream unavailable")
        relay = FcControllerRelay(args.controller, core.joypad)
        relay.start()
        core.load(args.rom)
        core.run_for(args.seconds)
        if relay.error is not None:
            raise relay.error
        return 0
    finally:
        if relay is not None:
            relay.stop()
        core.close()
        if audio is not None:
            audio.close()
        if stream is not None:
            stream.close()
        server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
