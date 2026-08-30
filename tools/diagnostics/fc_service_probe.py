#!/usr/bin/env python3
"""FC output diagnostic using the existing TFT and audio playback services."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.fc_input import FcControllerRelay
from services.game_audio_adapter import GamePlaybackAdapter
from services.game_tft_stream import GameTftStreamServer
from services.libretro_fc import LibretroFc
from services.playback_service import PlaybackService
from services.tft_preview_server import load_tft_preview_settings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rom")
    parser.add_argument("--core", default="/root/libretro-fceumm/fceumm_libretro.so")
    parser.add_argument("--controller", default="/dev/input/event2")
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--gain", type=float, default=0.4)
    parser.add_argument("--audio-drain-timeout", type=float, default=15.0)
    args = parser.parse_args()

    server = GameTftStreamServer(load_tft_preview_settings())
    audio_finished = threading.Event()
    player = PlaybackService(mode="game", on_turn_complete=audio_finished.set)
    audio = GamePlaybackAdapter(player, gain=args.gain)
    stream = relay = None
    last_sent = 0.0
    source_frames = source_skips = 0

    def on_frame(raw: bytes, width: int, height: int, pitch: int) -> None:
        nonlocal last_sent, source_frames, source_skips
        source_frames += 1
        now = time.monotonic()
        if now - last_sent < 1.0 / max(1.0, args.fps):
            source_skips += 1
            return
        image = np.frombuffer(raw, dtype=np.uint8).reshape(height, pitch // 4, 4)
        if stream is not None and stream.send_bgr(image[:, :width, :3]):
            last_sent = now

    server.start()
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
        if stream is not None:
            print(
                "FC frame metrics:"
                f" callbacks={source_frames}"
                f" gated={source_skips}"
                f" sent={stream._frame_index}"
                f" tft_prepare_s={stream.prepare_seconds:.3f}"
                f" tcp_send_s={stream.send_seconds:.3f}"
            )
        return 0
    finally:
        if relay is not None:
            relay.stop()
        core.close()
        audio.close()
        drain_started = time.monotonic()
        drained = audio_finished.wait(timeout=max(0.0, args.audio_drain_timeout))
        print(
            "FC audio metrics:"
            f" batches={audio.batches_submitted}"
            f" frames={audio.frames_submitted}"
            f" drain_s={time.monotonic() - drain_started:.3f}"
            f" drained={drained}"
        )
        if not drained:
            print("FC audio warning: playback did not finish before drain timeout")
        if stream is not None:
            stream.close()
        server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
