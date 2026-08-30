"""One controller-owned FC menu and libretro session without hardware output."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

import numpy as np

from services.fc_input import FcControllerRelay
from services.game_audio_adapter import GamePlaybackAdapter
from services.game_menu import GameMenu, discover_roms
from services.libretro_fc import LibretroFc


class FcGameSession:
    """Produce raw frames and PCM while an external adapter owns all hardware."""

    def __init__(
        self,
        *,
        core_path: str | Path,
        rom_directory: str | Path,
        controller_path: str,
        on_frame: Callable[[bytes, int, int, int], None],
        playback,
        gain: float = 0.4,
        on_game_started: Callable[[Path], None] | None = None,
        on_audio_end_queued: Callable[[], None] | None = None,
    ) -> None:
        self.core_path = str(core_path)
        self.rom_directory = Path(rom_directory)
        self.controller_path = controller_path
        self.on_frame = on_frame
        self.playback = playback
        self.gain = gain
        self.on_game_started = on_game_started
        self.on_audio_end_queued = on_audio_end_queued
        self.selected_rom: Path | None = None
        self.disconnect_error: Exception | None = None

    def run(self, stop: threading.Event) -> None:
        menu = GameMenu(discover_roms(self.rom_directory), on_frame=self._emit_menu_frame)
        relay = FcControllerRelay(self.controller_path, menu)
        audio = GamePlaybackAdapter(self.playback, gain=self.gain)
        core: LibretroFc | None = None
        try:
            relay.start()
            menu.emit()
            while not stop.wait(0.05):
                if relay.error is not None:
                    self.disconnect_error = relay.error
                    return
                if menu.chosen is not None:
                    self.selected_rom = menu.chosen
                    break
            if stop.is_set() or self.selected_rom is None:
                return

            core = LibretroFc(self.core_path, on_frame=self.on_frame, audio_sink=audio)
            relay.switch_sink(core.joypad)
            core.load(self.selected_rom)
            if self.on_game_started is not None:
                self.on_game_started(self.selected_rom)
            core.run_until(lambda: stop.is_set() or relay.error is not None)
            if relay.error is not None:
                self.disconnect_error = relay.error
        finally:
            relay.stop()
            if core is not None:
                core.close()
            if self.on_audio_end_queued is not None:
                self.on_audio_end_queued()
            audio.close()

    def _emit_menu_frame(self, bgr: np.ndarray) -> None:
        height, width = bgr.shape[:2]
        bgra = np.zeros((height, width, 4), dtype=np.uint8)
        bgra[:, :, :3] = bgr
        self.on_frame(bgra.tobytes(), width, height, width * 4)


__all__ = ["FcGameSession"]
