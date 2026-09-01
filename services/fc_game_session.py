"""One controller-owned FC menu and libretro session without hardware output."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

import numpy as np

from services.fc_input import FcControllerRelay
from services.game_audio_adapter import GamePlaybackAdapter
from services.game_hotkey import ButtonHold
from services.game_menu import GameMenu, discover_roms
from services.libretro_fc import LibretroFc


class _ReturnToMenuSink:
    """Relay gameplay input and recognise a long FC-A press as Back to menu."""

    def __init__(self, sink, *, hold_seconds: float = 2.0) -> None:
        self._sink = sink
        self._return_hold = ButtonHold(hold_seconds=hold_seconds)

    def set_key(self, key: str, down: bool) -> None:
        self._sink.set_key(key, down)
        if key == "F":  # FC A / the menu confirmation button
            self._return_hold.set_down(down)

    def should_return_to_menu(self) -> bool:
        return self._return_hold.poll()

    def close(self) -> None:
        self._sink.close()


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
            while not stop.wait(0.05):
                if relay.error is not None:
                    self.disconnect_error = relay.error
                    return
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
                gameplay_sink = _ReturnToMenuSink(core.joypad)
                relay.switch_sink(gameplay_sink)
                core.load(self.selected_rom)
                if self.on_game_started is not None:
                    self.on_game_started(self.selected_rom)
                core.run_until(
                    lambda: stop.is_set()
                    or relay.error is not None
                    or gameplay_sink.should_return_to_menu()
                )
                if relay.error is not None:
                    self.disconnect_error = relay.error
                    return
                if stop.is_set():
                    return
                core.close()
                core = None
                # The held A press that returned here does not select anything:
                # GameMenu sees only its eventual release event.
                menu = GameMenu(discover_roms(self.rom_directory), on_frame=self._emit_menu_frame)
                relay.switch_sink(menu)
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
