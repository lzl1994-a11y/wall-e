"""Controller-driven ROM list and raw-frame renderer for FC game mode."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np


ROM_SUFFIX = ".nes"


def discover_roms(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.iterdir() if path.is_file() and path.suffix.lower() == ROM_SUFFIX),
        key=lambda path: path.name.casefold(),
    )


class GameMenu:
    """Small menu whose input names match :class:`LibretroJoypad`."""

    PAGE_SIZE = 7

    def __init__(
        self,
        roms: list[Path],
        *,
        on_frame: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        self.roms = list(roms)
        self.selected = 0
        self.chosen: Path | None = None
        self._on_frame = on_frame
        self._down: set[str] = set()

    def set_key(self, key: str, down: bool) -> None:
        if down:
            if key in self._down:
                return
            self._down.add(key)
            if key == "KP_8":
                self.move(-1)
            elif key == "KP_2":
                self.move(1)
            elif key in {"F", "Return"}:
                self.choose()
            return
        self._down.discard(key)

    def move(self, offset: int) -> None:
        if not self.roms:
            return
        self.selected = (self.selected + int(offset)) % len(self.roms)
        self.emit()

    def choose(self) -> Path | None:
        if not self.roms:
            return None
        self.chosen = self.roms[self.selected]
        return self.chosen

    def emit(self) -> np.ndarray:
        frame = self.render()
        if self._on_frame is not None:
            self._on_frame(frame)
        return frame

    def render(self) -> np.ndarray:
        import cv2

        image = np.zeros((240, 256, 3), dtype=np.uint8)
        image[:] = (18, 8, 5)
        cv2.putText(image, "FC GAME LIST", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 218, 255), 1, cv2.LINE_AA)
        cv2.line(image, (10, 34), (246, 34), (130, 90, 70), 1)

        if not self.roms:
            cv2.putText(image, "NO .NES ROM FOUND", (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 240), 1, cv2.LINE_AA)
        else:
            page_start = (self.selected // self.PAGE_SIZE) * self.PAGE_SIZE
            for row, rom in enumerate(self.roms[page_start:page_start + self.PAGE_SIZE]):
                index = page_start + row
                y = 43 + row * 24
                active = index == self.selected
                if active:
                    cv2.rectangle(image, (7, y - 5), (249, y + 15), (132, 78, 32), -1)
                label = f"{index + 1:02d}  {_ellipsize(_display_name(rom.stem), 28)}"
                color = (255, 255, 255) if active else (210, 190, 180)
                cv2.putText(image, label, (12, y + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

        cv2.putText(image, "DPAD: SELECT   A: START", (12, 226), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 210, 130), 1, cv2.LINE_AA)
        return image

    def close(self) -> None:
        self._down.clear()


def _ellipsize(text: str, limit: int) -> str:
    clean = " ".join(str(text).split())
    return clean if len(clean) <= limit else clean[: max(1, limit - 1)] + "…"


def _display_name(text: str) -> str:
    if str(text).isascii():
        return str(text)
    try:
        from pypinyin import lazy_pinyin

        return " ".join(lazy_pinyin(str(text)))
    except ImportError:
        return "".join(char if char.isascii() else "?" for char in str(text))


__all__ = ["GameMenu", "discover_roms"]
