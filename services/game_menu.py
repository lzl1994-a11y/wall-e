"""Controller-driven ROM list and raw-frame renderer for FC game mode."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np


ROM_SUFFIX = ".nes"
CJK_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    r"C:\\Windows\\Fonts\\NotoSansSC-VF.ttf",
    r"C:\\Windows\\Fonts\\simhei.ttf",
)


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
        _put_text(image, "FC 游戏列表", (12, 7), 18, (80, 218, 255))
        cv2.line(image, (10, 34), (246, 34), (130, 90, 70), 1)

        if not self.roms:
            _put_text(image, "未找到 .NES 游戏", (12, 49), 16, (100, 100, 240))
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
                _put_text(image, label, (12, y - 3), 15, color)

        _put_text(image, "方向键：选择   A：开始", (12, 213), 14, (170, 210, 130))
        return image

    def close(self) -> None:
        self._down.clear()


def _ellipsize(text: str, limit: int) -> str:
    clean = " ".join(str(text).split())
    return clean if len(clean) <= limit else clean[: max(1, limit - 1)] + "…"


def _display_name(text: str) -> str:
    """Return the ROM filename's title verbatim for the game menu."""
    return str(text).strip()


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    size: int,
    color: tuple[int, int, int],
) -> None:
    """Draw UTF-8 text; Pillow gives the TFT menu real Chinese glyphs."""
    if _put_text_pillow(image, text, origin, size, color):
        return

    # This path is only for a damaged/minimal deployment missing its CJK font.
    # Keep the menu usable rather than drawing OpenCV's invalid UTF-8 glyphs.
    import cv2

    cv2.putText(
        image,
        _ascii_fallback(text),
        (origin[0], origin[1] + size),
        cv2.FONT_HERSHEY_SIMPLEX,
        size / 40,
        color,
        1,
        cv2.LINE_AA,
    )


def _put_text_pillow(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    size: int,
    color: tuple[int, int, int],
) -> bool:
    font = _cjk_font(size)
    if font is None:
        return False
    from PIL import Image, ImageDraw

    # OpenCV frames are BGR while Pillow uses RGB.
    canvas = Image.fromarray(image[:, :, ::-1])
    ImageDraw.Draw(canvas).text(origin, text, font=font, fill=color[::-1])
    image[:] = np.asarray(canvas)[:, :, ::-1]
    return True


@lru_cache(maxsize=8)
def _cjk_font(size: int):
    try:
        from PIL import ImageFont
    except ImportError:
        return None

    configured = os.environ.get("GAME_MENU_FONT_PATH")
    candidates = ((configured,) if configured else ()) + CJK_FONT_CANDIDATES
    for filename in candidates:
        if filename and Path(filename).is_file():
            try:
                return ImageFont.truetype(filename, size)
            except OSError:
                continue
    return None


def _ascii_fallback(text: str) -> str:
    if text.isascii():
        return text
    try:
        from pypinyin import lazy_pinyin

        return " ".join(lazy_pinyin(text))
    except ImportError:
        return "".join(char if char.isascii() else "?" for char in text)


__all__ = ["GameMenu", "discover_roms"]
