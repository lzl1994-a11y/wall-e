"""Small UTF-8 text helper shared by host-rendered TFT surfaces."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np


_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    r"C:\Windows\Fonts\NotoSansSC-VF.ttf",
    r"C:\Windows\Fonts\simhei.ttf",
)


def draw_text(image, text, origin, size, color) -> None:
    font = _font(size)
    if font is not None:
        from PIL import Image, ImageDraw

        canvas = Image.fromarray(image[:, :, ::-1])
        ImageDraw.Draw(canvas).text(origin, str(text), font=font, fill=color[::-1])
        image[:] = np.asarray(canvas)[:, :, ::-1]
        return

    import cv2

    target = image if image.flags.c_contiguous else np.ascontiguousarray(image)
    cv2.putText(
        target, _ascii_fallback(str(text)), (origin[0], origin[1] + size),
        cv2.FONT_HERSHEY_SIMPLEX, size / 40, color, 1, cv2.LINE_AA,
    )
    if target is not image:
        image[:] = target


@lru_cache(maxsize=12)
def _font(size: int):
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    configured = os.environ.get("WALI_TFT_FONT_PATH") or os.environ.get(
        "GAME_MENU_FONT_PATH"
    )
    for filename in ((configured,) if configured else ()) + _FONT_CANDIDATES:
        if filename and Path(filename).is_file():
            try:
                return ImageFont.truetype(filename, size)
            except OSError:
                pass
    return None


def _ascii_fallback(text: str) -> str:
    if text.isascii():
        return text
    try:
        from pypinyin import lazy_pinyin

        return " ".join(lazy_pinyin(text))
    except ImportError:
        return "".join(char if char.isascii() else "?" for char in text)


__all__ = ["draw_text"]
