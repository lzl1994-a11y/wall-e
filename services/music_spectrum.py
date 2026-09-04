"""Render compact spectrum bands into the TFT's existing raw-frame format."""

from __future__ import annotations

import numpy as np


def render_spectrum_frame(levels, *, size: int = 240) -> tuple[bytes, int, int, int]:
    values = np.clip(np.asarray(levels, dtype=np.float32).reshape(-1), 0.0, 1.0)
    if values.size == 0:
        values = np.zeros(20, dtype=np.float32)
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :, 0] = np.linspace(28, 6, size, dtype=np.uint8)[:, None]
    image[:, :, 2] = np.linspace(8, 28, size, dtype=np.uint8)[None, :]
    gap = max(2, size // (values.size * 5))
    bar_width = max(2, (size - 20 - gap * (values.size - 1)) // values.size)
    x = max(5, (size - (bar_width * values.size + gap * (values.size - 1))) // 2)
    baseline = size - 14
    usable_height = size - 38
    for value in values:
        height = max(2, int(value * usable_height))
        top = baseline - height
        color = (255, int(210 - value * 70), int(70 + value * 185))
        image[top:baseline + 1, x:x + bar_width] = color
        x += bar_width + gap
    bgra = np.empty((size, size, 4), dtype=np.uint8)
    bgra[:, :, :3] = image
    bgra[:, :, 3] = 0
    return bgra.tobytes(), size, size, size * 4


__all__ = ["render_spectrum_frame"]
