"""Render a cached neon music visualizer into the TFT raw-frame format."""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from services.tft_text import draw_text


def _track_lines(track: str) -> tuple[str, str]:
    clean = " ".join(str(track or "LOCAL AUDIO").replace("_", " ").split())
    for separator in ("+-+", " - ", " — ", " – "):
        if separator in clean:
            artist, title = clean.rsplit(separator, 1)
            return artist.strip()[:25], title.strip()[:18]
    return "NOW PLAYING", clean[:22]


@lru_cache(maxsize=16)
def _base_frame(size: int, track: str) -> np.ndarray:
    image = np.zeros((size, size, 4), dtype=np.uint8)
    y = np.arange(size, dtype=np.float32)[:, None]
    x = np.arange(size, dtype=np.float32)[None, :]
    glow = np.clip(
        1.0 - np.sqrt((x - size * 0.52) ** 2 + (y - size * 0.43) ** 2) / size,
        0,
        1,
    )
    image[:, :, 0] = (11 + glow * 32).astype(np.uint8)
    image[:, :, 1] = (5 + glow * 9).astype(np.uint8)
    image[:, :, 2] = (18 + glow * 20).astype(np.uint8)
    image[:, :, 3] = 255

    artist, title = _track_lines(track)
    draw_text(image[:, :, :3], artist, (11, 7), 11, (235, 145, 94))
    draw_text(image[:, :, :3], title, (11, 21), 18, (255, 245, 245))
    cv2.line(image, (10, 48), (229, 48), (75, 30, 85, 255), 1, cv2.LINE_AA)
    for grid_y in (78, 108, 138, 168):
        cv2.line(image, (10, grid_y), (229, grid_y), (35, 18, 47, 255), 1)
    for label_x, label in ((12, "60"), (58, "250"), (108, "1K"), (157, "4K"), (202, "16K")):
        draw_text(image[:, :, :3], label, (label_x, 215), 10, (140, 116, 150))
    draw_text(image[:, :, :3], "Hz", (220, 215), 9, (100, 85, 115))
    return image


def render_spectrum_frame(
    levels, *, title: str = "", size: int = 240
) -> tuple[bytes, int, int, int]:
    values = np.clip(np.asarray(levels, dtype=np.float32).reshape(-1), 0.0, 1.0)
    if values.size == 0:
        values = np.zeros(20, dtype=np.float32)
    image = _base_frame(size, str(title)).copy()
    left, right, baseline, top_limit = 10, size - 10, size - 35, 57
    gap = max(2, size // (values.size * 4))
    bar_width = max(2, (right - left - gap * (values.size - 1)) // values.size)
    span = bar_width * values.size + gap * (values.size - 1)
    positions = left + (right - left - span) // 2 + np.arange(values.size) * (bar_width + gap)
    tops = baseline - np.maximum(2, (values * (baseline - top_limit)).astype(int))

    points = []
    low = np.array((250, 70, 160), dtype=np.float32)
    high = np.array((80, 245, 255), dtype=np.float32)
    for x0, top, value in zip(positions, tops, values):
        x0, top = int(x0), int(top)
        height = baseline - top
        gradient = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
        image[top:baseline, x0:x0 + bar_width, :3] = (
            low + gradient * (high - low)
        ).astype(np.uint8)
        image[top:top + 2, x0:x0 + bar_width, :3] = (255, 255, 255)
        reflection = max(1, int(height * 0.12))
        image[baseline + 2:baseline + 2 + reflection, x0:x0 + bar_width, :3] = (
            np.array((120, 28, 72)) * float(value)
        ).astype(np.uint8)
        points.append((x0 + bar_width // 2, top))

    if len(points) > 1:
        cv2.polylines(
            image, [np.asarray(points, dtype=np.int32)], False,
            (255, 90, 245, 255), 1, cv2.LINE_AA,
        )
    cv2.line(
        image, (left, baseline), (right, baseline),
        (255, 105, 185, 255), 1, cv2.LINE_AA,
    )
    return image.tobytes(), size, size, size * 4


__all__ = ["render_spectrum_frame"]
