"""Pure service for camera preview classification and photo save decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PhotoCaptureDecision:
    answer: str | None = None
    error: str | None = None
    frame: bytes | None = None
    should_save: bool = False

    @property
    def is_ready_to_save(self) -> bool:
        return self.should_save and self.frame is not None

    @property
    def is_complete(self) -> bool:
        return not self.should_save and self.answer is not None


def decide_photo_preview(preview: Any) -> PhotoCaptureDecision:
    """Classify camera preview result into busy, missing frame, or ready to save.

    Attributes on preview (busy, last_frame, error) are accessed directly to
    preserve strict contract validation.
    """
    if preview.busy:
        return PhotoCaptureDecision(
            answer="我正在拍上一张，等一下再试。",
            error="camera_preview_busy",
            frame=None,
            should_save=False,
        )
    if not preview.last_frame:
        return PhotoCaptureDecision(
            answer="这次没有拍到，检查一下摄像头连接。",
            error=preview.error or "camera_frame_unavailable",
            frame=None,
            should_save=False,
        )
    return PhotoCaptureDecision(
        answer=None,
        error=None,
        frame=preview.last_frame,
        should_save=True,
    )


def photo_save_succeeded() -> PhotoCaptureDecision:
    """Decision state when photo was successfully saved to disk."""
    return PhotoCaptureDecision(
        answer="拍好了，照片已经保存。",
        error=None,
        frame=None,
        should_save=False,
    )


def photo_save_failed(error: Any) -> PhotoCaptureDecision:
    """Decision state when photo saving encountered an error."""
    return PhotoCaptureDecision(
        answer="画面拍到了，但照片保存失败了。",
        error=str(error) if error is not None else None,
        frame=None,
        should_save=False,
    )


__all__ = [
    "PhotoCaptureDecision",
    "decide_photo_preview",
    "photo_save_failed",
    "photo_save_succeeded",
]
