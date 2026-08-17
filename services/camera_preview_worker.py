#!/usr/bin/env python3
"""Isolated OpenCV worker for the configuration-page camera preview."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--fps", type=float, default=8.0)
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        emit({"type": "error", "error": "当前 Python 环境未安装 OpenCV，无法启动摄像头预览"})
        return 2

    capture = None
    try:
        emit({"type": "status", "phase": "opening"})
        backend = getattr(cv2, "CAP_V4L2", None) if args.device.startswith("/dev/video") else None
        capture = cv2.VideoCapture(args.device, backend) if backend is not None else cv2.VideoCapture(args.device)
        if not capture.isOpened():
            raise RuntimeError(f"无法打开摄像头 {args.device}，设备可能正被视觉跟踪占用")

        fourcc_property = getattr(cv2, "CAP_PROP_FOURCC", None)
        fourcc_factory = getattr(cv2, "VideoWriter_fourcc", None)
        if fourcc_property is not None and fourcc_factory is not None:
            capture.set(fourcc_property, fourcc_factory(*"MJPG"))
        for prop_name, value in (
            ("CAP_PROP_FRAME_WIDTH", 640),
            ("CAP_PROP_FRAME_HEIGHT", 480),
            ("CAP_PROP_FPS", args.fps),
            ("CAP_PROP_BUFFERSIZE", 1),
        ):
            prop = getattr(cv2, prop_name, None)
            if prop is not None:
                capture.set(prop, value)

        emit({"type": "status", "phase": "waiting_frame"})
        frame_interval = 1.0 / max(1.0, args.fps)
        sample_started = time.monotonic()
        sample_frames = 0
        measured_fps = 0.0
        failed_reads = 0
        while True:
            frame_started = time.monotonic()
            ok, image = capture.read()
            if not ok or image is None:
                failed_reads += 1
                if failed_reads >= 20:
                    raise RuntimeError("摄像头连续读帧失败")
                time.sleep(0.05)
                continue

            failed_reads = 0
            encoded_ok, encoded = cv2.imencode(
                ".jpg",
                image,
                [int(cv2.IMWRITE_JPEG_QUALITY), 82],
            )
            if not encoded_ok:
                continue

            sample_frames += 1
            elapsed = time.monotonic() - sample_started
            if elapsed >= 1.0:
                measured_fps = sample_frames / elapsed
                sample_started = time.monotonic()
                sample_frames = 0
            height, width = image.shape[:2]
            emit({
                "type": "frame",
                "jpeg": base64.b64encode(encoded.tobytes()).decode("ascii"),
                "width": int(width),
                "height": int(height),
                "fps": measured_fps,
            })

            remaining = frame_interval - (time.monotonic() - frame_started)
            if remaining > 0:
                time.sleep(remaining)
    except BrokenPipeError:
        return 0
    except Exception as exc:
        try:
            emit({"type": "error", "error": str(exc)})
        except BrokenPipeError:
            pass
        return 1
    finally:
        if capture is not None:
            capture.release()


if __name__ == "__main__":
    raise SystemExit(main())
