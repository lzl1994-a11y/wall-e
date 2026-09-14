"""ROS-independent control primitives for visual tracking."""

from __future__ import annotations

import math
from collections.abc import Iterable


TrackingBox = tuple[float, float, float]


class PID:
    """Small bounded PID controller used by chassis and neck tracking."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        out_min: float = -1.0,
        out_max: float = 1.0,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        self.integral = 0.0
        self.prev_error: float | None = None

    def update(self, error: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0
        self.integral += error * dt
        derivative = (
            0.0
            if self.prev_error is None
            else (error - self.prev_error) / dt
        )
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return max(min(output, self.out_max), self.out_min)

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = None


class TargetSelector:
    """Associates detections geometrically across jitter and short occlusions.

    This deliberately does not identify people. Once the memory expires, the
    largest visible detection becomes the new target.
    """

    def __init__(
        self,
        *,
        image_width: float,
        image_height: float,
        memory_seconds: float,
        filter_seconds: float,
    ) -> None:
        self.image_width = image_width
        self.image_height = image_height
        self.memory_seconds = memory_seconds
        self.filter_seconds = filter_seconds
        self._tracks: dict[str, tuple[TrackingBox, float]] = {}

    def clear(self) -> None:
        self._tracks.clear()

    def select(
        self,
        boxes: Iterable[TrackingBox],
        *,
        kind: str,
        dt: float,
        now: float,
    ) -> tuple[TrackingBox | None, bool]:
        candidates_input = list(boxes)
        if not candidates_input:
            return None, False

        previous = self._tracks.get(kind)
        reacquired = previous is None or now - previous[1] > self.memory_seconds
        if reacquired:
            best = self.largest_box(candidates_input)
        else:
            old = previous[0]

            def distance(box: TrackingBox) -> float:
                return math.hypot(
                    (box[0] - old[0]) / self.image_width,
                    (box[1] - old[1]) / self.image_height,
                )

            nearby = [
                box
                for box in candidates_input
                if distance(box) <= 0.30
                and 0.25 <= box[2] / max(old[2], 1e-6) <= 4.0
            ]
            if not nearby:
                return None, False
            observed = min(nearby, key=distance)
            alpha = 1.0 - math.exp(-dt / self.filter_seconds)
            best = tuple(
                old_value + alpha * (new_value - old_value)
                for old_value, new_value in zip(old, observed)
            )

        if best is not None:
            self._tracks[kind] = (best, now)
        return best, reacquired

    @staticmethod
    def largest_box(boxes: Iterable[TrackingBox]) -> TrackingBox | None:
        return max(boxes, key=lambda box: box[2], default=None)
