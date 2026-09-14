"""ROS-independent control primitives for visual tracking."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


TrackingBox = tuple[float, float, float]


@dataclass(frozen=True)
class MotorTarget:
    """Normalized differential-drive target for the ROS adapter to publish."""

    left: float
    right: float


@dataclass(frozen=True)
class HeadTarget:
    """Head yaw error and normalized neck pitch target."""

    x_error: float
    pitch: float


@dataclass(frozen=True)
class TrackingDecision:
    """Pure tracking result; missing targets mean that output should be held."""

    target_seen: bool
    motor: MotorTarget | None = None
    head: HeadTarget | None = None


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


class TrackingController:
    """Computes body-follow and face-gaze decisions without ROS dependencies."""

    def __init__(
        self,
        *,
        image_width: float,
        image_height: float,
        body_target_ratio: float,
        target_memory_seconds: float,
        filter_seconds: float,
        gaze_start_pitch: float,
        gaze_min_pitch: float,
        gaze_max_pitch: float,
        pitch_rate: float,
    ) -> None:
        self.image_width = image_width
        self.image_height = image_height
        self.body_target_ratio = body_target_ratio
        self.gaze_start_pitch = gaze_start_pitch
        self.gaze_min_pitch = gaze_min_pitch
        self.gaze_max_pitch = gaze_max_pitch
        self.pitch_rate = pitch_rate
        self.target_selector = TargetSelector(
            image_width=image_width,
            image_height=image_height,
            memory_seconds=target_memory_seconds,
            filter_seconds=filter_seconds,
        )
        self.chassis_yaw = PID(0.6, 0.0, 0.05)
        self.chassis_distance = PID(1.5, 0.0, 0.1)
        self.neck_pitch = PID(0.8, 0.0, 0.05)
        self.current_neck_pitch = 0.0
        self.last_horizontal_error = 0.0

    def reset(self, *, gaze: bool) -> None:
        self.target_selector.clear()
        self.current_neck_pitch = self.gaze_start_pitch if gaze else 0.0
        self.last_horizontal_error = 0.0
        self.reset_control_history()

    def reset_control_history(self) -> None:
        self.chassis_yaw.reset()
        self.chassis_distance.reset()
        self.neck_pitch.reset()

    def center_neck(self) -> None:
        self.current_neck_pitch = 0.0

    def follow_body(
        self,
        boxes: Iterable[TrackingBox],
        *,
        dt: float,
        now: float,
    ) -> TrackingDecision:
        target = self._select(boxes, kind="body", dt=dt, now=now)
        if target is None:
            return TrackingDecision(target_seen=False)

        cx, _, area_ratio = target
        x_error = -(cx - self.image_width / 2.0) / (self.image_width / 2.0)
        if abs(x_error) > 0.05:
            self.last_horizontal_error = x_error
        distance_error = self.body_target_ratio - area_ratio

        if abs(x_error) < 0.05:
            x_error = 0.0
        if abs(distance_error) < 0.05:
            distance_error = 0.0

        yaw = self.chassis_yaw.update(x_error, dt)
        distance = self.chassis_distance.update(distance_error, dt)
        distance *= max(0.0, 1.0 - abs(x_error) / 0.6)
        return TrackingDecision(
            target_seen=True,
            motor=MotorTarget(
                left=max(min(distance + yaw, 1.0), -1.0),
                right=max(min(distance - yaw, 1.0), -1.0),
            ),
            head=HeadTarget(x_error=x_error, pitch=0.0),
        )

    def gaze_at_face(
        self,
        face_boxes: Iterable[TrackingBox],
        body_boxes: Iterable[TrackingBox],
        *,
        dt: float,
        now: float,
    ) -> TrackingDecision:
        stop = MotorTarget(left=0.0, right=0.0)
        face = self._select(face_boxes, kind="face", dt=dt, now=now)
        if face is None:
            # A torso centre is not a face aim point. Keep pitch stable while
            # still using its horizontal position for a natural head turn.
            body = self._select(body_boxes, kind="gaze_body", dt=dt, now=now)
            self.neck_pitch.reset()
            if body is None:
                return TrackingDecision(target_seen=False, motor=stop)
            return TrackingDecision(
                target_seen=True,
                motor=stop,
                head=HeadTarget(
                    x_error=self._horizontal_error(body[0]),
                    pitch=self.current_neck_pitch,
                ),
            )

        cx, cy, _ = face
        x_error = self._horizontal_error(cx)
        y_error = (cy - self.image_height / 2.0) / (self.image_height / 2.0)
        if abs(x_error) < 0.05:
            x_error = 0.0
        if abs(y_error) < 0.08:
            self.neck_pitch.reset()
            pitch_output = 0.0
        else:
            pitch_output = self.neck_pitch.update(y_error, dt)
        self.current_neck_pitch = max(
            self.gaze_min_pitch,
            min(
                self.gaze_max_pitch,
                self.current_neck_pitch - pitch_output * self.pitch_rate * dt,
            ),
        )
        return TrackingDecision(
            target_seen=True,
            motor=stop,
            head=HeadTarget(x_error=x_error, pitch=self.current_neck_pitch),
        )

    def _select(
        self,
        boxes: Iterable[TrackingBox],
        *,
        kind: str,
        dt: float,
        now: float,
    ) -> TrackingBox | None:
        candidates = list(boxes)
        if not candidates:
            return None
        target, reacquired = self.target_selector.select(
            candidates,
            kind=kind,
            dt=dt,
            now=now,
        )
        if reacquired:
            self.reset_control_history()
        return target

    def _horizontal_error(self, cx: float) -> float:
        # The source image is horizontally flipped by the vision pipeline.
        return -(cx - self.image_width / 2.0) / (self.image_width / 2.0)
