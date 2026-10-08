"""Bounded handoff from the media loop to the node's existing ROS executor."""
from collections import deque
import threading


class RemoteAudioPublisher:
    MAX_PENDING_FRAMES = 16

    def __init__(self, node, publisher):
        self._publisher = publisher
        self._pending = deque()
        self._lock = threading.Lock()
        self._armed = False
        self.peak_pending_frames = 0
        self._guard = node.create_guard_condition(self._flush)

    def publish(self, message):
        """Transfer ownership of a fresh message; do not mutate it afterwards."""
        with self._lock:
            if len(self._pending) >= self.MAX_PENDING_FRAMES:
                # Fail the audio direction explicitly through its existing
                # ValueError handler instead of silently losing syllables.
                raise ValueError("ROS remote audio publication exceeded its bounded queue")
            self._pending.append(message)
            self.peak_pending_frames = max(self.peak_pending_frames, len(self._pending))
            trigger = not self._armed
            self._armed = True
        if trigger:
            self._guard.trigger()

    def _flush(self):
        # One bounded batch lets camera/PCM subscriptions keep their deadlines.
        for _ in range(self.MAX_PENDING_FRAMES):
            with self._lock:
                if not self._pending:
                    self._armed = False
                    return
                message = self._pending.popleft()
            self._publisher.publish(message)
        with self._lock:
            more = bool(self._pending)
            self._armed = more
        if more:
            self._guard.trigger()
