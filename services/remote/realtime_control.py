"""Small state gates for high-rate remote-control streams."""

from __future__ import annotations


class ChangedTargetGate:
    """Pass changed absolute targets at a bounded rate.

    Browser control packets also serve as the chassis dead-man heartbeat, so
    they must keep arriving even while the head is stationary.  Turning every
    heartbeat into a new action request, however, repeatedly preempts the same
    servo action and starves media work on a small robot computer.
    """

    def __init__(self, min_interval_sec: float = 0.1):
        self._min_interval_sec = max(0.0, float(min_interval_sec))
        self._last_targets: dict[str, int] | None = None
        self._last_sent_at = float("-inf")

    def accept(self, targets: dict[str, int], now: float) -> bool:
        normalized = {str(name): int(value) for name, value in targets.items()}
        if normalized == self._last_targets:
            return False
        if float(now) - self._last_sent_at < self._min_interval_sec:
            return False
        self._last_targets = normalized
        self._last_sent_at = float(now)
        return True

    def reset(self) -> None:
        self._last_targets = None
        self._last_sent_at = float("-inf")


class ChannelSequenceGate:
    """Independent sequence spaces; unreliable motion can arrive reordered."""

    def __init__(self):
        self._last: dict[str, int] = {}
        self.stopped = False

    def accept(self, channel: str, message: dict) -> bool:
        if self.stopped:
            return False
        if channel == "motion" and message["type"] != "control":
            raise ValueError("motion channel accepts only control")
        sequence = message["seq"]
        if sequence <= self._last.get(channel, -1):
            if channel == "motion":
                return False
            raise ValueError("reliable sequence repeated or reversed")
        self._last[channel] = sequence
        if message["type"] == "stop":
            self.stopped = True
        return True

    def reset(self):
        self._last.clear()
        self.stopped = False


__all__ = ["ChangedTargetGate", "ChannelSequenceGate"]
