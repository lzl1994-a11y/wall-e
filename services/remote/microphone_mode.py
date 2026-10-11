"""A controller connection temporarily owns microphone use, not capture.

The gateway reports actual PeerConnection state once a second. Three missed
reports release the lease, including after a gateway crash. A newly started
voice node waits for a report before it can wake during an existing call.
"""

import time
import threading
from functools import wraps


def local_voice_callback(callback):
    """Serialize local output against controller takeover and canceled turns."""
    @wraps(callback)
    def guarded(self, *args, **kwargs):
        with getattr(self, "_voice_mode_lock", threading.RLock()):
            if (getattr(self, "_remote_connected", False)
                    or getattr(self, "_voice_task_cancelled", lambda: False)()):
                return
            return callback(self, *args, **kwargs)
    return guarded

REMOTE_CONNECTION_TOPIC = "/remote_connection_state"
CONNECTION_HEARTBEAT_SEC = 1.0
CONNECTION_LEASE_SEC = 3 * CONNECTION_HEARTBEAT_SEC


class RemoteMicrophoneLease:
    def __init__(self, *, clock=time.monotonic):
        self._clock = clock
        self._until = clock() + CONNECTION_LEASE_SEC

    def update(self, connected: bool):
        if not isinstance(connected, bool):
            raise ValueError("connected must be a boolean")
        self._until = self._clock() + CONNECTION_LEASE_SEC if connected else 0.0

    @property
    def remote_active(self):
        return self._clock() < self._until
