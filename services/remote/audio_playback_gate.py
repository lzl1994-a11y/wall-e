"""Keep delayed ROS audio from reopening a stopped intercom turn."""
import json


REMOTE_AUDIO_EPOCH_LABEL_PREFIX = "walle.remote_audio_epoch:"
MAX_AUDIO_EPOCH = 2**63 - 1


def valid_audio_epoch(value):
    return isinstance(value, int) and not isinstance(value, bool) and 0 < value <= MAX_AUDIO_EPOCH


def audio_message_epoch(message):
    labels = [dimension.label[len(REMOTE_AUDIO_EPOCH_LABEL_PREFIX):]
              for dimension in message.layout.dim
              if dimension.label.startswith(REMOTE_AUDIO_EPOCH_LABEL_PREFIX)]
    if len(labels) != 1 or not labels[0].isascii() or not labels[0].isdigit() or len(labels[0]) > 19:
        return None
    value = int(labels[0])
    return value if valid_audio_epoch(value) else None


class AudioPlaybackGate:
    def __init__(self):
        self.epoch = 0
        self._voice = False
        self._call = False
        self._closed = True

    @property
    def active(self):
        return self._voice or self._call

    def close(self):
        self._voice = self._call = False
        self._closed = True

    def update(self, raw):
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            return False
        if not isinstance(payload, dict):
            return False
        epoch = payload.get("epoch")
        state = payload.get("state")
        mode = payload.get("mode")
        if (not valid_audio_epoch(epoch) or epoch < self.epoch
                or mode not in (None, "session")
                or state not in (("start", "end") if mode == "session" else ("start", "stop"))):
            return False
        if epoch > self.epoch:
            self.epoch = epoch
            self.close()
            self._closed = state != "start"
        elif state == "start" and self._closed:
            # A delayed start cannot undo a stop of the same turn.
            return False
        if mode == "session":
            self._call = state == "start"
        else:
            self._voice = state == "start"
        if not self.active:
            self._closed = True
        return True

    def accepts(self, epoch):
        return self.active and valid_audio_epoch(epoch) and epoch == self.epoch
