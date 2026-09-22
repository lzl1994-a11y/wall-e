"""Streaming filter for models that embed reasoning in content tags."""


class VisibleAnswerFilter:
    """Drop tagged reasoning and stream only the visible answer."""

    _THINK_OPEN = ("<think>", "<analysis>")
    _THINK_CLOSE = ("</think>", "</analysis>")
    _ANSWER_OPEN = ("<answer>", "<final>")
    _ANSWER_CLOSE = ("</answer>", "</final>")

    def __init__(self):
        self._state = "initial"
        self._pending = ""

    def feed(self, text):
        if not text or self._state == "done":
            return ""
        self._pending += str(text)
        return self._drain()

    def flush(self):
        if self._state in {"initial", "answer", "plain", "after_think"}:
            output = self._pending
        else:
            output = ""
        self._pending = ""
        return output

    def _drain(self):
        output = []
        while True:
            if self._state == "plain":
                output.append(self._pending)
                self._pending = ""
                break

            if self._state == "initial":
                stripped = self._pending.lstrip()
                leading = self._pending[: len(self._pending) - len(stripped)]
                marker = self._complete_prefix(stripped, self._THINK_OPEN)
                if marker:
                    self._pending = stripped[len(marker):]
                    self._state = "think"
                    continue
                marker = self._complete_prefix(stripped, self._ANSWER_OPEN)
                if marker:
                    self._pending = stripped[len(marker):]
                    self._state = "answer"
                    continue
                if not stripped or self._is_marker_prefix(
                    stripped, self._THINK_OPEN + self._ANSWER_OPEN
                ):
                    break
                output.append(leading + stripped)
                self._pending = ""
                self._state = "plain"
                break

            if self._state == "think":
                match = self._find_marker(self._pending, self._THINK_CLOSE)
                if match is None:
                    self._pending = self._marker_suffix(
                        self._pending, self._THINK_CLOSE
                    )
                    break
                index, marker = match
                self._pending = self._pending[index + len(marker):]
                self._state = "after_think"
                continue

            if self._state == "after_think":
                stripped = self._pending.lstrip()
                marker = self._complete_prefix(stripped, self._ANSWER_OPEN)
                if marker:
                    self._pending = stripped[len(marker):]
                    self._state = "answer"
                    continue
                if not stripped or self._is_marker_prefix(stripped, self._ANSWER_OPEN):
                    break
                self._pending = stripped
                self._state = "plain"
                continue

            if self._state == "answer":
                match = self._find_marker(self._pending, self._ANSWER_CLOSE)
                if match is not None:
                    index, marker = match
                    output.append(self._pending[:index])
                    self._pending = self._pending[index + len(marker):]
                    self._state = "done"
                    break
                suffix = self._marker_suffix(self._pending, self._ANSWER_CLOSE)
                emit_length = len(self._pending) - len(suffix)
                if emit_length:
                    output.append(self._pending[:emit_length])
                self._pending = suffix
                break

        return "".join(output)

    @staticmethod
    def _complete_prefix(text, markers):
        lowered = text.lower()
        for marker in markers:
            if lowered.startswith(marker):
                return marker
        return None

    @staticmethod
    def _is_marker_prefix(text, markers):
        lowered = text.lower()
        return any(marker.startswith(lowered) for marker in markers)

    @staticmethod
    def _find_marker(text, markers):
        lowered = text.lower()
        matches = [
            (lowered.find(marker), marker)
            for marker in markers
            if lowered.find(marker) >= 0
        ]
        return min(matches, default=None)

    @staticmethod
    def _marker_suffix(text, markers):
        lowered = text.lower()
        max_length = min(len(text), max(map(len, markers)) - 1)
        for length in range(max_length, 0, -1):
            suffix = lowered[-length:]
            if any(marker.startswith(suffix) for marker in markers):
                return text[-length:]
        return ""
