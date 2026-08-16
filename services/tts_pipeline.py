"""Concurrent TTS synthesis with ordered audio delivery."""

from concurrent.futures import ThreadPoolExecutor
import threading
import time


class OrderedTTSPipeline:
    """Prefetch speech concurrently while preserving submitted order."""

    def __init__(
        self,
        synthesize,
        on_audio,
        on_turn_end,
        on_error=None,
        workers=2,
    ):
        self._synthesize = synthesize
        self._on_audio = on_audio
        self._on_turn_end = on_turn_end
        self._on_error = on_error
        self._condition = threading.Condition()
        self._results = {}
        self._next_submit = 0
        self._next_emit = 0
        self._closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(workers)),
            thread_name_prefix="tts-synthesis",
        )
        self._emitter = threading.Thread(
            target=self._emit_worker,
            name="tts-ordered-emitter",
            daemon=True,
        )
        self._emitter.start()

    def submit_speech(self, text):
        sequence = self._reserve_sequence()
        self._executor.submit(self._synthesize_task, sequence, text)

    def submit_turn_end(self, turn_id):
        sequence = self._reserve_sequence()
        self._store_result(sequence, ("turn_end", turn_id, None, 0.0))

    def shutdown(self):
        with self._condition:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True)
        with self._condition:
            self._condition.notify_all()
        self._emitter.join(timeout=5.0)

    def _reserve_sequence(self):
        with self._condition:
            if self._closed:
                raise RuntimeError("TTS pipeline is closed")
            sequence = self._next_submit
            self._next_submit += 1
            return sequence

    def _synthesize_task(self, sequence, text):
        started = time.monotonic()
        try:
            samples = self._synthesize(text)
            result = ("speech", text, samples, time.monotonic() - started)
        except Exception as exc:
            result = ("error", text, exc, time.monotonic() - started)
        self._store_result(sequence, result)

    def _store_result(self, sequence, result):
        with self._condition:
            self._results[sequence] = result
            self._condition.notify_all()

    def _emit_worker(self):
        while True:
            with self._condition:
                while self._next_emit not in self._results:
                    if self._closed and self._next_emit >= self._next_submit:
                        return
                    self._condition.wait()
                result = self._results.pop(self._next_emit)
                self._next_emit += 1

            item_type, value, payload, elapsed = result
            try:
                if item_type == "speech":
                    self._on_audio(payload, value, elapsed)
                elif item_type == "turn_end":
                    self._on_turn_end(value)
                elif self._on_error:
                    self._on_error(value, payload, elapsed)
            except Exception as exc:
                if self._on_error:
                    self._on_error(value, exc, elapsed)
