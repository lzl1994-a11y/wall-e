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
        synthesize_stream=None,
        on_audio_chunk=None,
        on_stream_end=None,
    ):
        self._synthesize = synthesize
        self._on_audio = on_audio
        self._on_turn_end = on_turn_end
        self._on_error = on_error
        self._synthesize_stream = synthesize_stream
        self._on_audio_chunk = on_audio_chunk
        self._on_stream_end = on_stream_end
        self._condition = threading.Condition()
        self._results = {}
        self._next_submit = 0
        self._next_emit = 0
        self._closed = False
        self._turn_has_speech = False
        self._generation = 0
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
        with self._condition:
            if self._closed:
                raise RuntimeError("TTS pipeline is closed")
            sequence = self._next_submit
            self._next_submit += 1
            generation = self._generation
            use_stream = (
                not self._turn_has_speech
                and self._synthesize_stream is not None
                and self._on_audio_chunk is not None
            )
            self._turn_has_speech = True
        task = self._synthesize_stream_task if use_stream else self._synthesize_task
        self._executor.submit(task, sequence, text, generation)

    def submit_turn_end(self, turn_id):
        with self._condition:
            if self._closed:
                raise RuntimeError("TTS pipeline is closed")
            sequence = self._next_submit
            self._next_submit += 1
            self._turn_has_speech = False
            generation = self._generation
        self._store_result(sequence, ("turn_end", turn_id, None, 0.0, generation))

    def cancel_pending(self) -> None:
        """Invalidate queued and in-flight output from the current TTS turn."""
        with self._condition:
            if self._closed:
                return
            self._generation += 1
            self._results.clear()
            # The emitter must not wait for results belonging to the old turn.
            self._next_emit = self._next_submit
            self._turn_has_speech = False
            self._condition.notify_all()

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

    def _synthesize_task(self, sequence, text, generation):
        started = time.monotonic()
        try:
            samples = self._synthesize(text)
            result = ("speech", text, samples, time.monotonic() - started, generation)
        except Exception as exc:
            result = ("error", text, exc, time.monotonic() - started, generation)
        self._store_result(sequence, result)

    def _synthesize_stream_task(self, sequence, text, generation):
        started = time.monotonic()
        try:
            stream = self._synthesize_stream(text)
            result = ("stream", text, stream, started, generation)
        except Exception as exc:
            result = ("error", text, exc, time.monotonic() - started, generation)
        self._store_result(sequence, result)

    def _store_result(self, sequence, result):
        with self._condition:
            if sequence < self._next_emit:
                return
            self._results[sequence] = result
            self._condition.notify_all()

    def _generation_is_current(self, generation):
        with self._condition:
            return generation == self._generation

    def _emit_worker(self):
        while True:
            with self._condition:
                while self._next_emit not in self._results:
                    if self._closed and self._next_emit >= self._next_submit:
                        return
                    self._condition.wait()
                result = self._results.pop(self._next_emit)
                self._next_emit += 1

            if len(result) == 5:
                item_type, value, payload, elapsed, generation = result
            else:
                item_type, value, payload, elapsed = result
                generation = self._generation
            try:
                if not self._generation_is_current(generation):
                    continue
                if item_type == "speech":
                    self._on_audio(payload, value, elapsed)
                elif item_type == "stream":
                    self._emit_stream(value, payload, elapsed, generation)
                elif item_type == "turn_end":
                    self._on_turn_end(value)
                elif self._on_error:
                    self._on_error(value, payload, elapsed)
            except Exception as exc:
                if self._on_error:
                    self._on_error(value, exc, elapsed)

    def _emit_stream(self, text, stream, started, generation):
        emitted = False
        stream_error = None
        try:
            for samples in stream:
                if not self._generation_is_current(generation):
                    return
                if samples is None or len(samples) == 0:
                    continue
                first_chunk = not emitted
                emitted = True
                self._on_audio_chunk(
                    samples,
                    text,
                    time.monotonic() - started,
                    first_chunk,
                )
            if not emitted:
                raise RuntimeError("streaming TTS returned no PCM chunks")
        except Exception as exc:
            stream_error = exc
        finally:
            if emitted and self._on_stream_end and self._generation_is_current(generation):
                self._on_stream_end(text, time.monotonic() - started)

        if stream_error is None or not self._generation_is_current(generation):
            return
        # A timeout means the upstream network path is unhealthy. Repeating
        # the same request through full synthesis would block the turn again.
        if isinstance(stream_error, TimeoutError):
            if self._on_error:
                self._on_error(text, stream_error, time.monotonic() - started)
            return
        if emitted:
            if self._on_error:
                self._on_error(text, stream_error, time.monotonic() - started)
            return

        fallback_started = time.monotonic()
        try:
            samples = self._synthesize(text)
            if self._generation_is_current(generation):
                self._on_audio(samples, text, time.monotonic() - fallback_started)
        except Exception as fallback_error:
            if self._on_error:
                self._on_error(
                    text,
                    RuntimeError(
                        f"streaming failed ({stream_error}); fallback failed ({fallback_error})"
                    ),
                    time.monotonic() - started,
                )
