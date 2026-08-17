import threading
import unittest

from services.tts_pipeline import OrderedTTSPipeline


class OrderedTTSPipelineTests(unittest.TestCase):
    def test_synthesis_runs_concurrently_but_emits_in_submit_order(self):
        both_started = threading.Event()
        release = threading.Event()
        starts = []
        starts_lock = threading.Lock()
        emitted = []

        def synthesize(text):
            with starts_lock:
                starts.append(text)
                if len(starts) == 2:
                    both_started.set()
            release.wait(timeout=2.0)
            return text.encode("ascii")

        pipeline = OrderedTTSPipeline(
            synthesize=synthesize,
            on_audio=lambda samples, text, _elapsed: emitted.append(("speech", text, samples)),
            on_turn_end=lambda turn_id: emitted.append(("turn_end", turn_id)),
            workers=2,
        )
        pipeline.submit_speech("first")
        pipeline.submit_speech("second")
        pipeline.submit_turn_end("turn-1")

        self.assertTrue(both_started.wait(timeout=1.0))
        release.set()
        pipeline.shutdown()

        self.assertCountEqual(starts, ["first", "second"])
        self.assertEqual(
            emitted,
            [
                ("speech", "first", b"first"),
                ("speech", "second", b"second"),
                ("turn_end", "turn-1"),
            ],
        )

    def test_failed_segment_does_not_block_later_audio_or_turn_end(self):
        errors = []
        emitted = []

        def synthesize(text):
            if text == "bad":
                raise RuntimeError("failed")
            return text

        pipeline = OrderedTTSPipeline(
            synthesize=synthesize,
            on_audio=lambda samples, text, _elapsed: emitted.append((text, samples)),
            on_turn_end=lambda turn_id: emitted.append(("end", turn_id)),
            on_error=lambda text, error, _elapsed: errors.append((text, str(error))),
            workers=2,
        )
        pipeline.submit_speech("bad")
        pipeline.submit_speech("good")
        pipeline.submit_turn_end("turn-2")
        pipeline.shutdown()

        self.assertEqual(errors, [("bad", "failed")])
        self.assertEqual(emitted, [("good", "good"), ("end", "turn-2")])

    def test_first_segment_streams_before_completion_and_later_audio_stays_ordered(self):
        first_chunk_emitted = threading.Event()
        release_stream = threading.Event()
        emitted = []

        def synthesize(text):
            return f"full:{text}".encode("ascii")

        def synthesize_stream(text):
            yield b"chunk-1"
            release_stream.wait(timeout=2.0)
            yield b"chunk-2"

        def on_chunk(samples, text, _elapsed, first_chunk):
            emitted.append(("chunk", text, samples, first_chunk))
            first_chunk_emitted.set()

        pipeline = OrderedTTSPipeline(
            synthesize=synthesize,
            synthesize_stream=synthesize_stream,
            on_audio=lambda samples, text, _elapsed: emitted.append(
                ("speech", text, samples)
            ),
            on_audio_chunk=on_chunk,
            on_stream_end=lambda text, _elapsed: emitted.append(("stream_end", text)),
            on_turn_end=lambda turn_id: emitted.append(("turn_end", turn_id)),
            workers=2,
        )
        pipeline.submit_speech("first")
        pipeline.submit_speech("second")
        pipeline.submit_turn_end("turn-3")

        self.assertTrue(first_chunk_emitted.wait(timeout=1.0))
        self.assertEqual(emitted, [("chunk", "first", b"chunk-1", True)])
        release_stream.set()
        pipeline.shutdown()

        self.assertEqual(
            emitted,
            [
                ("chunk", "first", b"chunk-1", True),
                ("chunk", "first", b"chunk-2", False),
                ("stream_end", "first"),
                ("speech", "second", b"full:second"),
                ("turn_end", "turn-3"),
            ],
        )

    def test_stream_failure_before_audio_falls_back_to_full_synthesis(self):
        emitted = []

        def failed_stream(_text):
            raise RuntimeError("stream unavailable")
            yield

        pipeline = OrderedTTSPipeline(
            synthesize=lambda text: f"fallback:{text}",
            synthesize_stream=failed_stream,
            on_audio=lambda samples, text, _elapsed: emitted.append((text, samples)),
            on_audio_chunk=lambda *_args: None,
            on_turn_end=lambda turn_id: emitted.append(("end", turn_id)),
        )
        pipeline.submit_speech("hello")
        pipeline.submit_turn_end("turn-4")
        pipeline.shutdown()

        self.assertEqual(
            emitted,
            [("hello", "fallback:hello"), ("end", "turn-4")],
        )


if __name__ == "__main__":
    unittest.main()
