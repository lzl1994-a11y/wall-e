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


if __name__ == "__main__":
    unittest.main()
