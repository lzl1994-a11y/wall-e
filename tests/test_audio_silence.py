import unittest

import numpy as np

from services.audio_silence import StreamingTailSilenceTrimmer, TurnAudioTrimmer


class TurnAudioTrimmerTests(unittest.TestCase):
    SAMPLE_RATE = 48000

    def segment(self):
        leading = np.zeros(int(0.2 * self.SAMPLE_RATE), dtype=np.int16)
        speech = np.full(int(0.3 * self.SAMPLE_RATE), 1200, dtype=np.int16)
        trailing = np.zeros(int(0.7 * self.SAMPLE_RATE), dtype=np.int16)
        return np.concatenate([leading, speech, trailing])

    def test_first_segment_preserves_leading_audio_but_trims_trailing_silence(self):
        trimmer = TurnAudioTrimmer(sample_rate=self.SAMPLE_RATE, keep_silence_ms=100)

        result = trimmer.process(self.segment())

        self.assertTrue(result.first_segment)
        self.assertAlmostEqual(result.leading_cut_ms, 0.0, places=1)
        self.assertAlmostEqual(result.trailing_cut_ms, 600.0, places=1)
        self.assertAlmostEqual(result.processed_ms, 600.0, places=1)

    def test_later_segments_keep_one_hundred_ms_at_both_boundaries(self):
        trimmer = TurnAudioTrimmer(sample_rate=self.SAMPLE_RATE, keep_silence_ms=100)
        trimmer.process(self.segment())

        result = trimmer.process(self.segment())

        self.assertFalse(result.first_segment)
        self.assertAlmostEqual(result.leading_cut_ms, 100.0, places=1)
        self.assertAlmostEqual(result.trailing_cut_ms, 600.0, places=1)
        self.assertAlmostEqual(result.processed_ms, 500.0, places=1)

    def test_retained_trailing_silence_is_forced_to_digital_zero(self):
        leading = np.full(int(0.1 * self.SAMPLE_RATE), 8, dtype=np.int16)
        speech = np.full(int(0.2 * self.SAMPLE_RATE), 1200, dtype=np.int16)
        trailing = np.full(int(0.3 * self.SAMPLE_RATE), 8, dtype=np.int16)
        trimmer = TurnAudioTrimmer(sample_rate=self.SAMPLE_RATE, keep_silence_ms=100)

        result = trimmer.process(np.concatenate([leading, speech, trailing]))

        tail = result.samples[-int(0.1 * self.SAMPLE_RATE) :]
        self.assertTrue(np.all(tail == 0))

    def test_turn_reset_restores_first_segment_behavior(self):
        trimmer = TurnAudioTrimmer(sample_rate=self.SAMPLE_RATE, keep_silence_ms=100)
        trimmer.process(self.segment())
        trimmer.reset()

        result = trimmer.process(self.segment())

        self.assertTrue(result.first_segment)
        self.assertAlmostEqual(result.leading_cut_ms, 0.0, places=1)

    def test_all_silence_is_left_unchanged(self):
        samples = np.zeros(self.SAMPLE_RATE, dtype=np.int16)
        trimmer = TurnAudioTrimmer(sample_rate=self.SAMPLE_RATE, keep_silence_ms=100)

        result = trimmer.process(samples)

        self.assertEqual(result.samples.size, samples.size)
        self.assertEqual(result.leading_cut_ms, 0.0)
        self.assertEqual(result.trailing_cut_ms, 0.0)

    def test_streaming_trimmer_emits_speech_without_waiting_and_trims_tail(self):
        trimmer = StreamingTailSilenceTrimmer(
            sample_rate=self.SAMPLE_RATE,
            keep_silence_ms=100,
        )
        samples = self.segment()
        chunk_size = int(0.07 * self.SAMPLE_RATE)
        emitted = []

        for offset in range(0, samples.size, chunk_size):
            output = trimmer.process(samples[offset : offset + chunk_size])
            if output.size:
                emitted.append(output)

        self.assertTrue(emitted)
        self.assertGreater(sum(chunk.size for chunk in emitted), 0)
        emitted.append(trimmer.finish())
        result = np.concatenate(emitted)

        self.assertAlmostEqual(result.size * 1000 / self.SAMPLE_RATE, 600.0, places=1)
        np.testing.assert_array_equal(
            result[: int(0.2 * self.SAMPLE_RATE)],
            np.zeros(int(0.2 * self.SAMPLE_RATE), dtype=np.int16),
        )

    def test_streaming_segment_advances_whole_segment_trimmer_state(self):
        trimmer = TurnAudioTrimmer(sample_rate=self.SAMPLE_RATE, keep_silence_ms=100)

        self.assertTrue(trimmer.mark_segment())
        result = trimmer.process(self.segment())

        self.assertFalse(result.first_segment)
        self.assertAlmostEqual(result.leading_cut_ms, 100.0, places=1)

    def test_streaming_retained_tail_is_digital_zero(self):
        trimmer = StreamingTailSilenceTrimmer(
            sample_rate=self.SAMPLE_RATE,
            keep_silence_ms=100,
        )
        speech = np.full(int(0.2 * self.SAMPLE_RATE), 1200, dtype=np.int16)
        quiet = np.full(int(0.3 * self.SAMPLE_RATE), 8, dtype=np.int16)

        emitted = trimmer.process(np.concatenate([speech, quiet]))
        tail = trimmer.finish()

        self.assertGreater(emitted.size, 0)
        self.assertEqual(tail.size, int(0.1 * self.SAMPLE_RATE))
        self.assertTrue(np.all(tail == 0))


if __name__ == "__main__":
    unittest.main()
