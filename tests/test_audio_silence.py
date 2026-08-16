import unittest

import numpy as np

from services.audio_silence import TurnAudioTrimmer


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


if __name__ == "__main__":
    unittest.main()
