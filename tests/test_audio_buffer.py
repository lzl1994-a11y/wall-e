import unittest

import numpy as np

from services.audio_buffer import StreamingPCMPrebuffer


class StreamingPCMPrebufferTests(unittest.TestCase):
    def test_holds_stream_until_target_duration_then_passes_through(self):
        buffer = StreamingPCMPrebuffer(sample_rate=1000, prebuffer_ms=400)

        first, started = buffer.push(np.full(100, 1, dtype=np.int16))
        second, second_started = buffer.push(np.full(200, 2, dtype=np.int16))
        third, third_started = buffer.push(np.full(100, 3, dtype=np.int16))
        later, later_started = buffer.push(np.full(50, 4, dtype=np.int16))

        self.assertEqual(first.size, 0)
        self.assertFalse(started)
        self.assertEqual(second.size, 0)
        self.assertFalse(second_started)
        self.assertTrue(third_started)
        np.testing.assert_array_equal(
            third,
            np.concatenate([
                np.full(100, 1, dtype=np.int16),
                np.full(200, 2, dtype=np.int16),
                np.full(100, 3, dtype=np.int16),
            ]),
        )
        self.assertFalse(later_started)
        np.testing.assert_array_equal(later, np.full(50, 4, dtype=np.int16))

    def test_short_stream_is_released_when_stream_finishes(self):
        buffer = StreamingPCMPrebuffer(sample_rate=1000, prebuffer_ms=400)
        buffer.push(np.full(150, 5, dtype=np.int16))

        output, started = buffer.finish()

        self.assertTrue(started)
        np.testing.assert_array_equal(output, np.full(150, 5, dtype=np.int16))

    def test_reset_starts_a_new_prebuffer_window(self):
        buffer = StreamingPCMPrebuffer(sample_rate=1000, prebuffer_ms=100)
        buffer.push(np.full(100, 1, dtype=np.int16))
        buffer.reset()

        output, started = buffer.push(np.full(50, 2, dtype=np.int16))

        self.assertFalse(started)
        self.assertEqual(output.size, 0)


if __name__ == "__main__":
    unittest.main()
