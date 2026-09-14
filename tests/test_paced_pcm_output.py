import threading
import time
import unittest

import numpy as np

from services.paced_pcm_output import PacedPCMOutput


class PacedPCMOutputTests(unittest.TestCase):
    def test_initial_prebuffer_bursts_then_pcm_is_paced(self):
        emitted = []
        done = threading.Event()
        output = PacedPCMOutput(
            sample_rate=1000,
            chunk_ms=100,
            prebuffer_ms=200,
            max_queued_audio_sec=2,
            on_pcm=lambda samples: emitted.append((time.monotonic(), samples.copy())),
            on_turn_end=lambda _turn_id: done.set(),
        )
        try:
            output.submit(np.arange(500, dtype=np.int16))
            output.finish_turn("turn")
            self.assertTrue(done.wait(timeout=1.0))
        finally:
            output.close()

        self.assertEqual([samples.size for _, samples in emitted], [100] * 5)
        self.assertLess(emitted[1][0] - emitted[0][0], 0.05)
        self.assertGreaterEqual(emitted[3][0] - emitted[0][0], 0.09)
        self.assertGreaterEqual(emitted[4][0] - emitted[0][0], 0.19)

    def test_turn_marker_waits_for_all_pcm(self):
        events = []
        done = threading.Event()
        output = PacedPCMOutput(
            sample_rate=1000,
            chunk_ms=100,
            prebuffer_ms=0,
            max_queued_audio_sec=2,
            on_pcm=lambda samples: events.append(("pcm", samples.size)),
            on_turn_end=lambda turn_id: (events.append(("end", turn_id)), done.set()),
        )
        try:
            output.submit(np.ones(250, dtype=np.int16))
            output.finish_turn("turn")
            self.assertTrue(done.wait(timeout=1.0))
        finally:
            output.close()

        self.assertEqual(events, [("pcm", 100), ("pcm", 100), ("pcm", 50), ("end", "turn")])


if __name__ == "__main__":
    unittest.main()
