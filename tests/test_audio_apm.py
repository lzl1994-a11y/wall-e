import queue
import unittest
from unittest.mock import Mock

import numpy as np

from services.audio_apm import WebRTCApm
from services.audio_pipeline import AudioPipeline


class WebRTCApmBackpressureTests(unittest.TestCase):
    def test_full_input_queue_marks_apm_overloaded(self):
        apm = WebRTCApm(lambda _pcm: None)
        apm._running = True
        apm._queue = queue.Queue(maxsize=1)
        apm._queue.put(b"first")

        self.assertFalse(apm.submit(b"second"))
        self.assertTrue(apm.overloaded)

    def test_audio_callback_schedules_nonblocking_fallback(self):
        pipeline = AudioPipeline.__new__(AudioPipeline)
        pipeline._is_running = True
        pipeline._is_paused = False
        pipeline._device_sample_rate = AudioPipeline.SAMPLE_RATE
        pipeline._apm_disable_scheduled = False
        pipeline._queue_processed_pcm = Mock()
        pipeline._disable_overloaded_apm = Mock()
        pipeline._apm = Mock(overloaded=True)
        pipeline._apm.submit.return_value = False

        samples = np.zeros((AudioPipeline.FRAME_SIZE, 1), dtype=np.float32)
        pipeline._audio_callback(samples, len(samples), None, None)

        pipeline._disable_overloaded_apm.assert_called_once_with(pipeline._apm)
        pipeline._queue_processed_pcm.assert_called_once()


if __name__ == "__main__":
    unittest.main()
