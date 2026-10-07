import queue
import unittest
from unittest.mock import Mock, mock_open, patch

import numpy as np

from services.audio.audio_apm import WebRTCApm
from services.audio.audio_pipeline import AudioPipeline


class WebRTCApmBackpressureTests(unittest.TestCase):
    def test_pending_apm_audio_is_bounded_to_four_frames(self):
        apm = WebRTCApm(lambda _pcm: None)
        apm._running = True
        for _ in range(4):
            self.assertTrue(apm.submit(b"pcm"))
        self.assertFalse(apm.submit(b"stale"))
        apm._queue.get_nowait()
        self.assertFalse(apm.submit(b"late"))

    def test_native_intercom_pcm_and_asr_formats_do_not_cross(self):
        with patch("builtins.open", mock_open(read_data="{}")):
            pipeline = AudioPipeline(raw_only=True)
        self.assertEqual(pipeline.SAMPLE_RATE, 48000)
        self.assertEqual(pipeline.FRAME_MS, 20)
        self.assertEqual(pipeline.FRAME_BYTES, 1920)
        self.assertEqual(AudioPipeline.SAMPLE_RATE, 16000)
        pipeline._device_sample_rate = 48000
        samples = np.arange(960, dtype=np.int16)
        np.testing.assert_array_equal(pipeline._resample_fallback(samples), samples)
        command = WebRTCApm(lambda _: None, output_rate=48000, frame_ms=20)._command(48000)
        self.assertIn("blocksize=1920", command)
        self.assertIn("audio/x-raw,format=S16LE,layout=interleaved,rate=48000,channels=1", command)
        self.assertIn("rate=16000", " ".join(WebRTCApm(lambda _: None)._command(48000)))

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
