import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from services.audio.audio_pipeline import AudioPipeline
from services.remote.microphone_mode import RemoteMicrophoneLease


class SharedMicrophoneTests(unittest.TestCase):
    def test_connected_lease_expires_and_false_releases_immediately(self):
        clock = MagicMock(return_value=0.0)
        lease = RemoteMicrophoneLease(clock=clock)
        self.assertTrue(lease.remote_active)  # Wait for gateway's first status.
        lease.update(False)
        self.assertFalse(lease.remote_active)
        lease.update(True)
        clock.return_value = 2.9
        self.assertTrue(lease.remote_active)
        clock.return_value = 3.0
        self.assertFalse(lease.remote_active)
        lease.update(True)
        lease.update(False)
        self.assertFalse(lease.remote_active)
        with self.assertRaises(ValueError):
            lease.update("false")

    def make_pipeline(self):
        with patch("services.audio.audio_pipeline.WakeWordDetector"), patch.object(
            AudioPipeline, "_init_vad"
        ), patch("builtins.open", unittest.mock.mock_open(read_data="{}")):
            return AudioPipeline(shared_capture=True)

    def test_shared_start_never_opens_another_microphone_or_apm(self):
        pipe = self.make_pipeline()
        with patch.object(pipe, "_run"), patch.object(pipe, "_device_monitor") as device, patch(
            "services.audio.audio_pipeline.WebRTCApm"
        ) as apm:
            pipe.start()
            pipe.stop()
        device.assert_not_called()
        apm.assert_not_called()

    def test_shared_pcm_resampling_is_continuous_and_pause_discards_old_audio(self):
        pipe = self.make_pipeline()
        pipe._is_running = True
        t = np.arange(48000) / 48000
        pcm = (np.sin(2 * np.pi * 625 * t) * 10000).astype(np.int16)
        for chunk in np.split(pcm, 50):
            pipe.accept_shared_pcm(chunk.tobytes())
        frames = []
        while not pipe.audio_queue.empty():
            frames.append(pipe.audio_queue.get_nowait())
        self.assertEqual(len(frames), 33)
        self.assertTrue(all(len(frame) == 960 for frame in frames))
        signal = np.frombuffer(b"".join(frames), np.int16).astype(float)
        self.assertGreater(np.corrcoef(signal[100:], (np.sin(2 * np.pi * 625 * np.arange(len(signal)) / 16000) * 10000)[100:])[0, 1], .99)
        pipe.pause()
        pipe.accept_shared_pcm(pcm.tobytes())
        self.assertTrue(pipe.audio_queue.empty())
        pipe.resume()
        for _ in range(3):
            pipe.accept_shared_pcm(bytes(1920))
        self.assertTrue(all(not any(frame) for frame in list(pipe.audio_queue.queue)))


if __name__ == "__main__":
    unittest.main()
