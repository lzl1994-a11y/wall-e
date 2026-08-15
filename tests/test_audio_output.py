import asyncio
import sys
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.audio_output import OUTPUT_SAMPLE_RATE
from services.audio_pipeline import AudioPipeline
from services.playback_service import PlaybackService
from services.tts_service import TTSService


class AudioSampleRateContractTests(unittest.TestCase):
    def test_input_pipeline_remains_16khz(self):
        self.assertEqual(AudioPipeline.SAMPLE_RATE, 16000)

    @patch("services.playback_service.threading.Thread")
    @patch.object(PlaybackService, "_refresh_device", return_value=True)
    def test_playback_service_defaults_to_48khz(self, _refresh_device, thread_class):
        player = PlaybackService()

        self.assertEqual(player.sample_rate, OUTPUT_SAMPLE_RATE)
        self.assertEqual(OUTPUT_SAMPLE_RATE, 48000)
        thread_class.return_value.start.assert_called_once_with()

    def test_tts_audio_is_resampled_to_48khz(self):
        service = TTSService.__new__(TTSService)
        service.voice = "test-voice"
        service.rate = "+0%"
        service.pitch = "+0Hz"
        service.sample_rate = OUTPUT_SAMPLE_RATE

        source = MagicMock()
        at_rate = MagicMock()
        mono = MagicMock()
        pcm = MagicMock()
        source.set_frame_rate.return_value = at_rate
        at_rate.set_channels.return_value = mono
        mono.set_sample_width.return_value = pcm
        pcm.get_array_of_samples.return_value = [1, -2, 3]

        class FakeCommunicate:
            def __init__(self, *args, **kwargs):
                pass

            async def stream(self):
                yield {"type": "audio", "data": b"mp3-data"}

        with (
            patch("services.tts_service.edge_tts.Communicate", FakeCommunicate),
            patch("pydub.AudioSegment.from_mp3", return_value=source),
        ):
            samples = asyncio.run(service._download("你好"))

        source.set_frame_rate.assert_called_once_with(48000)
        at_rate.set_channels.assert_called_once_with(1)
        mono.set_sample_width.assert_called_once_with(2)
        np.testing.assert_array_equal(samples, np.array([1, -2, 3], dtype=np.int16))

    def test_wake_response_asset_is_48khz_pcm_mono(self):
        with wave.open(str(ROOT / "assets" / "wake_response.wav"), "rb") as wav_file:
            self.assertEqual(wav_file.getframerate(), 48000)
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getsampwidth(), 2)


if __name__ == "__main__":
    unittest.main()
