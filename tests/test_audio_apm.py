import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.audio_apm import WebRTCApm
from services.audio_pipeline import AudioPipeline
from services.web_server import validate_config


class WebRTCApmTests(unittest.TestCase):
    def test_command_resamples_to_16khz_before_web_rtc_apm(self):
        command = WebRTCApm(lambda _pcm: None, pre_gain_db=6)._command(48000)
        caps = "audio/x-raw,format=S16LE,layout=interleaved,rate=16000,channels=1"
        self.assertIn("sample-rate=48000", command)
        self.assertIn(caps, command)
        self.assertLess(command.index(caps), command.index("webrtcdsp"))
        self.assertLess(command.index(caps), command.index("volume"))
        self.assertEqual(command[-3:], ["fdsink", "fd=1", "sync=false"])

    def test_fallback_resampler_keeps_vad_pcm_at_16khz(self):
        pipeline = AudioPipeline.__new__(AudioPipeline)
        pipeline._device_sample_rate = 48000
        source = np.arange(1440, dtype=np.int16)
        result = pipeline._resample_fallback(source)
        np.testing.assert_array_equal(result, source[::3])

    def test_web_config_validates_apm_gain_range(self):
        config = {
            "pipeline": {"mode": "asr_llm"}, "asr": {}, "wake_word": {},
            "system_prompt": "x", "tts": {}, "serial": {}, "i2c": {}, "vad": {"provider": "webrtc"},
            "audio_capture": {"webrtc_apm_enabled": True, "webrtc_pre_gain_db": 25},
        }
        self.assertIn("audio_capture.webrtc_pre_gain_db 必须在 -12 到 24 之间", validate_config(config))


if __name__ == "__main__":
    unittest.main()
