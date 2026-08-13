import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.audio_pipeline import AudioPipeline


class VadConfigTests(unittest.TestCase):
    def config_file(self, vad):
        temp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", encoding="utf-8", delete=False
        )
        yaml.safe_dump({"wake_word": {"enabled": False}, "vad": vad}, temp)
        temp.close()
        self.addCleanup(Path(temp.name).unlink, missing_ok=True)
        return temp.name

    def test_silero_provider_loads_existing_model_and_scores_frame(self):
        pipeline = AudioPipeline(self.config_file({
            "provider": "silero",
            "model_path": str(ROOT / "models" / "silero_vad.onnx"),
            "threshold": 0.5,
            "aggressiveness": 3,
        }))
        frame = np.zeros(AudioPipeline.FRAME_SIZE, dtype=np.int16).tobytes()
        probability = pipeline._vad_prob(frame)

        self.assertEqual(pipeline._vad_backend, "silero")
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_missing_silero_model_falls_back_to_webrtc(self):
        fake_vad = type("FakeVad", (), {"is_speech": lambda self, frame, rate: False})
        fake_module = type("FakeWebRtc", (), {"Vad": lambda _level: fake_vad()})
        with patch.dict(sys.modules, {"webrtcvad": fake_module}):
            pipeline = AudioPipeline(self.config_file({
                "provider": "silero",
                "model_path": "models/does-not-exist.onnx",
                "threshold": 0.5,
                "aggressiveness": 2,
            }))
        self.assertEqual(pipeline._vad_backend, "webrtc")


if __name__ == "__main__":
    unittest.main()
