import sys
import tempfile
import unittest
import wave
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

    def test_silero_uses_512_new_samples_with_64_sample_context(self):
        pipeline = AudioPipeline(self.config_file({
            "provider": "silero",
            "model_path": str(ROOT / "models" / "silero_vad.onnx"),
            "threshold": 0.5,
        }))
        frame = np.zeros(AudioPipeline.FRAME_SIZE, dtype=np.int16).tobytes()

        pipeline._vad_prob(frame)
        pipeline._vad_prob(frame)

        self.assertEqual(
            pipeline._silero_pending.size,
            AudioPipeline.FRAME_SIZE * 2 - AudioPipeline.SILERO_CHUNK_SIZE,
        )
        self.assertEqual(
            pipeline._silero_context.size,
            AudioPipeline.SILERO_CONTEXT_SIZE,
        )

    def test_silero_detects_repository_voice_sample(self):
        pipeline = AudioPipeline(self.config_file({
            "provider": "silero",
            "model_path": str(ROOT / "models" / "silero_vad.onnx"),
            "threshold": 0.5,
        }))
        with wave.open(str(ROOT / "assets" / "wake_response.wav"), "rb") as source:
            self.assertEqual(source.getframerate(), 48000)
            samples = np.frombuffer(
                source.readframes(source.getnframes()), dtype=np.int16
            )[::3]

        probabilities = []
        for offset in range(0, samples.size, AudioPipeline.FRAME_SIZE):
            frame = samples[offset:offset + AudioPipeline.FRAME_SIZE]
            if frame.size != AudioPipeline.FRAME_SIZE:
                break
            probabilities.append(pipeline._vad_prob(frame.tobytes()))

        self.assertGreater(max(probabilities), 0.9)

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

    def test_silence_endpoint_uses_configured_value_and_safe_bounds(self):
        configured = AudioPipeline(self.config_file({
            "provider": "webrtc",
            "aggressiveness": 3,
            "silence_sec": 0.45,
        }))
        too_short = AudioPipeline(self.config_file({
            "provider": "webrtc",
            "aggressiveness": 3,
            "silence_sec": 0.1,
        }))

        self.assertEqual(configured._silence_sec, 0.45)
        self.assertEqual(too_short._silence_sec, 0.3)


if __name__ == "__main__":
    unittest.main()
