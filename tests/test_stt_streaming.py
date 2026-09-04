import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from services.audio_pipeline import _prefer_keyword_models
from services.stt_service import STTService


class STTStreamingTests(unittest.TestCase):
    def setUp(self):
        self.debug_dir = tempfile.TemporaryDirectory(prefix="wali-stt-stream-")
        self.addCleanup(self.debug_dir.cleanup)

    def service(self, adapter):
        service = STTService.__new__(STTService)
        service.asr_adapter = adapter
        service.on_sentence_received = MagicMock()
        service._awake = True
        service._awake_timeout = 8.0
        service._awake_timer = None
        service._awake_timer_generation = 0
        service._awake_lock = threading.Lock()
        service._streaming_active = False
        service._streaming_lock = threading.Lock()
        return service

    def debug_path(self):
        return str(Path(self.debug_dir.name) / "stt_debug_last.wav")

    def test_keyword_models_prefer_epoch99_int8_then_fp32(self):
        models = [
            "/models/encoder-epoch-99-avg-1.onnx",
            "/models/encoder-epoch-12-avg-1.int8.onnx",
            "/models/encoder-epoch-99-avg-1.int8.onnx",
        ]
        self.assertEqual(
            _prefer_keyword_models(models)[0],
            "/models/encoder-epoch-99-avg-1.int8.onnx",
        )
        self.assertEqual(
            _prefer_keyword_models([models[0]])[0],
            "/models/encoder-epoch-99-avg-1.onnx",
        )

    def test_streaming_adapter_publishes_only_final_text(self):
        adapter = MagicMock()
        adapter.supports_streaming = True
        adapter.finish_stream.return_value = "瓦力你好"
        service = self.service(adapter)
        initial = b"\x01\x00" * 4800
        following = b"\x02\x00" * 480

        service._on_speech_start(initial)
        service._on_speech_audio(following)
        with patch("services.stt_service.os.path.expanduser", return_value=self.debug_path()):
            service._on_sentence(initial + following)

        adapter.start_stream.assert_called_once_with(16000)
        self.assertEqual(adapter.accept_audio.call_args_list[0].args[0], initial)
        self.assertEqual(adapter.accept_audio.call_args_list[1].args[0], following)
        adapter.finish_stream.assert_called_once_with()
        adapter.recognize.assert_not_called()
        service.on_sentence_received.assert_called_once_with("瓦力你好")

    def test_final_stream_error_falls_back_to_complete_wav(self):
        adapter = MagicMock()
        adapter.supports_streaming = True
        adapter.finish_stream.side_effect = RuntimeError("connection lost")
        adapter.recognize.return_value = "回退成功"
        service = self.service(adapter)
        pcm = b"\x01\x00" * 4800

        service._on_speech_start(pcm)
        with patch("services.stt_service.os.path.expanduser", return_value=self.debug_path()):
            service._on_sentence(pcm)

        adapter.finish_stream.assert_called_once_with()
        adapter.recognize.assert_called_once()
        wav_path, sample_rate = adapter.recognize.call_args.args
        self.assertEqual(sample_rate, 16000)
        self.assertTrue(str(wav_path).endswith(".wav"))
        service.on_sentence_received.assert_called_once_with("回退成功")


if __name__ == "__main__":
    unittest.main()
