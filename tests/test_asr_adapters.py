import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.asr.baidu_asr import BaiduASR
from services.asr import create_asr
from services.asr.faster_whisper_asr import FasterWhisperASR
from services.asr.sherpa_onnx_asr import SherpaParaformerASR


class BaiduASRTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="wali-baidu-asr-")
        self.wav_path = Path(self.temp_dir.name) / "speech.wav"
        with wave.open(str(self.wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x01\x00" * 4000)

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def adapter(**overrides):
        settings = {
            "app_id": 123456,
            "api_key": "baidu-key",
            "dev_pid": 15372,
            "cuid": "wali-x3",
        }
        settings.update(overrides)
        return BaiduASR(**settings)

    @patch("services.asr.baidu_asr.time.sleep", return_value=None)
    @patch("websocket.create_connection")
    def test_sends_baidu_protocol_and_returns_final_text(self, create_connection, _sleep):
        connection = MagicMock()
        connection.recv.side_effect = [
            json.dumps({"type": "MID_TEXT", "err_no": 0, "result": "瓦"}),
            json.dumps({"type": "FIN_TEXT", "err_no": 0, "result": "瓦力你好"}),
        ]
        create_connection.return_value = connection

        result = self.adapter(lm_id=88).recognize(str(self.wav_path))

        self.assertEqual(result, "瓦力你好")
        url = create_connection.call_args.args[0]
        self.assertTrue(url.startswith("wss://vop.baidu.com/realtime_asr?sn="))
        start = json.loads(connection.send.call_args_list[0].args[0])
        self.assertEqual(start["type"], "START")
        self.assertEqual(
            start["data"],
            {
                "appid": 123456,
                "appkey": "baidu-key",
                "dev_pid": 15372,
                "cuid": "wali-x3",
                "format": "pcm",
                "sample": 16000,
                "lm_id": 88,
            },
        )
        chunks = [call.args[0] for call in connection.send_binary.call_args_list]
        self.assertEqual(b"".join(chunks), b"\x01\x00" * 4000)
        self.assertTrue(all(0 < len(chunk) <= BaiduASR.CHUNK_BYTES for chunk in chunks))
        self.assertEqual(json.loads(connection.send.call_args_list[-1].args[0]), {"type": "FINISH"})
        connection.close.assert_called_once_with()

    @patch("services.asr.baidu_asr.time.sleep", return_value=None)
    @patch("websocket.create_connection")
    def test_api_error_returns_empty_text(self, create_connection, _sleep):
        connection = MagicMock()
        connection.recv.return_value = json.dumps({"type": "FIN_TEXT", "err_no": 3301, "err_msg": "audio error"})
        create_connection.return_value = connection

        self.assertEqual(self.adapter().recognize(str(self.wav_path)), "")
        connection.close.assert_called_once_with()

    @patch("websocket.create_connection")
    def test_rejects_non_16khz_wav_before_connecting(self, create_connection):
        bad_path = Path(self.temp_dir.name) / "bad.wav"
        with wave.open(str(bad_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(8000)
            wav_file.writeframes(b"\x00\x00" * 800)

        self.assertEqual(self.adapter().recognize(str(bad_path)), "")
        create_connection.assert_not_called()

    def test_factory_creates_baidu_adapter_from_nested_config(self):
        config_path = Path(self.temp_dir.name) / "config.yaml"
        config_path.write_text(
            """asr:
  provider: baidu
  baidu:
    app_id: 123456
    api_key: baidu-key
    dev_pid: 15376
    cuid: wali-x3
    url: wss://vop.baidu.com/realtime_asr
    lm_id: 88
    user: dialect-user
""",
            encoding="utf-8",
        )

        adapter = create_asr(str(config_path))

        self.assertIsInstance(adapter, BaiduASR)
        self.assertEqual(adapter.app_id, 123456)
        self.assertEqual(adapter.api_key, "baidu-key")
        self.assertEqual(adapter.dev_pid, 15376)
        self.assertEqual(adapter.lm_id, 88)
        self.assertEqual(adapter.user, "dialect-user")


class ConfiguredModelTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="wali-asr-model-")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_config(self, provider, model=None):
        config_path = Path(self.temp_dir.name) / "config.yaml"
        provider_config = {"api_key": "test-key"}
        if model is not None:
            provider_config["model"] = model
        if provider == "zhipu":
            provider_config["url"] = "https://example.com/audio/transcriptions"
        config_path.write_text(
            yaml.safe_dump(
                {"asr": {"provider": provider, provider: provider_config}},
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        return config_path

    def test_zhipu_model_must_be_configured(self):
        config_path = self._write_config("zhipu")

        with self.assertRaisesRegex(ValueError, r"asr\.zhipu\.model"):
            create_asr(str(config_path))

    def test_aliyun_model_must_be_configured(self):
        config_path = self._write_config("aliyun", model="   ")

        with self.assertRaisesRegex(ValueError, r"asr\.aliyun\.model"):
            create_asr(str(config_path))

    def test_configured_model_is_preserved(self):
        config_path = self._write_config("zhipu", model="  configured-asr  ")

        adapter = create_asr(str(config_path))

        self.assertEqual(adapter.model, "configured-asr")


class LocalASRTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="wali-local-asr-")
        self.root = Path(self.temp_dir.name)
        self.wav_path = self.root / "speech.wav"
        with wave.open(str(self.wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x01\x00" * 1600)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _file(self, name):
        path = self.root / name
        path.write_bytes(b"model")
        return path

    def test_sherpa_paraformer_converts_wav_and_returns_text(self):
        stream = MagicMock()
        recognizer = MagicMock()
        recognizer.create_stream.return_value = stream
        recognizer.get_result.return_value = SimpleNamespace(text="  瓦力你好  ")
        offline = SimpleNamespace(from_paraformer=MagicMock(return_value=recognizer))
        module = SimpleNamespace(OfflineRecognizer=offline)

        with patch("services.asr.sherpa_onnx_asr._load_sherpa_onnx", return_value=module):
            adapter = SherpaParaformerASR(
                model=str(self._file("model.onnx")),
                tokens=str(self._file("tokens.txt")),
                num_threads=3,
            )

        result = adapter.recognize(str(self.wav_path))

        self.assertEqual(result, "瓦力你好")
        samples = stream.accept_waveform.call_args.args[1]
        self.assertEqual(stream.accept_waveform.call_args.args[0], 16000)
        self.assertEqual(samples.dtype.name, "float32")
        recognizer.decode_stream.assert_called_once_with(stream)
        offline.from_paraformer.assert_called_once_with(
            paraformer=str((self.root / "model.onnx").resolve()),
            tokens=str((self.root / "tokens.txt").resolve()),
            num_threads=3,
            sample_rate=16000,
            feature_dim=80,
        )

    def test_factory_selects_only_configured_local_engine(self):
        model = self._file("model.onnx")
        tokens = self._file("tokens.txt")
        config_path = self.root / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "asr": {
                        "mode": "local",
                        "engine": "sherpa_onnx_paraformer",
                        "sherpa_onnx_paraformer": {
                            "model": str(model),
                            "tokens": str(tokens),
                            "num_threads": 4,
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        sentinel = object()

        with patch(
            "services.asr.sherpa_onnx_asr.SherpaParaformerASR",
            return_value=sentinel,
        ) as adapter_class:
            result = create_asr(str(config_path))

        self.assertIs(result, sentinel)
        adapter_class.assert_called_once_with(
            model=str(model.resolve()),
            tokens=str(tokens.resolve()),
            num_threads=4,
        )

    def test_factory_rejects_missing_local_model_file(self):
        config_path = self.root / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "asr": {
                        "mode": "local",
                        "engine": "sherpa_onnx_paraformer",
                        "sherpa_onnx_paraformer": {
                            "model": str(self.root / "missing.onnx"),
                            "tokens": str(self._file("tokens.txt")),
                            "num_threads": 2,
                        },
                    }
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(FileNotFoundError, r"asr\.sherpa_onnx_paraformer\.model"):
            create_asr(str(config_path))

    def test_faster_whisper_is_loaded_lazily_and_returns_text(self):
        model_dir = self.root / "faster-model"
        model_dir.mkdir()
        model = MagicMock()
        model.transcribe.return_value = (
            iter([SimpleNamespace(text=" 瓦力"), SimpleNamespace(text="你好 ")]),
            object(),
        )
        whisper_model = MagicMock(return_value=model)
        fake_module = SimpleNamespace(WhisperModel=whisper_model)

        with patch.dict(sys.modules, {"faster_whisper": fake_module}):
            adapter = FasterWhisperASR(
                model_path=str(model_dir),
                language="zh",
                device="cpu",
                compute_type="int8",
            )

        self.assertEqual(adapter.recognize(str(self.wav_path)), "瓦力你好")
        whisper_model.assert_called_once_with(
            str(model_dir.resolve()),
            device="cpu",
            compute_type="int8",
        )
        model.transcribe.assert_called_once_with(
            str(self.wav_path),
            language="zh",
            task="transcribe",
        )


if __name__ == "__main__":
    unittest.main()
