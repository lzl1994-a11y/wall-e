import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.asr.baidu_asr import BaiduASR
from services.asr import create_asr


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


if __name__ == "__main__":
    unittest.main()
