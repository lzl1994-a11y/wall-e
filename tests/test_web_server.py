import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.web_server import DEFAULT_STATIC_DIR, create_server


def sample_config():
    return {
        "pipeline": {"mode": "asr_llm"},
        "asr": {
            "provider": "zhipu",
            "model": "asr-model",
            "url": "https://example.com/asr",
            "key": "asr-secret",
        },
        "llm": {
            "provider": "aliyun",
            "model": "llm-model",
            "url": "https://example.com/llm",
            "key": "llm-secret",
            "temperature": 0.4,
            "max_tokens": 2048,
        },
        "launch": {"serial": True, "tracking": False},
        "wake_word": {
            "enabled": True,
            "keyword": "瓦力瓦力",
            "model_dir": "models/sherpa-onnx",
            "threshold": 0.2,
            "awake_timeout": 8.0,
            "response_wav": "assets/wake_response.wav",
        },
        "system_prompt": "你是瓦力。",
        "tts": {"engine": "edge-tts", "voice": "zh-CN-XiaoxiaoNeural", "output_device": "default"},
        "serial": {"doa_port": "COM1", "lower_board_port": "COM2", "baudrate": 115200},
        "i2c": {"bus": 1, "pca9685_address": 64, "pwm_frequency": 50},
        "vision": {
            "camera_index": 0,
            "model_path": "models/yolo.onnx",
            "enabled_on_start": False,
            "pid": {"kp": 0.12, "ki": 0.0, "kd": 0.02},
        },
        "servos": [{"id": 0, "name": "eye_r", "limit_1": 2000, "limit_2": 4300, "init": 3000}],
        "motors": [{"id": 0, "name": "track_r", "max_speed": 100, "neutral_speed": 0, "invert_direction": False}],
    }


class ConfigWebServerTests(unittest.TestCase):
    BAIDU_CONFIG = {
        "app_id": 123456,
        "api_key": "baidu-secret",
        "dev_pid": 15372,
        "cuid": "wali-x3",
        "url": "wss://vop.baidu.com/realtime_asr",
    }

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="wali-config-web-")
        self.config_path = Path(self.temp_dir.name) / "config.yaml"
        self.config_path.write_text(
            yaml.safe_dump(sample_config(), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        self.server = create_server(
            host="127.0.0.1",
            port=0,
            config_path=self.config_path,
            static_dir=DEFAULT_STATIC_DIR,
            token="test-token",
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp_dir.cleanup()

    def request(self, path, *, method="GET", payload=None, token="test-token"):
        data = None
        headers = {}
        if token is not None:
            headers["X-Wali-Token"] = token
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers=headers,
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            body = response.read()
            return response.status, json.loads(body) if "application/json" in response.headers.get("Content-Type", "") else body

    def test_static_page_is_available_without_token(self):
        status, body = self.request("/", token=None)
        self.assertEqual(status, 200)
        self.assertIn(b"WALI", body)

    def test_api_requires_token(self):
        with self.assertRaises(urllib.error.HTTPError) as context:
            self.request("/api/config", token=None)
        self.assertEqual(context.exception.code, 401)

    def test_get_redacts_secrets(self):
        status, body = self.request("/api/config")
        self.assertEqual(status, 200)
        self.assertEqual(body["config"]["asr"]["key"], "")
        self.assertEqual(body["config"]["llm"]["key"], "")
        self.assertTrue(body["secret_fields"]["asr.key"])
        self.assertTrue(body["secret_fields"]["llm.key"])

    def test_save_preserves_blank_secrets(self):
        _, body = self.request("/api/config")
        config = body["config"]
        config["llm"]["temperature"] = 0.8
        status, result = self.request("/api/config", method="POST", payload={"config": config})
        self.assertEqual(status, 200)
        self.assertTrue(result["ok"])
        stored = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(stored["asr"]["key"], "asr-secret")
        self.assertEqual(stored["llm"]["key"], "llm-secret")
        self.assertEqual(stored["llm"]["temperature"], 0.8)

    def test_patch_save_only_changes_requested_module(self):
        before = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        status, result = self.request(
            "/api/config",
            method="POST",
            payload={"patch": {"llm": {"temperature": 0.8}}},
        )
        self.assertEqual(status, 200)
        self.assertTrue(result["ok"])
        stored = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(stored["llm"]["temperature"], 0.8)
        self.assertEqual(stored["llm"]["model"], before["llm"]["model"])
        self.assertEqual(stored["llm"]["key"], before["llm"]["key"])
        self.assertEqual(stored["asr"], before["asr"])
        self.assertEqual(stored["servos"], before["servos"])

    def test_invalid_patch_is_rejected_without_writing(self):
        before = self.config_path.read_text(encoding="utf-8")
        with self.assertRaises(urllib.error.HTTPError) as context:
            self.request(
                "/api/config",
                method="POST",
                payload={"patch": {"llm": {"temperature": 9}}},
            )
        self.assertEqual(context.exception.code, 400)
        self.assertEqual(self.config_path.read_text(encoding="utf-8"), before)

    def test_valid_baidu_config_is_saved_and_redacted(self):
        status, result = self.request(
            "/api/config",
            method="POST",
            payload={"patch": {"asr": {"provider": "baidu", "baidu": self.BAIDU_CONFIG}}},
        )
        self.assertEqual(status, 200)
        self.assertTrue(result["ok"])
        self.assertEqual(result["config"]["asr"]["baidu"]["api_key"], "")
        self.assertTrue(result["secret_fields"]["asr.baidu.api_key"])
        stored = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(stored["asr"]["provider"], "baidu")
        self.assertEqual(stored["asr"]["baidu"]["api_key"], "baidu-secret")

    def test_blank_baidu_key_preserves_saved_secret(self):
        self.request(
            "/api/config",
            method="POST",
            payload={"patch": {"asr": {"provider": "baidu", "baidu": self.BAIDU_CONFIG}}},
        )
        status, result = self.request(
            "/api/config",
            method="POST",
            payload={"patch": {"asr": {"baidu": {"api_key": "", "cuid": "wali-x3-new"}}}},
        )
        self.assertEqual(status, 200)
        self.assertTrue(result["ok"])
        stored = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(stored["asr"]["baidu"]["api_key"], "baidu-secret")
        self.assertEqual(stored["asr"]["baidu"]["cuid"], "wali-x3-new")

    def test_baidu_requires_api_key(self):
        config = dict(self.BAIDU_CONFIG)
        del config["api_key"]
        with self.assertRaises(urllib.error.HTTPError) as context:
            self.request(
                "/api/config",
                method="POST",
                payload={"patch": {"asr": {"provider": "baidu", "baidu": config}}},
            )
        self.assertEqual(context.exception.code, 400)

    def test_baidu_rejects_unsupported_pid(self):
        config = {**self.BAIDU_CONFIG, "dev_pid": 99999}
        with self.assertRaises(urllib.error.HTTPError) as context:
            self.request(
                "/api/config",
                method="POST",
                payload={"patch": {"asr": {"provider": "baidu", "baidu": config}}},
            )
        self.assertEqual(context.exception.code, 400)

    def test_baidu_dialect_pid_requires_user(self):
        config = {**self.BAIDU_CONFIG, "dev_pid": 15376}
        with self.assertRaises(urllib.error.HTTPError) as context:
            self.request(
                "/api/config",
                method="POST",
                payload={"patch": {"asr": {"provider": "baidu", "baidu": config}}},
            )
        self.assertEqual(context.exception.code, 400)

    def test_invalid_servo_range_is_rejected_without_writing(self):
        before = self.config_path.read_text(encoding="utf-8")
        _, body = self.request("/api/config")
        config = body["config"]
        config["servos"][0]["init"] = 9000
        with self.assertRaises(urllib.error.HTTPError) as context:
            self.request("/api/config", method="POST", payload={"config": config})
        self.assertEqual(context.exception.code, 400)
        self.assertEqual(self.config_path.read_text(encoding="utf-8"), before)

    def test_lan_binding_requires_token(self):
        with self.assertRaisesRegex(ValueError, "访问令牌"):
            create_server(
                host="0.0.0.0",
                port=0,
                config_path=self.config_path,
                static_dir=DEFAULT_STATIC_DIR,
            )

    def test_non_ascii_token_is_rejected_with_clear_error(self):
        with self.assertRaisesRegex(ValueError, "不能包含中文"):
            create_server(
                host="0.0.0.0",
                port=0,
                config_path=self.config_path,
                static_dir=DEFAULT_STATIC_DIR,
                token="换成一个足够长的随机令牌",
            )


if __name__ == "__main__":
    unittest.main()
