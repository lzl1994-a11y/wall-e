import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

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
            "reasoning_effort": "fast",
        },
        "launch": {"serial": True, "tracking": False},
        "hardware": {"backend": "serial_mcu"},
        "remote_control": {"servo_step_size": 40.0, "update_rate_hz": 20},
        "wake_word": {
            "enabled": True,
            "keyword": "瓦力瓦力",
            "model_dir": "models/sherpa-onnx",
            "threshold": 0.2,
            "awake_timeout": 8.0,
            "response_wav": "assets/wake_response.wav",
        },
        "vad": {
            "provider": "webrtc",
            "aggressiveness": 3,
            "model_path": "models/silero_vad.onnx",
            "threshold": 0.5,
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

    def test_remote_control_settings_are_on_servo_page(self):
        _, body = self.request("/", token=None)
        html = body.decode("utf-8")
        servo_panel = html.split('data-panel="servos"', 1)[1].split('data-panel="motors"', 1)[0]
        self.assertIn('data-module="remote_control"', servo_panel)
        self.assertIn('data-path="remote_control.servo_step_size"', servo_panel)
        self.assertIn('data-path="remote_control.update_rate_hz"', servo_panel)

    def test_usb_role_selectors_are_on_hardware_page(self):
        _, body = self.request("/", token=None)
        html = body.decode("utf-8")
        self.assertIn('data-module="usb_devices"', html)
        self.assertIn('data-usb-role="camera"', html)
        self.assertIn('data-usb-role="screen_motion"', html)
        self.assertIn('data-usb-role="voice"', html)

    def test_hardware_backend_controls_are_visible(self):
        _, body = self.request("/", token=None)
        html = body.decode("utf-8")
        self.assertIn('data-module="hardware"', html)
        self.assertIn('data-path="hardware.backend"', html)
        self.assertIn('value="serial_mcu"', html)
        self.assertIn('value="ubuntu_i2c"', html)
        self.assertIn('data-hardware-backend-panel="ubuntu_i2c"', html)

    def test_hardware_backend_patch_is_saved_independently(self):
        before = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        _, body = self.request(
            "/api/config",
            method="POST",
            payload={"patch": {"hardware": {"backend": "ubuntu_i2c"}}},
        )
        stored = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.assertTrue(body["ok"])
        self.assertEqual(stored["hardware"]["backend"], "ubuntu_i2c")
        self.assertEqual(stored["i2c"], before["i2c"])

    def test_invalid_hardware_backend_is_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as context:
            self.request(
                "/api/config",
                method="POST",
                payload={"patch": {"hardware": {"backend": "gpio"}}},
            )
        self.assertEqual(context.exception.code, 400)

    def test_access_token_change_controls_are_visible(self):
        _, body = self.request("/", token=None)
        html = body.decode("utf-8")
        self.assertIn('id="new-access-token"', html)
        self.assertIn('id="change-token-button"', html)
        self.assertIn("修改令牌", html)

    def test_vad_configuration_controls_are_visible(self):
        _, body = self.request("/", token=None)
        html = body.decode("utf-8")
        self.assertIn('data-module="vad"', html)
        self.assertIn('data-path="vad.provider"', html)
        self.assertIn('data-path="vad.aggressiveness"', html)
        self.assertIn('data-path="vad.model_path"', html)
        self.assertIn('data-path="vad.threshold"', html)

    def test_vad_patch_is_saved_independently(self):
        before = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        _, body = self.request(
            "/api/config",
            method="POST",
            payload={
                "patch": {
                    "vad": {
                        "provider": "silero",
                        "aggressiveness": 2,
                        "model_path": "models/silero_vad.onnx",
                        "threshold": 0.62,
                    }
                }
            },
        )
        stored = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.assertTrue(body["ok"])
        self.assertEqual(stored["vad"]["provider"], "silero")
        self.assertEqual(stored["vad"]["threshold"], 0.62)
        self.assertEqual(stored["llm"], before["llm"])

    def test_invalid_vad_provider_is_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as context:
            self.request(
                "/api/config",
                method="POST",
                payload={
                    "patch": {
                        "vad": {
                            "provider": "unknown",
                            "aggressiveness": 3,
                            "model_path": "models/silero_vad.onnx",
                            "threshold": 0.5,
                        }
                    }
                },
            )
        self.assertEqual(context.exception.code, 400)

    def test_api_requires_token(self):
        with self.assertRaises(urllib.error.HTTPError) as context:
            self.request("/api/config", token=None)
        self.assertEqual(context.exception.code, 401)

    def test_access_token_can_be_changed_and_is_persisted(self):
        status, result = self.request(
            "/api/access-token",
            method="POST",
            payload={"new_token": "updated-token"},
        )
        self.assertEqual(status, 200)
        self.assertTrue(result["ok"])
        self.assertFalse(result["restart_required"])

        with self.assertRaises(urllib.error.HTTPError) as context:
            self.request("/api/config", token="test-token")
        self.assertEqual(context.exception.code, 401)

        status, result = self.request("/api/config", token="updated-token")
        self.assertEqual(status, 200)
        self.assertTrue(result["ok"])
        self.assertEqual(result["config"]["web"]["access_token"], "")
        self.assertTrue(result["secret_fields"]["web.access_token"])

        stored = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(stored["web"]["access_token"], "updated-token")

    def test_access_token_rejects_blank_or_non_ascii(self):
        for value in ("", "新的令牌"):
            with self.assertRaises(urllib.error.HTTPError) as context:
                self.request(
                    "/api/access-token",
                    method="POST",
                    payload={"new_token": value},
                )
            self.assertEqual(context.exception.code, 400)

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
        self.assertTrue(result["restart_required"])
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

    def test_invalid_reasoning_effort_is_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as context:
            self.request(
                "/api/config",
                method="POST",
                payload={"patch": {"llm": {"reasoning_effort": "ultra"}}},
            )
        self.assertEqual(context.exception.code, 400)

    def test_remote_control_patch_is_saved_independently(self):
        before = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        status, result = self.request(
            "/api/config",
            method="POST",
            payload={"patch": {"remote_control": {"servo_step_size": 55.5, "update_rate_hz": 30}}},
        )
        self.assertEqual(status, 200)
        self.assertTrue(result["ok"])
        self.assertFalse(result["restart_required"])
        self.assertIn("自动生效", result["message"])
        stored = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(stored["remote_control"], {"servo_step_size": 55.5, "update_rate_hz": 30})
        self.assertEqual(stored["servos"], before["servos"])
        self.assertEqual(stored["motors"], before["motors"])

    def test_legacy_config_without_remote_control_can_still_be_saved(self):
        legacy = sample_config()
        del legacy["remote_control"]
        self.config_path.write_text(
            yaml.safe_dump(legacy, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        status, result = self.request(
            "/api/config",
            method="POST",
            payload={"patch": {"launch": {"tracking": True}}},
        )
        self.assertEqual(status, 200)
        self.assertTrue(result["ok"])
        stored = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.assertNotIn("remote_control", stored)
        self.assertTrue(stored["launch"]["tracking"])

    def test_invalid_remote_control_rate_is_rejected(self):
        before = self.config_path.read_text(encoding="utf-8")
        with self.assertRaises(urllib.error.HTTPError) as context:
            self.request(
                "/api/config",
                method="POST",
                payload={"patch": {"remote_control": {"update_rate_hz": 0}}},
            )
        self.assertEqual(context.exception.code, 400)
        self.assertEqual(self.config_path.read_text(encoding="utf-8"), before)

    @patch("services.web_server.list_usb_devices")
    def test_usb_scan_returns_current_physical_devices(self, list_devices):
        list_devices.return_value = [
            {
                "id": "303a:1001:abc",
                "label": "WALL-E USB (303a:1001, SN abc)",
                "selector": {
                    "vendor_id": "303a",
                    "product_id": "1001",
                    "serial_number": "abc",
                },
                "interfaces": {"serial": ["/dev/ttyACM0"], "video": [], "audio_cards": []},
            }
        ]
        status, body = self.request("/api/usb-devices")
        self.assertEqual(status, 200)
        self.assertEqual(body["devices"][0]["selector"]["serial_number"], "abc")

    def test_usb_roles_save_without_restart_and_can_be_cleared(self):
        selector = {
            "vendor_id": "303a",
            "product_id": "1001",
            "serial_number": "screen-1",
        }
        status, result = self.request(
            "/api/config",
            method="POST",
            payload={"patch": {"usb_devices": {"screen_motion": selector}}},
        )
        self.assertEqual(status, 200)
        self.assertFalse(result["restart_required"])
        stored = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(stored["usb_devices"]["screen_motion"], selector)

        self.request(
            "/api/config",
            method="POST",
            payload={"patch": {"usb_devices": {}}},
        )
        stored = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(stored["usb_devices"], {})

    def test_invalid_usb_selector_is_rejected(self):
        before = self.config_path.read_text(encoding="utf-8")
        with self.assertRaises(urllib.error.HTTPError) as context:
            self.request(
                "/api/config",
                method="POST",
                payload={
                    "patch": {
                        "usb_devices": {
                            "camera": {"vendor_id": "bad", "product_id": "0001"}
                        }
                    }
                },
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
                token=None,
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
