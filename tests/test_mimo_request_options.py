import unittest
import tempfile
from pathlib import Path

import yaml

from services.llm_request_options import normalize_tool_choice, reasoning_request_options
from services.multimodal import create_multimodal
from services.multimodal.xiaomi_mimo_multimodal import XiaomiMiMoMultimodal


class MiMoRequestOptionsTests(unittest.TestCase):
    EXPECTED_DISABLED = {
        "extra_body": {"thinking": {"type": "disabled"}}
    }

    def test_fast_mode_disables_thinking_for_mimo_provider(self):
        self.assertEqual(
            reasoning_request_options({
                "provider": "xiaomi_mimo",
                "model": "mimo-v2.5-pro",
                "url": "https://api.xiaomimimo.com/v1",
                "reasoning_effort": "fast",
            }),
            self.EXPECTED_DISABLED,
        )

    def test_existing_mislabeled_config_is_detected_by_model_and_endpoint(self):
        self.assertEqual(
            reasoning_request_options({
                "provider": "tencent_hunyuan",
                "model": "mimo-v2.5-pro",
                "url": "https://api.xiaomimimo.com/v1",
                "reasoning_effort": "fast",
            }),
            self.EXPECTED_DISABLED,
        )

    def test_default_mode_keeps_mimo_provider_default(self):
        self.assertEqual(
            reasoning_request_options({
                "provider": "xiaomi_mimo",
                "model": "mimo-v2.5-pro",
                "url": "https://api.xiaomimimo.com/v1",
                "reasoning_effort": "default",
            }),
            {},
        )

    def test_mimo_asr_and_tts_do_not_receive_chat_thinking_switch(self):
        for model in ("mimo-v2.5-asr", "mimo-v2.5-tts", "mimo-v2.5-tts-voiceclone"):
            with self.subTest(model=model):
                self.assertEqual(
                    reasoning_request_options({
                        "provider": "xiaomi_mimo",
                        "model": model,
                        "url": "https://api.xiaomimimo.com/v1",
                        "reasoning_effort": "fast",
                    }),
                    {},
                )

    def test_mimo_named_tool_choice_is_normalized_to_auto(self):
        named_choice = {
            "type": "function",
            "function": {"name": "direct_answer"},
        }
        self.assertEqual(
            normalize_tool_choice({
                "provider": "xiaomi_mimo",
                "model": "mimo-v2.5-pro",
            }, named_choice),
            "auto",
        )

    def test_other_provider_keeps_named_tool_choice(self):
        named_choice = {
            "type": "function",
            "function": {"name": "direct_answer"},
        }
        self.assertIs(
            normalize_tool_choice({
                "provider": "custom",
                "model": "custom-model",
            }, named_choice),
            named_choice,
        )

    def test_similar_but_unrelated_hostname_is_not_treated_as_mimo(self):
        self.assertEqual(
            reasoning_request_options({
                "provider": "custom",
                "model": "custom-model",
                "url": "https://api.xiaomimimo.com.example.org/v1",
                "reasoning_effort": "fast",
            }),
            {},
        )

    def test_audio_adapter_uses_mimo_data_url_shape(self):
        message = XiaomiMiMoMultimodal().build_audio_message("aGVsbG8=")

        self.assertEqual(message["role"], "user")
        self.assertIn("不得抄写本条提示文字", message["content"][0]["text"])
        self.assertEqual(message["content"][1], {
            "type": "input_audio",
            "input_audio": {
                "data": "data:audio/wav;base64,aGVsbG8=",
            },
        })

    def test_factory_accepts_mimo_provider_names(self):
        for provider in ("xiaomi_mimo", "xiaomi", "mimo"):
            with self.subTest(provider=provider):
                with tempfile.TemporaryDirectory(prefix="wali-mimo-") as directory:
                    config_path = Path(directory) / "config.yaml"
                    config_path.write_text(
                        yaml.safe_dump({"llm": {"provider": provider}}),
                        encoding="utf-8",
                    )
                    adapter = create_multimodal(str(config_path))

                self.assertIsInstance(adapter, XiaomiMiMoMultimodal)


if __name__ == "__main__":
    unittest.main()
