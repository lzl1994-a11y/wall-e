import tempfile
import unittest
from pathlib import Path

import yaml

from services.llm.llm_request_options import reasoning_request_options
from services.llm.multimodal import create_multimodal
from services.llm.multimodal.siliconflow_multimodal import (
    SiliconFlowMultimodal,
)


class SiliconFlowMultimodalTests(unittest.TestCase):
    def test_builds_documented_audio_url_content(self):
        message = SiliconFlowMultimodal(
            model="Qwen/Qwen3.5-397B-A17B"
        ).build_audio_message("aGVsbG8=")

        self.assertEqual(message["role"], "user")
        self.assertEqual(message["content"][0]["type"], "text")
        self.assertEqual(message["content"][1], {
            "type": "audio_url",
            "audio_url": {
                "url": "data:audio/wav;base64,aGVsbG8=",
            },
        })

    def test_kimi_k2_6_explains_that_asr_pipeline_is_required(self):
        adapter = SiliconFlowMultimodal(model="Pro/moonshotai/Kimi-K2.6")

        with self.assertRaisesRegex(NotImplementedError, "asr_llm"):
            adapter.build_audio_message("aGVsbG8=")

    def test_factory_selects_siliconflow_adapter_and_passes_model(self):
        with tempfile.TemporaryDirectory(prefix="wali-siliconflow-") as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(
                yaml.safe_dump({
                    "llm": {
                        "provider": "siliconflow",
                        "model": "Pro/moonshotai/Kimi-K2.6",
                    }
                }),
                encoding="utf-8",
            )

            adapter = create_multimodal(str(config_path))

        self.assertIsInstance(adapter, SiliconFlowMultimodal)
        self.assertEqual(adapter.model, "Pro/moonshotai/Kimi-K2.6")

    def test_fast_mode_uses_siliconflow_thinking_switch(self):
        options = reasoning_request_options({
            "provider": "siliconflow",
            "model": "Pro/moonshotai/Kimi-K2.6",
            "reasoning_effort": "fast",
        })

        self.assertEqual(options, {"extra_body": {"enable_thinking": False}})

    def test_default_mode_keeps_provider_default(self):
        options = reasoning_request_options({
            "provider": "siliconflow",
            "model": "Pro/moonshotai/Kimi-K2.6",
            "reasoning_effort": "default",
        })

        self.assertEqual(options, {})


if __name__ == "__main__":
    unittest.main()
