import tempfile
import unittest
from pathlib import Path

import yaml

from services.llm_request_options import reasoning_request_options
from services.multimodal import create_multimodal
from services.multimodal.tencent_hunyuan_multimodal import (
    TencentHunyuanMultimodal,
)


class TencentHunyuanTests(unittest.TestCase):
    def _create_adapter(self, provider):
        with tempfile.TemporaryDirectory(prefix="wali-hunyuan-") as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(
                yaml.safe_dump({"llm": {"provider": provider}}),
                encoding="utf-8",
            )
            return create_multimodal(str(config_path))

    def test_factory_accepts_tencent_provider_names(self):
        for provider in ("tencent_hunyuan", "tencent", "hunyuan"):
            with self.subTest(provider=provider):
                self.assertIsInstance(
                    self._create_adapter(provider),
                    TencentHunyuanMultimodal,
                )

    def test_audio_direct_mode_explains_supported_pipeline(self):
        with self.assertRaisesRegex(NotImplementedError, "asr_llm"):
            TencentHunyuanMultimodal().build_audio_message("aGVsbG8=")

    def test_fast_mode_does_not_send_undocumented_thinking_switch(self):
        options = reasoning_request_options({
            "provider": "tencent_hunyuan",
            "model": "hunyuan-turbos-latest",
            "reasoning_effort": "fast",
        })

        self.assertEqual(options, {})
