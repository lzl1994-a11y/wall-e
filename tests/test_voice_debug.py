import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.voice_debug import RollingVoiceDebugStore, voice_debug_enabled


class VoiceDebugTests(unittest.TestCase):
    def test_disabled_store_does_not_create_directory(self):
        with tempfile.TemporaryDirectory(prefix="wali-voice-debug-") as temp_dir:
            root = Path(temp_dir) / "debug"
            store = RollingVoiceDebugStore(enabled=False, root=root)

            self.assertIsNone(store.save_json("llm_input", {"prompt": "hello"}))
            self.assertFalse(root.exists())

    def test_store_keeps_latest_twenty_files_per_group(self):
        with tempfile.TemporaryDirectory(prefix="wali-voice-debug-") as temp_dir:
            store = RollingVoiceDebugStore(enabled=True, root=temp_dir, limit=20)
            for index in range(23):
                store.save_json("llm_input", {"index": index})

            files = list((Path(temp_dir) / "llm_input").glob("*.json"))
            values = {json.loads(path.read_text(encoding="utf-8"))["index"] for path in files}
            self.assertEqual(len(files), 20)
            self.assertEqual(values, set(range(3, 23)))

    def test_groups_have_independent_retention(self):
        with tempfile.TemporaryDirectory(prefix="wali-voice-debug-") as temp_dir:
            source = Path(temp_dir) / "source.wav"
            source.write_bytes(b"RIFF")
            store = RollingVoiceDebugStore(enabled=True, root=temp_dir, limit=2)

            for _ in range(3):
                store.save_file("asr_input", source)
            store.save_json("llm_input", {"prompt": "hello"})

            self.assertEqual(len(list((Path(temp_dir) / "asr_input").glob("*.wav"))), 2)
            self.assertEqual(len(list((Path(temp_dir) / "llm_input").glob("*.json"))), 1)

    def test_environment_requires_explicit_truthy_value(self):
        with patch.dict("os.environ", {"WALI_SAVE_VOICE_DEBUG": "0"}):
            self.assertFalse(voice_debug_enabled())
        with patch.dict("os.environ", {"WALI_SAVE_VOICE_DEBUG": "1"}):
            self.assertTrue(voice_debug_enabled())


if __name__ == "__main__":
    unittest.main()
