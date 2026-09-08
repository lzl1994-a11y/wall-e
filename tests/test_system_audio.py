import json
from pathlib import Path
import tempfile
import time
import unittest
import wave

from services.esp32_network_prompt import Esp32NetworkPromptSelector
from services.system_audio_protocol import decode_system_audio, encode_system_audio


def _write_prompt(path: Path, value: int) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(48000)
        output.writeframes(value.to_bytes(2, "little", signed=True) * 32)


class SystemAudioProtocolTests(unittest.TestCase):
    def test_round_trip_preserves_identity_cue_and_pcm(self):
        encoded = encode_system_audio("request-1", "network", b"\x01\x00\x02\x00")
        self.assertEqual(
            decode_system_audio(encoded),
            ("request-1", "network", b"\x01\x00\x02\x00"),
        )

    def test_rejects_empty_or_incomplete_audio(self):
        with self.assertRaises(ValueError):
            encode_system_audio("request-1", "network", b"")
        with self.assertRaises(ValueError):
            decode_system_audio(
                '{"request_id":"request-1","cue":"network","audio_base64":"AQ=="}'
            )


class Esp32NetworkPromptSelectorTests(unittest.TestCase):
    def test_bundled_prompts_have_playback_format(self):
        selector = Esp32NetworkPromptSelector(Path(__file__).resolve().parents[1] / "assets")
        self.assertIsNotNone(selector.select('{"state":"configuring"}'))
        self.assertIsNotNone(selector.select('{"state":"connected"}'))

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        _write_prompt(root / "esp_network_configuring.wav", 1)
        _write_prompt(root / "esp_network_connected.wav", 2)
        self.selector = Esp32NetworkPromptSelector(root)

    def tearDown(self):
        self.directory.cleanup()

    def test_plays_each_phase_once_for_same_request(self):
        configuring = json.dumps({"state": "configuring", "request_id": "net-1"})
        connected = json.dumps({"state": "connected", "request_id": "net-1"})

        self.assertEqual(
            self.selector.select(configuring)[:2],
            ("net-1", "esp32_net_configuring"),
        )
        self.assertIsNone(self.selector.select(configuring))
        self.assertEqual(
            self.selector.select(connected)[:2],
            ("net-1", "esp32_net_connected"),
        )
        self.assertIsNone(self.selector.select(connected))

    def test_failed_is_silent_but_allows_a_new_unidentified_transition(self):
        configuring = '{"state":"configuring"}'
        failed = '{"state":"failed","detail":"timeout"}'
        self.assertIsNotNone(self.selector.select(configuring))
        self.assertIsNone(self.selector.select(configuring))
        self.assertIsNone(self.selector.select(failed))
        self.assertIsNotNone(self.selector.select(configuring))

    def test_timestamped_retained_status_is_ignored_when_stale(self):
        stale = json.dumps(
            {
                "state": "connected",
                "request_id": "old-session",
                "published_at": time.time() - 60,
            }
        )
        self.assertIsNone(self.selector.select(stale))

    def test_ignores_invalid_and_unrelated_status(self):
        for message in ("", "[]", "{}", '{"state":"failed"}', '{"state":3}'):
            self.assertIsNone(self.selector.select(message))


if __name__ == "__main__":
    unittest.main()
