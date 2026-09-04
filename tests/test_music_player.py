import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from services.music_player import MusicPlayer, SpectrumAnalyzer, resolve_track
from services.music_protocol import decode_music_state, encode_music_state
from services.music_spectrum import render_spectrum_frame


class _Process:
    def __init__(self, pcm: bytes, returncode=0):
        self.stdout = io.BytesIO(pcm)
        self.stderr = io.BytesIO(b"decoder failed" if returncode else b"")
        self.returncode = None
        self._final_returncode = returncode

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = self._final_returncode
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


class MusicPlayerTests(unittest.TestCase):
    def test_track_resolution_stays_inside_library(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Blue Sky.mp3").touch()
            (root / "notes.txt").touch()
            self.assertEqual(resolve_track(root, "blue sky").name, "Blue Sky.mp3")
            self.assertEqual(resolve_track(root, "Sky").name, "Blue Sky.mp3")
            with self.assertRaises(ValueError):
                resolve_track(root, "../outside.mp3")

    def test_streams_pcm_and_spectrum_without_owning_audio_hardware(self):
        with tempfile.TemporaryDirectory() as directory:
            track = Path(directory) / "tone.wav"
            track.touch()
            samples = (np.sin(np.arange(2400) * 2 * np.pi * 440 / 48000) * 12000).astype(np.int16)
            audio, spectra, states, ends = [], [], [], []
            player = MusicPlayer(
                directory=directory,
                on_audio=audio.append,
                on_audio_end=lambda: ends.append(True),
                on_spectrum=spectra.append,
                on_state=lambda *value: states.append(value),
                popen_factory=lambda *_args, **_kwargs: _Process(samples.tobytes()),
            )

            player.play("tone")
            deadline = time.monotonic() + 1.0
            while player._thread is not None and time.monotonic() < deadline:
                time.sleep(0.01)

        np.testing.assert_array_equal(audio[0], samples)
        self.assertEqual(len(spectra[0]), 20)
        self.assertEqual([state[0] for state in states], ["loading", "playing", "stopped"])
        self.assertEqual(ends, [True])

    def test_silence_decays_to_zero_and_renderer_matches_tft_frame_contract(self):
        levels = SpectrumAnalyzer().analyze(np.zeros(2400, dtype=np.int16))
        raw, width, height, pitch = render_spectrum_frame(levels)
        self.assertEqual(levels, [0.0] * 20)
        self.assertEqual((width, height, pitch), (240, 240, 960))
        self.assertEqual(len(raw), pitch * height)

    def test_spectrum_plan_is_reused_for_equal_pcm_chunks(self):
        analyzer = SpectrumAnalyzer()
        samples = np.zeros(4800, dtype=np.int16)
        original = np.hanning
        with patch.object(np, "hanning", wraps=original) as hanning:
            analyzer.analyze(samples)
            analyzer.analyze(samples)
        hanning.assert_called_once_with(4800)

    def test_music_state_codec_rejects_unknown_states(self):
        self.assertEqual(
            decode_music_state(encode_music_state("playing", "tone"))["track"],
            "tone",
        )
        self.assertIsNone(decode_music_state('{"state":"paused"}'))


if __name__ == "__main__":
    unittest.main()
