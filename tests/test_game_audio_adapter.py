import ctypes
import unittest

from services.game_audio_adapter import GamePlaybackAdapter


class _Playback:
    def __init__(self):
        self.items = []
        self.ended = False

    def play(self, samples):
        self.items.append(samples)

    def mark_turn_end(self):
        self.ended = True


class GamePlaybackAdapterTests(unittest.TestCase):
    def test_downmixes_stereo_and_applies_gain(self):
        playback = _Playback()
        adapter = GamePlaybackAdapter(playback, gain=0.5)
        samples = (ctypes.c_short * 4)(1000, -1000, 12000, 4000)
        adapter.push_batch(samples, 2)
        self.assertEqual(playback.items[0].tolist(), [0, 4000])

    def test_close_marks_the_existing_player_turn_complete(self):
        playback = _Playback()
        adapter = GamePlaybackAdapter(playback)
        adapter.close()
        adapter.close()
        self.assertTrue(playback.ended)

    def test_close_rejects_any_late_pcm(self):
        playback = _Playback()
        adapter = GamePlaybackAdapter(playback)
        adapter.close()
        adapter.push_sample(1000, 1000)
        self.assertEqual(playback.items, [])


if __name__ == "__main__":
    unittest.main()
