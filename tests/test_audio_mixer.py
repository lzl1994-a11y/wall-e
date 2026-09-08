import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from services.audio_mixer import AudioMixer
from services.mixing_playback_service import MixingPlaybackService


class AudioMixerTests(unittest.TestCase):
    def test_music_advances_during_speech_and_recovers_after_completion(self):
        mixer = AudioMixer(sample_rate=1000)
        music = np.arange(1, 1001, dtype=np.int16) * 10
        mixer.play_music(music)
        mixer.play(np.full(100, 8000, dtype=np.int16))
        mixer.end_speech()
        blocks, completions = [], []
        for index in range(35):
            block, done = mixer.render(20)
            blocks.append(block)
            completions.extend((index, token) for token in done)
        output = np.concatenate(blocks)
        # At 40 ms the attack has settled; both voices are in the SAME samples.
        np.testing.assert_allclose(output[40:100], (8000 + music[40:100] * 0.2) / 32768, atol=1e-7)
        self.assertEqual(len(completions), 1)
        self.assertEqual(completions[0][1], "dialogue")
        # Music has continued to advance and regained full volume after release.
        np.testing.assert_allclose(output[600:700], music[600:700] / 32768, atol=1e-7)
        self.assertTrue(mixer.music.active)

    def test_ends_wait_for_output_latency_without_waiting_for_music_end(self):
        mixer = AudioMixer(sample_rate=1000)
        mixer.play_music(np.full(1000, 5000, dtype=np.int16))
        mixer.play(np.full(20, 1000, dtype=np.int16))
        mixer.end_speech("turn")
        completed = []
        for index in range(20):
            _, done = mixer.render(20, output_latency=0.2)
            if done:
                completed.append(index * 20)
        self.assertEqual(len(completed), 1)
        self.assertGreaterEqual(completed[0], 20 + 200)
        self.assertTrue(mixer.music.active)

    def test_music_end_does_not_end_or_cut_off_speech(self):
        mixer = AudioMixer(sample_rate=1000)
        mixer.play_music(np.full(20, 2000, dtype=np.int16))
        mixer.end_music()
        mixer.play(np.full(100, 10000, dtype=np.int16))
        mixer.end_speech("dialogue")
        mixer.render(20)
        audio, completed = mixer.render(20)
        np.testing.assert_allclose(audio, 10000 / 32768)
        self.assertEqual(completed, [])
        self.assertTrue(mixer.active)

    def test_speech_delivery_gaps_keep_music_running_and_ducked(self):
        mixer = AudioMixer(sample_rate=1000)
        mixer.play_music(np.full(1000, 10000, dtype=np.int16))
        mixer.play(np.full(20, 1000, dtype=np.int16))
        mixer.render(20)
        mixer.render(20)
        gap, done = mixer.render(20)
        np.testing.assert_allclose(gap, 2000 / 32768)
        self.assertFalse(done)
        mixer.play(np.full(20, 3000, dtype=np.int16))
        audio, _ = mixer.render(20)
        np.testing.assert_allclose(audio, 5000 / 32768)

    def test_loud_mix_saturates_without_int16_wraparound(self):
        mixer = AudioMixer(sample_rate=1000)
        mixer.play_music(np.full(100, 32767, dtype=np.int16))
        mixer.play(np.full(100, 32767, dtype=np.int16))
        audio, _ = mixer.render(100)
        self.assertTrue(np.all(audio == 1.0))

    def test_worker_uses_one_stream_and_distinct_wake_and_dialogue_acks(self):
        events = []
        with patch("services.playback_service.threading.Thread"):
            player = MixingPlaybackService(
                sample_rate=1000,
                on_turn_complete=lambda: events.append("dialogue"),
                on_wake_complete=lambda request_id: events.append(request_id),
            )
        stream = MagicMock()
        stream.latency = 0.04
        stream.write.side_effect = lambda _audio: events.append("write")
        player._stream = stream
        player.play_music(np.full(3000, 1000, dtype=np.int16))
        player.play_wake(np.full(40, 3000, dtype=np.int16), "wake-1")
        player.play(np.full(40, 2000, dtype=np.int16))
        player.mark_turn_end()
        for _ in range(30):
            player._play_mix_block()
        self.assertEqual([event for event in events if event != "write"], ["wake-1", "dialogue"])
        for index, event in enumerate(events):
            if event != "write":
                self.assertEqual(events[index - 1], "write")
        stream.stop.assert_not_called()
        stream.close.assert_not_called()

    def test_missing_speaker_still_acknowledges_turns_and_wake(self):
        events = []
        with patch("services.playback_service.threading.Thread"):
            player = MixingPlaybackService(
                sample_rate=1000,
                on_turn_complete=lambda: events.append("dialogue"),
                on_wake_complete=lambda value: events.append(value),
            )
        player._stopped.wait = MagicMock()
        player.play_wake(np.ones(20, dtype=np.int16), "wake-2")
        player.mark_turn_end()
        with patch.object(player, "_ensure_stream", side_effect=RuntimeError("USB offline")):
            for _ in range(25):
                player._play_mix_block()
        self.assertEqual(events, ["wake-2", "dialogue"])


if __name__ == "__main__":
    unittest.main()
