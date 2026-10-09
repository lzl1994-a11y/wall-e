"""Release/disconnect must leave the USB speaker's retained DMA silent."""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from services.audio.mixing_playback_service import MixingPlaybackService


class RetainedDmaSink:
    """Closing ALSA does not clear an independently clocked USB DAC ring."""

    latency = 0.08

    def __init__(self):
        self.ring = np.full(256, 0.2, np.float32)
        self.cursor = 0
        self.events = []
        self.frames_written = 0

    def write(self, audio):
        samples = np.asarray(audio).reshape(-1)
        for value in samples:
            self.ring[self.cursor % len(self.ring)] = value
            self.cursor += 1
        self.frames_written += len(samples)
        self.events.append("silence" if not samples.any() else "audio")
        return False

    def abort(self):
        self.events.append("abort")

    def start(self):
        self.events.append("start")

    def stop(self):
        self.events.append("stop")

    def close(self):
        self.events.append("close")


class PlaybackStopSilenceTests(unittest.TestCase):
    def player(self):
        with patch("services.audio.playback_service.threading.Thread"):
            return MixingPlaybackService(sample_rate=1000)

    def test_abort_clears_retained_audio_before_closing(self):
        player = self.player()
        sink = player._stream = RetainedDmaSink()
        player._close_stream(drain=False)
        self.assertFalse(sink.ring.any(), "old speech repeats after UAC closes")
        self.assertEqual(sink.events[:2], ["abort", "start"])
        self.assertEqual(sink.events[-2:], ["stop", "close"])
        self.assertEqual(sink.frames_written, 360)
        self.assertIsNone(player._stream)

    def test_natural_end_also_clears_retained_audio(self):
        player = self.player()
        sink = player._stream = RetainedDmaSink()
        player._close_stream(drain=True)
        self.assertFalse(sink.ring.any())
        self.assertNotIn("abort", sink.events)
        self.assertNotIn("start", sink.events)

    def test_idle_worker_handles_stop_before_drain_and_consumes_interrupt(self):
        player = self.player()
        sink = player._stream = RetainedDmaSink()
        player.stop_speech()
        # Run exactly one production worker iteration and its shutdown cleanup.
        player._stopped.is_set = Mock(side_effect=[False, True])
        player._play_worker()
        self.assertEqual(sink.events[:2], ["abort", "start"])
        self.assertFalse(player._interrupt_requested.is_set())
        self.assertFalse(sink.ring.any())

    def test_already_closed_speaker_does_not_open_on_stop(self):
        player = self.player()
        player.stop_speech()
        with patch.object(player, "_ensure_stream") as ensure:
            player._play_mix_block()
            player._close_stream(drain=True)
        ensure.assert_not_called()
        self.assertFalse(player._interrupt_requested.is_set())

    def test_stop_keeps_background_music_on_same_stream(self):
        player = self.player()
        sink = player._stream = RetainedDmaSink()
        player.play_music(np.full(100, 2000, np.int16))
        player.play(np.full(100, 9000, np.int16))
        player.stop_speech()
        player._play_mix_block()
        self.assertIs(player._stream, sink)
        self.assertEqual(sink.events, ["abort", "start", "audio"])
        np.testing.assert_allclose(sink.ring[:20], 2000 / 32768)
        self.assertTrue(player._mixer.music.active)

    def test_second_turn_arriving_during_silence_is_preserved(self):
        player = self.player()
        sink = player._stream = Mock(latency=.08)
        # Inject the new turn from the first silence write, as the ROS callback can.
        sink.write.side_effect = lambda _: player.play_realtime(np.full(20, 2000, np.int16)) if not player._mixer.active else None
        player._close_stream(drain=False)
        new_sink = player._stream = Mock(latency=.08)
        player._play_mix_block()
        np.testing.assert_allclose(new_sink.write.call_args.args[0], 2000 / 32768)

    def test_second_turn_already_queued_resumes_without_shutdown_gap(self):
        player = self.player()
        sink = player._stream = RetainedDmaSink()
        player.play_realtime(np.full(20, 9000, np.int16))
        player.stop_speech()
        player.play_realtime(np.full(20, 2000, np.int16))
        player._play_mix_block()
        self.assertEqual(sink.events, ["abort", "start", "audio"])
        np.testing.assert_allclose(sink.ring[:20], 2000 / 32768)

    def test_production_silence_is_full_20ms_reference_blocks(self):
        with patch("services.audio.playback_service.threading.Thread"):
            player = MixingPlaybackService(sample_rate=48000)
        sink = player._stream = Mock(latency=.08)
        player._close_stream(drain=True)
        self.assertEqual(sink.write.call_count, 18)
        for call in sink.write.call_args_list:
            self.assertEqual(call.args[0].shape, (960, 1))
            self.assertEqual(call.args[0].dtype, np.float32)
            self.assertFalse(call.args[0].any())

    def test_silence_reaches_echo_reference_before_end(self):
        player = self.player()
        events = []
        sink = player._stream = Mock(latency=.08)
        sink.write.side_effect = lambda audio: events.append("write")
        sink.stop.side_effect = lambda: events.append("drained")
        reference = player._echo_reference = Mock()
        reference.send.side_effect = lambda audio, delay: events.append("reference")
        reference.end.side_effect = lambda: events.append("end")
        player._close_stream(drain=False)
        self.assertEqual(events, ["write", "reference"] * 18 + ["drained", "end"])
        for call in reference.send.call_args_list:
            self.assertEqual(call.args[0].shape, (20,))
            self.assertFalse(call.args[0].any())
            self.assertEqual(call.args[1], .08)

    def test_device_write_failure_still_closes_and_ends_reference(self):
        player = self.player()
        sink = player._stream = Mock(latency=.08)
        sink.write.side_effect = RuntimeError("USB unplugged")
        player._echo_reference = Mock()
        with self.assertRaisesRegex(RuntimeError, "USB unplugged"):
            player._close_stream(drain=True)
        sink.close.assert_called_once()
        player._echo_reference.end.assert_called_once()
        player._echo_reference.send.assert_not_called()
        self.assertIsNone(player._stream)

    def test_unplug_during_playback_and_flush_does_not_kill_worker(self):
        player = self.player()
        sink = player._stream = Mock(latency=.08)
        sink.write.side_effect = RuntimeError("USB unplugged")
        player.play_realtime(np.ones(20, np.int16))
        player._stopped.is_set = Mock(side_effect=[False, True])
        player._stopped.wait = Mock()
        player._play_worker()
        sink.close.assert_called_once()
        self.assertIsNone(player._stream)
        player._stopped.wait.assert_called_once_with(player.BLOCK_SEC)


if __name__ == "__main__":
    unittest.main()
