import math
import socket
import sys
import time
import unittest
import uuid
from unittest.mock import Mock, patch

import numpy as np

from services.audio.audio_apm import WebRTCApm
from services.audio.echo_reference import (
    FRAME_BYTES, HEADER, MAGIC, EchoReferenceReceiver, EchoReferenceSender, decode_reference,
)
from services.audio.mixing_playback_service import MixingPlaybackService
from services.audio.native_apm import NativeEchoApm


class EchoReferenceTests(unittest.TestCase):
    def test_stale_future_truncated_and_invalid_reference_are_rejected(self):
        pcm = bytes(FRAME_BYTES)
        for sent, delay, magic, payload in (
            (99.7, 80, MAGIC, pcm), (100.1, 80, MAGIC, pcm),
            (100, 501, MAGIC, pcm), (100, 80, b"junk", pcm),
            (100, 80, MAGIC, pcm[:-1]), (math.nan, 80, MAGIC, pcm),
        ):
            self.assertIsNone(decode_reference(HEADER.pack(magic, sent, delay) + payload, 100))
        self.assertEqual(decode_reference(HEADER.pack(MAGIC, 99.9, 80) + pcm, 100), (pcm, 80, 99.9))
        self.assertEqual(decode_reference(HEADER.pack(MAGIC, 100, 0), 100), (b"", 0, 100))

    def test_reference_tracks_final_mix_after_successful_write(self):
        with patch("services.audio.playback_service.threading.Thread"):
            player = MixingPlaybackService(sample_rate=1000)
        events = []
        player._echo_reference = Mock()
        player._echo_reference.send.side_effect = lambda audio, delay: events.append((audio.copy(), delay))
        player._stream = Mock(latency=0.08)
        player._stream.write.side_effect = lambda audio: events.append("write")
        player.play_music(np.full(200, 10000, dtype=np.int16))
        player.play(np.full(100, 3000, dtype=np.int16))
        player._play_mix_block()
        self.assertEqual(events[0], "write")
        np.testing.assert_array_equal(events[1][0], player._stream.write.call_args.args[0].reshape(-1))
        self.assertEqual(events[1][1], 0.08)
        player._stream.write.side_effect = RuntimeError("device unplugged")
        with self.assertRaises(RuntimeError):
            player._play_mix_block()
        self.assertEqual(player._echo_reference.send.call_count, 1)

    def test_abort_and_drain_notify_reference_only_when_stream_existed(self):
        with patch("services.audio.playback_service.threading.Thread"):
            player = MixingPlaybackService(sample_rate=1000)
        player._echo_reference = Mock()
        stream = player._stream = Mock()
        player._close_stream(drain=False)
        stream.abort.assert_called_once()
        player._echo_reference.end.assert_called_once()
        player._close_stream(drain=True)
        player._echo_reference.end.assert_called_once()

    def test_native_api_rejects_partial_frames_before_calling_cpp(self):
        for pcm in (b"", bytes(959), bytes(961)):
            with self.assertRaises(ValueError):
                NativeEchoApm._samples(pcm)
        self.assertEqual(NativeEchoApm._samples(bytes(1920)), 960)

    def test_ear_s3_reference_includes_firmware_gain(self):
        sender = EchoReferenceSender.__new__(EchoReferenceSender)
        sender._send = Mock()
        sender.configure_device("Walle Ear S3: USB Audio (hw:0,0)")
        sender.send(np.full(960, .1, dtype=np.float32), .08)
        self.assertTrue(np.all(np.frombuffer(sender._send.call_args.args[0], "<i2") == 13107))
        sender.configure_device("Other USB Audio")
        sender.send(np.full(960, .1, dtype=np.float32), .08)
        self.assertTrue(np.all(np.frombuffer(sender._send.call_args.args[0], "<i2") == 3277))

    def test_loud_ear_s3_mix_is_limited_before_dac_and_reference(self):
        with patch("services.audio.playback_service.threading.Thread"):
            player = MixingPlaybackService(sample_rate=1000)
        player._echo_reference = Mock(device_gain=4)
        player._stream = Mock(latency=.08)
        player.play(np.full(20, 20000, np.int16))
        player._play_mix_block()
        actual = player._stream.write.call_args.args[0].reshape(-1)
        self.assertLessEqual(float(np.max(np.abs(actual))) * 4, 32767 / 32768 + 1e-7)
        np.testing.assert_array_equal(player._echo_reference.send.call_args.args[0], actual)
        previous_gain = player._speaker_gain
        player.play(np.full(20, 1000, np.int16))
        player._play_mix_block()
        self.assertAlmostEqual(player._speaker_gain, previous_gain + .1)

    def test_native_aec_start_failure_is_explicit_and_releases_processor(self):
        apm = WebRTCApm(lambda _: None, output_rate=48000, frame_ms=20, echo_cancel=True)
        native = Mock()
        with patch("services.audio.native_apm.NativeEchoApm", return_value=native), patch(
            "services.audio.echo_reference.EchoReferenceReceiver", side_effect=OSError("address in use")
        ):
            self.assertFalse(apm.start(48000))
        native.close.assert_called_once()
        self.assertFalse(apm._running)

    def test_native_worker_passes_mic_during_remote_speech_without_muting(self):
        output = []
        apm = WebRTCApm(output.append, output_rate=48000, frame_ms=20, echo_cancel=True)
        apm._running = True
        apm._native = Mock()
        apm._native.capture.return_value = b"processed near-end speech"
        apm._reference = Mock(frames=1, dropped=0)
        apm._reference.drain.return_value = [(bytes(1920), 80, 100.0)]
        apm._queue.put(bytes(1920))
        apm._queue.put(None)
        with patch("services.audio.audio_apm.time.monotonic", return_value=100.0):
            apm._write_native()
        self.assertEqual(output, [b"processed near-end speech"])
        apm._native.render.assert_called_once_with(bytes(1920))
        apm._native.capture.assert_called_once_with(bytes(1920), 100)
        apm._native.close.assert_called_once()
        apm._reference.close.assert_called_once()

    def test_native_delay_includes_capture_buffer_and_processing_queue(self):
        apm = WebRTCApm(lambda _: None, output_rate=48000, frame_ms=20,
                       echo_cancel=True, capture_delay_ms=60)
        apm._running = True
        apm._native = Mock()
        apm._reference = Mock(frames=1, dropped=0)
        apm._reference.drain.return_value = [(bytes(1920), 80, 100.0)]
        apm._queue.put((bytes(1920), 99.96))
        apm._queue.put(None)
        with patch("services.audio.audio_apm.time.monotonic", return_value=100.0):
            apm._write_native()
        apm._native.capture.assert_called_once_with(bytes(1920), 180)

    @unittest.skipUnless(sys.platform == "linux", "Linux abstract UNIX socket/credentials")
    def test_local_ipc_pcm_saturation_end_and_restart(self):
        with patch("services.audio.echo_reference.address", return_value="\0walle.echo-test." + uuid.uuid4().hex):
            self._check_local_ipc()

    def _check_local_ipc(self):
        receiver = EchoReferenceReceiver()
        sender = EchoReferenceSender()
        try:
            sender.send(np.ones(960, dtype=np.float32), 0.08)
            sender.end()
            packets = receiver.drain()
            self.assertEqual(len(packets), 2)
            self.assertTrue(np.all(np.frombuffer(packets[0][0], dtype="<i2") == 32767))
            self.assertEqual(packets[1][0], b"")
            receiver.close()
            receiver = EchoReferenceReceiver()
            sender.send(np.zeros(960, dtype=np.float32), 0.08)
            self.assertEqual(len(receiver.drain()), 1)
        finally:
            receiver.close()
            sender.close()


if __name__ == "__main__":
    unittest.main()
