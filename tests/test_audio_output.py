import asyncio
import io
import queue
import sys
import threading
import time
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.audio_output import OUTPUT_SAMPLE_RATE
from services.audio_pipeline import AudioPipeline
from services.playback_service import PlaybackService
from services.tts_service import TTSService


class AudioSampleRateContractTests(unittest.TestCase):
    def test_input_pipeline_remains_16khz(self):
        self.assertEqual(AudioPipeline.SAMPLE_RATE, 16000)

    def test_vad_preroll_keeps_audio_before_first_detected_speech_frame(self):
        pipeline = AudioPipeline.__new__(AudioPipeline)
        pipeline.audio_queue = queue.Queue()
        pipeline._is_running = True
        pipeline._paused_event = threading.Event()
        pipeline._ww = types.SimpleNamespace(enabled=False)
        pipeline._awake = True
        pipeline._vad_thresh = 0.5

        quiet_frame = b"Q" * AudioPipeline.FRAME_BYTES
        speech_frame = b"S" * AudioPipeline.FRAME_BYTES
        trailing_frame = b"T" * AudioPipeline.FRAME_BYTES
        pre_roll_count = int(AudioPipeline.PRE_ROLL_SEC / (AudioPipeline.FRAME_MS / 1000.0))
        max_silence = int(AudioPipeline.SILENCE_SEC / (AudioPipeline.FRAME_MS / 1000.0))
        speech_count = int(
            AudioPipeline.SPEECH_START_MS / AudioPipeline.FRAME_MS
        )
        frames = (
            [quiet_frame] * pre_roll_count
            + [speech_frame] * speech_count
            + [trailing_frame] * (max_silence + 1)
        )
        pipeline.audio_queue.put(b"".join(frames))
        pipeline._vad_prob = lambda frame: 1.0 if frame == speech_frame else 0.0
        emitted = []

        def on_sentence(pcm):
            emitted.append(pcm)
            pipeline._is_running = False

        pipeline.on_sentence = on_sentence
        pipeline._run()

        self.assertEqual(len(emitted), 1)
        self.assertTrue(emitted[0].startswith(quiet_frame * pre_roll_count))
        self.assertIn(speech_frame * speech_count, emitted[0])

    def test_vad_requires_300ms_of_continuous_speech_before_start(self):
        pipeline = AudioPipeline.__new__(AudioPipeline)
        pipeline.audio_queue = queue.Queue()
        pipeline._is_running = True
        pipeline._paused_event = threading.Event()
        pipeline._ww = types.SimpleNamespace(enabled=False)
        pipeline._awake = True
        pipeline._vad_thresh = 0.5

        quiet_frame = b"Q" * AudioPipeline.FRAME_BYTES
        speech_frame = b"S" * AudioPipeline.FRAME_BYTES
        required = int(AudioPipeline.SPEECH_START_MS / AudioPipeline.FRAME_MS)
        frames = [speech_frame] * (required - 1) + [quiet_frame]
        pipeline.audio_queue.put(b"".join(frames))
        pipeline._vad_prob = lambda frame: 1.0 if frame == speech_frame else 0.0
        started = []
        pipeline.on_speech_start = started.append
        pipeline.on_speech_audio = None
        pipeline.on_speech_cancel = None
        pipeline.on_sentence = None

        def stop_after_frames(frame):
            probability = 1.0 if frame == speech_frame else 0.0
            if frame == quiet_frame:
                pipeline._is_running = False
            return probability

        pipeline._vad_prob = stop_after_frames
        pipeline._run()

        self.assertEqual(started, [])

    def test_vad_stream_callbacks_cover_capture_once_in_order(self):
        pipeline = AudioPipeline.__new__(AudioPipeline)
        pipeline.audio_queue = queue.Queue()
        pipeline._is_running = True
        pipeline._paused_event = threading.Event()
        pipeline._ww = types.SimpleNamespace(enabled=False)
        pipeline._awake = True
        pipeline._vad_thresh = 0.5

        quiet_frame = b"Q" * AudioPipeline.FRAME_BYTES
        speech_frame = b"S" * AudioPipeline.FRAME_BYTES
        trailing_frame = b"T" * AudioPipeline.FRAME_BYTES
        pre_roll_count = int(AudioPipeline.PRE_ROLL_SEC / (AudioPipeline.FRAME_MS / 1000.0))
        max_silence = int(AudioPipeline.SILENCE_SEC / (AudioPipeline.FRAME_MS / 1000.0))
        speech_count = int(
            AudioPipeline.SPEECH_START_MS / AudioPipeline.FRAME_MS
        )
        frames = (
            [quiet_frame] * pre_roll_count
            + [speech_frame] * speech_count
            + [trailing_frame] * (max_silence + 1)
        )
        pipeline.audio_queue.put(b"".join(frames))
        pipeline._vad_prob = lambda frame: 1.0 if frame == speech_frame else 0.0
        streamed = []

        pipeline.on_speech_start = streamed.append
        pipeline.on_speech_audio = streamed.append
        pipeline.on_speech_cancel = None

        def on_sentence(_pcm):
            pipeline._is_running = False

        pipeline.on_sentence = on_sentence
        pipeline._run()

        self.assertEqual(b"".join(streamed), b"".join(frames))

    @patch("services.playback_service.threading.Thread")
    @patch.object(PlaybackService, "_refresh_device", return_value=True)
    def test_playback_service_defaults_to_48khz(self, _refresh_device, thread_class):
        player = PlaybackService()

        self.assertEqual(player.sample_rate, OUTPUT_SAMPLE_RATE)
        self.assertEqual(OUTPUT_SAMPLE_RATE, 48000)
        thread_class.return_value.start.assert_called_once_with()

    def test_tts_audio_is_resampled_to_48khz(self):
        service = TTSService.__new__(TTSService)
        service.voice = "test-voice"
        service.rate = "+0%"
        service.pitch = "+0Hz"
        service.sample_rate = OUTPUT_SAMPLE_RATE

        source = MagicMock()
        at_rate = MagicMock()
        mono = MagicMock()
        pcm = MagicMock()
        source.set_frame_rate.return_value = at_rate
        at_rate.set_channels.return_value = mono
        mono.set_sample_width.return_value = pcm
        pcm.get_array_of_samples.return_value = [1, -2, 3]

        class FakeCommunicate:
            def __init__(self, *args, **kwargs):
                pass

            async def stream(self):
                yield {"type": "audio", "data": b"mp3-data"}

        with (
            patch("services.tts_service.edge_tts.Communicate", FakeCommunicate),
            patch("pydub.AudioSegment.from_mp3", return_value=source),
        ):
            samples = asyncio.run(service._download("你好"))

        source.set_frame_rate.assert_called_once_with(48000)
        at_rate.set_channels.assert_called_once_with(1)
        mono.set_sample_width.assert_called_once_with(2)
        np.testing.assert_array_equal(samples, np.array([1, -2, 3], dtype=np.int16))

    def test_tts_streaming_decodes_incremental_mp3_to_48khz_pcm(self):
        service = TTSService(
            voice="test-voice",
            rate="+0%",
            pitch="+0Hz",
            sample_rate=OUTPUT_SAMPLE_RATE,
        )
        pcm = np.arange(4800, dtype=np.int16)
        input_closed = threading.Event()

        class FakeCommunicate:
            def __init__(self, *args, **kwargs):
                pass

            async def stream(self):
                yield {"type": "audio", "data": b"mp3-a"}
                yield {"type": "audio", "data": b"mp3-b"}

        class FakeStdin:
            def __init__(self):
                self.writes = []

            def write(self, data):
                self.writes.append(data)

            def close(self):
                input_closed.set()

        class FakeStdout:
            def __init__(self):
                self._data = io.BytesIO(pcm.tobytes())

            def read(self, size=-1):
                input_closed.wait(timeout=1.0)
                return self._data.read(size)

        class FakeProcess:
            def __init__(self):
                self.stdin = FakeStdin()
                self.stdout = FakeStdout()
                self.stderr = io.BytesIO()

            def wait(self, timeout=None):
                return 0

            def poll(self):
                return 0

            def terminate(self):
                pass

            def kill(self):
                pass

        process = FakeProcess()
        try:
            with (
                patch("services.tts_service.edge_tts.Communicate", FakeCommunicate),
                patch("services.tts_service.subprocess.Popen", return_value=process) as popen,
            ):
                chunks = list(service.synthesize_stream("你好", chunk_ms=100))
        finally:
            service.shutdown()

        np.testing.assert_array_equal(np.concatenate(chunks), pcm)
        self.assertEqual(process.stdin.writes, [b"mp3-a", b"mp3-b"])
        command = popen.call_args.args[0]
        self.assertIn("48000", command)
        self.assertIn("pcm_s16le", command)

    def test_tts_streaming_idle_timeout_flushes_audio_without_waiting_for_edge_end(self):
        service = TTSService(
            voice="test-voice",
            rate="+0%",
            pitch="+0Hz",
            sample_rate=OUTPUT_SAMPLE_RATE,
        )
        pcm = np.arange(960, dtype=np.int16)
        input_closed = threading.Event()

        class HangingCommunicate:
            def __init__(self, *args, **kwargs):
                pass

            async def stream(self):
                yield {"type": "audio", "data": b"mp3-a"}
                await asyncio.Event().wait()

        class FakeStdin:
            def write(self, _data):
                pass

            def close(self):
                input_closed.set()

        class FakeStdout:
            def __init__(self):
                self._sent_audio = False

            def read(self, _size=-1):
                if not self._sent_audio:
                    self._sent_audio = True
                    return pcm.tobytes()
                input_closed.wait(timeout=1.0)
                return b""

        class FakeProcess:
            def __init__(self):
                self.stdin = FakeStdin()
                self.stdout = FakeStdout()
                self.stderr = io.BytesIO()

            def wait(self, timeout=None):
                return 0

            def poll(self):
                return 0 if input_closed.is_set() else None

            def terminate(self):
                input_closed.set()

            def kill(self):
                input_closed.set()

        started = time.monotonic()
        try:
            with (
                patch("services.tts_service.edge_tts.Communicate", HangingCommunicate),
                patch("services.tts_service.subprocess.Popen", return_value=FakeProcess()),
            ):
                chunks = list(
                    service.synthesize_stream(
                        "你好",
                        chunk_ms=20,
                        idle_timeout_sec=0.05,
                    )
                )
        finally:
            service.shutdown()

        self.assertLess(time.monotonic() - started, 1.0)
        self.assertTrue(input_closed.is_set())
        np.testing.assert_array_equal(np.concatenate(chunks), pcm)

    def test_wake_response_asset_is_48khz_pcm_mono(self):
        with wave.open(str(ROOT / "assets" / "wake_response.wav"), "rb") as wav_file:
            self.assertEqual(wav_file.getframerate(), 48000)
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getsampwidth(), 2)


if __name__ == "__main__":
    unittest.main()
