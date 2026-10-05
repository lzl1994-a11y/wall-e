import queue
import threading

from services.audio.audio_pipeline import AudioPipeline


def _session_pipeline():
    pipeline = AudioPipeline.__new__(AudioPipeline)
    pipeline.audio_queue = queue.Queue(maxsize=32)
    pipeline._is_running = True
    pipeline._is_paused = False
    pipeline._paused_event = threading.Event()
    pipeline._ww = type("Wake", (), {"enabled": False})()
    pipeline._awake = False
    pipeline._vad_thresh = 0.5
    pipeline._silence_sec = AudioPipeline.SILENCE_SEC
    pipeline._external_audio_lock = threading.Lock()
    pipeline._external_session_active = False
    pipeline._external_session_end_pending = False
    pipeline._external_turn_active = False
    pipeline._external_turn_buffer = bytearray()
    pipeline._external_turn_max_bytes = AudioPipeline.SAMPLE_RATE * 2 * 30
    pipeline._reset_vad_state = lambda: None
    return pipeline


def test_continuous_remote_audio_is_vad_split_and_end_flushes_tail():
    pipeline = _session_pipeline()
    speech = b"S" * AudioPipeline.FRAME_BYTES
    assert pipeline.begin_external_session()
    pipeline._vad_prob = lambda frame: 1.0 if frame == speech else 0.0

    emitted = []
    session_end = []

    def on_sentence(pcm):
        emitted.append(pcm)
        pipeline._is_running = False

    pipeline.on_sentence = on_sentence
    pipeline.on_external_session_end = lambda: session_end.append(True)
    pipeline.audio_queue.put(speech * 10)
    assert pipeline.end_external_session()
    pipeline._run()

    assert len(emitted) == 1
    assert speech * 10 in emitted[0]
    assert pipeline._external_session_active is False
    assert session_end == [True]


def test_remote_audio_is_bounded_without_blocking_when_queue_is_full():
    pipeline = _session_pipeline()
    assert pipeline.begin_external_session()
    pipeline.audio_queue = queue.Queue(maxsize=1)
    pipeline.audio_queue.put(b"old")

    pipeline.accept_external_pcm(b"new" * 100, sample_rate=16_000)

    assert pipeline.audio_queue.qsize() == 1
    assert pipeline.audio_queue.get_nowait() != b"old"
