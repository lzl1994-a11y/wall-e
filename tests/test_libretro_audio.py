import ctypes
import time
import unittest

from services.libretro_audio import LibretroAudioPlayer


class _FakeStream:
    def __init__(self):
        self.writes = []
        self.closed = False

    def start(self):
        pass

    def write(self, data):
        self.writes.append(bytes(data))

    def abort(self):
        pass

    def close(self):
        self.closed = True


class _FakeSounddevice:
    def __init__(self):
        self.stream = _FakeStream()

    def query_devices(self):
        return [{"max_output_channels": 2}]

    def RawOutputStream(self, **_kwargs):
        return self.stream


class LibretroAudioPlayerTests(unittest.TestCase):
    def test_worker_writes_stereo_pcm_after_prebuffering(self):
        sounddevice = _FakeSounddevice()
        player = LibretroAudioPlayer(
            sounddevice_module=sounddevice, sample_rate=1_000, max_buffer_ms=10, prebuffer_ms=2
        )
        player.start()
        samples = (ctypes.c_short * 4)(1, 2, 3, 4)
        player.push_batch(samples, 2)
        deadline = time.monotonic() + 1.0
        while not sounddevice.stream.writes and time.monotonic() < deadline:
            time.sleep(0.01)
        player.close()
        self.assertEqual(sounddevice.stream.writes, [ctypes.string_at(samples, 8)])
        self.assertTrue(sounddevice.stream.closed)

    def test_oldest_pcm_is_dropped_when_ring_is_full(self):
        player = LibretroAudioPlayer(
            sounddevice_module=_FakeSounddevice(), sample_rate=1_000, max_buffer_ms=1, prebuffer_ms=1
        )
        first = (ctypes.c_short * 2)(1, 2)
        second = (ctypes.c_short * 2)(3, 4)
        player.push_batch(first, 1)
        player.push_batch(second, 1)
        self.assertEqual(player._queued_bytes, 4)
        self.assertEqual(player._chunks[0], ctypes.string_at(second, 4))


if __name__ == "__main__":
    unittest.main()
