import ctypes
import unittest

from services.libretro_audio import LibretroAudioPlayer


class _FakeStream:
    def start(self):
        pass

    def abort(self):
        pass

    def close(self):
        pass


class _FakeSounddevice:
    def query_devices(self):
        return [{"max_output_channels": 2}]

    def RawOutputStream(self, **_kwargs):
        return _FakeStream()


class LibretroAudioPlayerTests(unittest.TestCase):
    def test_batches_are_stereo_pcm_and_output_is_zero_padded(self):
        player = LibretroAudioPlayer(sounddevice_module=_FakeSounddevice(), max_buffer_ms=1)
        samples = (ctypes.c_short * 4)(1, 2, 3, 4)
        player.push_batch(samples, 2)
        output = bytearray(12)
        player._output_callback(output, 3, None, None)
        self.assertEqual(output[:8], ctypes.string_at(samples, 8))
        self.assertEqual(output[8:], b"\x00" * 4)

    def test_buffer_is_bounded(self):
        player = LibretroAudioPlayer(
            sounddevice_module=_FakeSounddevice(), sample_rate=1_000, max_buffer_ms=1
        )
        samples = (ctypes.c_short * 4)(1, 2, 3, 4)
        player.push_batch(samples, 2)
        self.assertEqual(len(player._buffer), 4)


if __name__ == "__main__":
    unittest.main()
