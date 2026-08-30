import io
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.alsa_capture import ArecordInputStream


class FakeProcess:
    def __init__(self, pcm):
        self.stdout = io.BytesIO(pcm)
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def kill(self):
        self.returncode = -9


class ArecordInputStreamTests(unittest.TestCase):
    @patch("services.alsa_capture.subprocess.Popen")
    def test_reads_exact_pcm_blocks_without_portaudio(self, popen):
        source = np.arange(8, dtype=np.int16).reshape(4, 2)
        process = FakeProcess(source.tobytes())
        popen.return_value = process
        received = []
        ready = threading.Event()

        def callback(audio, frames, _time_info, _status):
            received.append((audio.copy(), frames))
            ready.set()

        stream = ArecordInputStream(
            device="plughw:3,0",
            channels=2,
            samplerate=48000,
            blocksize=4,
            callback=callback,
        )
        stream.start()
        self.assertTrue(ready.wait(timeout=1.0))
        stream.close()

        command = popen.call_args.args[0]
        self.assertEqual(command[:4], ["arecord", "-q", "-D", "plughw:3,0"])
        self.assertIn("48000", command)
        self.assertEqual(received[0][1], 4)
        np.testing.assert_allclose(received[0][0], source.astype(np.float32) / 32768.0)


if __name__ == "__main__":
    unittest.main()
