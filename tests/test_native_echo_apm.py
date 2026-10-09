"""Opt-in real system DSP checks, with synthetic signals only."""
import os
import unittest

import numpy as np

from services.audio.native_apm import NativeEchoApm


@unittest.skipUnless(os.environ.get("WALLE_TEST_NATIVE_AEC") == "1", "native AEC library required")
class NativeEchoApmTests(unittest.TestCase):
    def test_delayed_echo_is_removed_while_independent_near_signal_survives(self):
        rate, seconds = 48000, 14
        count = rate * seconds
        random = np.random.default_rng(48)

        def signal():
            spectrum = np.fft.rfft(random.normal(0, 1, count))
            frequencies = np.fft.rfftfreq(count, 1 / rate)
            spectrum[(frequencies < 180) | (frequencies > 3300)] = 0
            audio = np.fft.irfft(spectrum, count)
            audio /= np.std(audio)
            envelope = .1 + .9 * np.maximum(0, np.sin(2 * np.pi * 2.3 * np.arange(count) / rate))
            return audio * envelope * 4500

        far, near = signal(), signal() * .65
        near[:rate * 8] = 0
        delay = round(.12 * rate)
        echo = np.r_[np.zeros(delay), far[:-delay]] * .45
        mic = np.clip(echo + near, -32000, 32000).astype(np.int16)
        far = far.astype(np.int16)
        processors = [NativeEchoApm(0), NativeEchoApm(0)]
        outputs = [[], []]
        try:
            for offset in range(0, count, 960):
                pcm = mic[offset:offset + 960].tobytes()
                for index, processor in enumerate(processors):
                    processor.render(far[offset:offset + 960].tobytes() if index == 0 else bytes(1920))
                    outputs[index].append(np.frombuffer(processor.capture(pcm, 120), np.int16).copy())
        finally:
            for processor in processors:
                processor.close()
        cancelled, baseline = [np.concatenate(output).astype(float) for output in outputs]
        window = slice(rate * 4, rate * 8)
        rms = lambda audio: np.sqrt(np.mean(audio * audio))
        reduction = 20 * np.log10(rms(baseline[window]) / max(1, rms(cancelled[window])))
        self.assertGreater(reduction, 20)
        # Filter latency is included; search +/-30 ms rather than requiring
        # waveform identity after noise suppression and the digital compressor.
        clean_near = near[rate * 10:rate * 13]
        correlation = max(np.corrcoef(clean_near, cancelled[rate * 10 + lag:rate * 13 + lag])[0, 1]
                          for lag in range(-1440, 1441, 48))
        self.assertGreater(correlation, .65)
        self.assertGreater(rms(cancelled[rate * 10:rate * 13]), 500)


if __name__ == "__main__":
    unittest.main()
