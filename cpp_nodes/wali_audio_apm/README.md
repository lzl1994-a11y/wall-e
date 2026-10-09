# Intercom echo cancellation

Build on the robot (Ubuntu 22.04, system `webrtc-audio-processing` 0.3.1):

```sh
apt-get install --no-install-recommends libwebrtc-audio-processing-dev
bash tools/build_audio_apm.sh
WALLE_TEST_NATIVE_AEC=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_native_echo_apm.py tests/test_echo_reference.py
```

The 48 kHz mono intercom uses native AEC, high-pass filtering, moderate noise
suppression and limited digital gain. Processing runs in 10 ms chunks on one
worker; ctypes releases the GIL. The microphone stays live during playback.
The 16 kHz ASR pipeline retains its existing GStreamer processor.

`MixingPlaybackService` supplies the final successfully written mixture through
a bounded, nonblocking local UNIX datagram socket. Only messages from the same
UID are accepted. Old, malformed and oversized references are discarded. An
abort/drain marks the end of the render stream; idle render audio is silence.

The deployed Walle Ear S3 firmware (`i2s_mic_driver_write_tx`) applies saturating
4x gain before its DAC. This device profile models that gain in the reference
and limits mixture peaks before writing so the firmware gain does not clip.
Ordinary USB speakers use a unity reference. Update the profile if the ear
firmware's gain changes.

AEC system delay is actual output latency plus the capture buffer reserve
(`ArecordInputStream.BUFFER_PERIODS - 1` periods), plus microphone processing
queue age. This follows the native `set_stream_delay_ms` contract. Physical
tests showed slow/unstable convergence with the old library's delay-agnostic
mode; explicit device latency worked better. The capture reserve is derived
from the capture backend, not a sleep or a network delay.

Startup/runtime failures and 10-second frame/drop/DSP statistics are logged.
If native AEC fails, the existing capture fallback remains live but has no
echo cancellation; treat that diagnostic as a fault. No recordings are made
by the implementation or regression tests. AEC reduces echo; it cannot undo
microphone ADC clipping or guarantee zero residual in every acoustic setting.
