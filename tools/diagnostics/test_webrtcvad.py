import sounddevice as sd
import numpy as np
import time

try:
    import webrtcvad
    vad = webrtcvad.Vad(3) # Aggressiveness mode 3 (most aggressive filtering)
    print("webrtcvad is installed!")
except ImportError:
    print("webrtcvad is not installed.")
    import sys
    sys.exit(0)

print("Please speak for 5 seconds...")

speech_detected = False

def callback(indata, frames, time_info, status):
    global speech_detected
    if status: pass
    
    # Webrtcvad needs 16-bit PCM bytes
    audio_int16 = (indata[:, 0] * 32767).astype(np.int16)
    
    # Chunk into 30ms frames (480 samples = 960 bytes)
    for i in range(0, len(audio_int16) - 480 + 1, 480):
        chunk = audio_int16[i:i+480].tobytes()
        if vad.is_speech(chunk, 16000):
            speech_detected = True
            print("Speech detected by webrtcvad!")

stream = sd.InputStream(channels=1, dtype="float32", samplerate=16000, blocksize=480, callback=callback)
stream.start()
time.sleep(5)
stream.stop()

if not speech_detected:
    print("No speech detected by webrtcvad.")
