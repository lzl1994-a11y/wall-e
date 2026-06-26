import sounddevice as sd
import numpy as np
import time

print("Starting audio stream. Please speak into the microphone for 5 seconds...")
print(sd.query_devices())

max_amp = 0.0

def callback(indata, frames, time_info, status):
    global max_amp
    if status:
        print(status)
    
    audio = indata[:, 0]
    m = np.max(np.abs(audio))
    if m > max_amp:
        max_amp = m

stream = sd.InputStream(channels=1, dtype="float32", samplerate=16000, blocksize=480, callback=callback)
stream.start()

time.sleep(5)
stream.stop()
print(f"Done. Max amplitude during 5s: {max_amp}")
