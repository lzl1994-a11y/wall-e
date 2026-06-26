import sounddevice as sd
import numpy as np
import onnxruntime as ort
import time

model_path = "models/silero_vad.onnx"
vad = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
vad_state = np.zeros((2, 1, 128), dtype=np.float32)

print("Please speak for 5 seconds (with 50x GAIN)...")

max_prob = 0.0
max_amp = 0.0

def callback(indata, frames, time_info, status):
    global vad_state, max_prob, max_amp
    if status: pass
    
    audio_orig = indata[:, 0] * 50.0 # APPLY 50x GAIN
    audio_orig = np.clip(audio_orig, -1.0, 1.0)
    
    amp = np.max(np.abs(audio_orig))
    if amp > max_amp: max_amp = amp
    
    audio = audio_orig.reshape(1, -1)
    
    try:
        out_prob, out_state = vad.run(None, {
            "input": audio,
            "state": vad_state,
            "sr": np.array(16000, dtype=np.int64),
        })
        vad_state = out_state
        prob = out_prob[0][0]
        if prob > max_prob: max_prob = prob
        
    except Exception as e:
        pass

stream = sd.InputStream(channels=1, dtype="float32", samplerate=16000, blocksize=480, callback=callback)
stream.start()
time.sleep(5)
stream.stop()
print(f"Max Amplitude: {max_amp:.6f}, Max VAD Prob: {max_prob:.6f}")
