import numpy as np
import wave
import onnxruntime as ort

model_path = "models/silero_vad.onnx"
vad = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

def test_vad(audio, name):
    state = np.zeros((2, 1, 128), dtype=np.float32)
    chunk_size = 512
    probs = []
    for i in range(0, len(audio) - chunk_size, chunk_size):
        chunk = audio[i:i+chunk_size]
        # Remove DC offset!
        chunk = chunk - np.mean(chunk)
        chunk = chunk.reshape(1, -1)
        
        out_prob, out_state = vad.run(None, {
            "input": chunk,
            "state": state,
            "sr": np.array(16000, dtype=np.int64),
        })
        state = out_state
        probs.append(out_prob[0][0])
    print(f"{name} -> Max: {np.max(probs):.6f}, Mean: {np.mean(probs):.6f}")

with wave.open("test_transcription_sync.wav", "rb") as wf:
    data = wf.readframes(wf.getnframes())
real_audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

dc_audio = np.clip(real_audio + 0.5, -1.0, 1.0)
test_vad(dc_audio, "Voice with +0.5 DC Offset (Fixed by mean subtraction)")
