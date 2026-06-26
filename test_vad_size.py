import numpy as np
import onnxruntime as ort

model_path = "models/silero_vad.onnx"
vad = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

def get_prob(samples):
    state = np.zeros((2, 1, 128), dtype=np.float32)
    # Generate 3 seconds of a 400Hz sine wave (human voice fundamental freq)
    t = np.arange(16000 * 3) / 16000
    audio_full = 0.5 * np.sin(2 * np.pi * 400 * t).astype(np.float32)
    
    probs = []
    for i in range(0, len(audio_full) - samples, samples):
        chunk = audio_full[i:i+samples].reshape(1, -1)
        out_prob, out_state = vad.run(None, {
            "input": chunk,
            "state": state,
            "sr": np.array(16000, dtype=np.int64),
        })
        state = out_state
        probs.append(out_prob[0][0])
    return np.max(probs), np.mean(probs)

max480, mean480 = get_prob(480)
max512, mean512 = get_prob(512)

print(f"480 samples -> Max: {max480:.6f}, Mean: {mean480:.6f}")
print(f"512 samples -> Max: {max512:.6f}, Mean: {mean512:.6f}")
