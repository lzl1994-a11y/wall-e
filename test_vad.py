import numpy as np
import onnxruntime as ort

model_path = "models/silero_vad.onnx"
vad = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
vad_state = np.zeros((2, 1, 128), dtype=np.float32)

# Try with 480 samples (30ms at 16kHz)
audio = np.zeros(480, dtype=np.float32).reshape(1, -1)

try:
    out_prob, out_state = vad.run(None, {
        "input": audio,
        "state": vad_state,
        "sr": np.array(16000, dtype=np.int64),
    })
    print("Success with 480 samples! Prob:", out_prob)
except Exception as e:
    print("Error with 480 samples:", e)

# Try with 512 samples
audio_512 = np.zeros(512, dtype=np.float32).reshape(1, -1)
try:
    out_prob, out_state = vad.run(None, {
        "input": audio_512,
        "state": vad_state,
        "sr": np.array(16000, dtype=np.int64),
    })
    print("Success with 512 samples! Prob:", out_prob)
except Exception as e:
    print("Error with 512 samples:", e)
