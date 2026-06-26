import wave
import numpy as np
import onnxruntime as ort

model_path = "models/silero_vad.onnx"
vad = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

def run_vad_on_file(wav_path, chunk_size):
    state = np.zeros((2, 1, 128), dtype=np.float32)
    
    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        data = wf.readframes(n_frames)
        
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    
    probs = []
    for i in range(0, len(audio) - chunk_size, chunk_size):
        chunk = audio[i:i+chunk_size].reshape(1, -1)
        out_prob, out_state = vad.run(None, {
            "input": chunk,
            "state": state,
            "sr": np.array(16000, dtype=np.int64),
        })
        state = out_state
        probs.append(out_prob[0][0])
        
    return probs

probs_480 = run_vad_on_file("test_transcription_sync.wav", 480)
probs_512 = run_vad_on_file("test_transcription_sync.wav", 512)

print("480 chunk max prob:", np.max(probs_480))
print("480 chunk mean prob:", np.mean(probs_480))
print("512 chunk max prob:", np.max(probs_512))
print("512 chunk mean prob:", np.mean(probs_512))
