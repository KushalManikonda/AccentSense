import io
import numpy as np
import soundfile as sf
import librosa

TARGET_SR = 16000

def read_audio_bytes_to_np(data_bytes):
    y, sr = sf.read(io.BytesIO(data_bytes), dtype="float32")
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    return y.astype(np.float32)

def np_audio_to_wav_bytes(y, sr=16000):
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format="WAV")
    return buffer.getvalue()