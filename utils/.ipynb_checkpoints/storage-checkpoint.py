import os
import soundfile as sf
from datetime import datetime
from db.mongo import history_collection

def save_audio_and_prediction(audio_np, label, confidence, user_id):
    UPLOAD_DIR = "uploads"
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    timestamp = int(datetime.utcnow().timestamp())
    filename = f"{user_id}_{timestamp}.wav"
    filepath = os.path.join(UPLOAD_DIR, filename)

    sf.write(filepath, audio_np, 16000)

    history_collection.insert_one({
        "user_id": user_id,
        "accent": label,
        "audio_path": filepath,
        "confidence": float(confidence),
        "timestamp": datetime.utcnow()
    })