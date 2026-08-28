import torch
import numpy as np
import joblib
import tensorflow as tf
from transformers import Wav2Vec2FeatureExtractor, HubertModel

import os

BASE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models"
)

MODEL_PATH = os.path.join(BASE_DIR, "accent_bilstm_model.h5")
LABEL_ENC_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

HUBERT_NAME = "facebook/hubert-base-ls960"

device = torch.device("cpu")

def load_models():
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_NAME)
    hubert = HubertModel.from_pretrained(HUBERT_NAME)
    hubert.eval()
    hubert.to(device)

    tf.keras.utils.disable_interactive_logging()

    from keras.layers import LSTM
    original_init = LSTM.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.pop("time_major", None)
        original_init(self, *args, **kwargs)

    LSTM.__init__ = patched_init

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.build(input_shape=(None, 768))

    le = joblib.load(LABEL_ENC_PATH)

    return extractor, hubert, model, le

def extract_embedding(y, extractor, hubert):
    with torch.no_grad():
        inp = extractor(y, sampling_rate=16000, return_tensors="pt", padding=True)
        emb = hubert(inp.input_values.to(device)).last_hidden_state.mean(dim=1).cpu().numpy()
    return emb

def predict(emb, model, label_enc):
    probs = model.predict(emb, verbose=0)[0]
    idx = np.argmax(probs)
    label = label_enc.inverse_transform([idx])[0]
    return label, float(probs[idx])