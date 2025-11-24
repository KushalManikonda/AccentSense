# app.py
import os
import io
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av
import soundfile as sf
import librosa
import joblib
import tensorflow as tf
import torch
from transformers import Wav2Vec2FeatureExtractor, HubertModel
import time

def np_audio_to_wav_bytes(y, sr=16000):
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format="WAV")
    return buffer.getvalue()


# Store last result
if "upload_prediction" not in st.session_state:
    st.session_state.upload_prediction = None
if "mic_prediction" not in st.session_state:
    st.session_state.mic_prediction = None

# ------------------------------
# Config
# ------------------------------
st.set_page_config(page_title="Indian English Accent ID", page_icon="🎙️", layout="centered")

MODEL_PATH = "accent_bilstm_model.h5"
LABEL_ENC_PATH = "label_encoder.pkl"
HUBERT_NAME = "facebook/hubert-base-ls960"
TARGET_SR = 16000

CUISINE_MAP = {
    "hindi": {
        "breakfast": ["Aloo Paratha", "Poha", "Bedmi Puri"],
        "lunch": ["Dal Makhani", "Chole Bhature", "Rajma Chawal"],
        "snacks": ["Samosa", "Aloo Tikki", "Kachori"],
        "dinner": ["Butter Chicken", "Paneer Butter Masala", "Tandoori Roti"],
        "dessert": ["Gulab Jamun", "Rasmalai", "Gajar ka Halwa"],
        "drink": ["Masala Chai", "Lassi", "Jaljeera"],
        "seasonal_fruit": ["Mango", "Lychee", "Jamun"]
    },

    "gujarati": {
        "breakfast": ["Thepla", "Poha", "Muthiya"],
        "lunch": ["Gujarati Thali", "Undhiyu", "Kadhi Khichdi"],
        "snacks": ["Fafda Jalebi", "Khaman Dhokla", "Sev Khamani"],
        "dinner": ["Bajra Rotla", "Ringan Nu Shaak"],
        "dessert": ["Basundi", "Shrikhand", "Sutarfeni"],
        "drink": ["Chaas (Buttermilk)", "Kairi Panna"],
        "seasonal_fruit": ["Chiku", "Custard Apple"]
    },

    "kannada": {
        "breakfast": ["Bisi Bele Bath", "Rava Idli", "Neer Dosa"],
        "lunch": ["Mysore Rasam", "Ragi Mudde with Sambar"],
        "snacks": ["Maddur Vada", "Masala Mandakki"],
        "dinner": ["Akki Roti", "Kotte Kadubu"],
        "dessert": ["Mysore Pak", "Kesari Bath"],
        "drink": ["Filter Coffee", "Majjige"],
        "seasonal_fruit": ["Jackfruit", "Banana"]
    },

    "malayalam": {
        "breakfast": ["Appam with Stew", "Puttu with Kadala Curry", "Idiyappam"],
        "lunch": ["Karimeen Fry", "Avial", "Fish Curry Rice"],
        "snacks": ["Banana Chips", "Unniyappam"],
        "dinner": ["Malabar Parotta with Beef Curry", "Thalassery Biryani"],
        "dessert": ["Payasam", "Ada Pradhaman"],
        "drink": ["Tender Coconut Water", "Sulaimani Chai"],
        "seasonal_fruit": ["Banana", "Jackfruit"]
    },

    "tamil": {
        "breakfast": ["Pongal", "Idli Sambar", "Medu Vada"],
        "lunch": ["South Indian Thali", "Sambar Rice", "Kootu"],
        "snacks": ["Murukku", "Sundal"],
        "dinner": ["Dosa with Chutney", "Kothu Parotta"],
        "dessert": ["Kesari", "Payasam"],
        "drink": ["Filter Coffee", "Panagam"],
        "seasonal_fruit": ["Banana", "Mango"]
    },

    "telugu": {
        "breakfast": ["Pesarattu", "Upma", "Poori with Aloo Kurma"],
        "lunch": ["Gongura Pachadi", "Pulihora", "Andhra Veg Curry"],
        "snacks": ["Punugulu", "Garelu (Medu Vada)"],
        "dinner": ["Hyderabadi Biryani", "Ragi Sangati with Kodi Pulusu"],
        "dessert": ["Pootharekulu", "Double ka Meetha"],
        "drink": ["Irani Chai", "Buttermilk with Curry Leaves"],
        "seasonal_fruit": ["Mango (Banganapalli)", "Custard Apple"]
    }
}

# ------------------------------
# Load Models (cached)
# ------------------------------
@st.cache_resource(show_spinner=True)
def load_hubert():
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_NAME)
    hubert = HubertModel.from_pretrained(HUBERT_NAME)
    hubert.eval()
    return extractor, hubert

@st.cache_resource(show_spinner=True)
def load_accent_model():
    # Patch LSTM to ignore unsupported config keys
    from keras.layers import LSTM
    def lstm_patched(**kwargs):
        kwargs.pop("time_major", None)
        return LSTM(**kwargs)

    tf.keras.utils.get_custom_objects().update({"LSTM": lstm_patched})

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.build(input_shape=(None, 768))
    le = joblib.load(LABEL_ENC_PATH)
    return model, le

feature_extractor, hubert = load_hubert()
tf_model, label_enc = load_accent_model()
device = torch.device("cpu")
hubert.to(device)

# ------------------------------
# Audio Helpers
# ------------------------------
def read_audio_bytes_to_np(data_bytes):
    y, sr = sf.read(io.BytesIO(data_bytes), dtype="float32")
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    return y.astype(np.float32)

def extract_hubert_embedding(y):
    with torch.no_grad():
        inp = feature_extractor(y, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        x = hubert(inp.input_values.to(device)).last_hidden_state.mean(dim=1).cpu().numpy()
    return x

def predict_accent(emb):
    probs = tf_model.predict(emb, verbose=0)[0]
    idx = np.argmax(probs)
    label = label_enc.inverse_transform([idx])[0]
    return label, float(probs[idx])

def show_recommendations(label):
    region = CUISINE_MAP.get(label.lower())

    if not region:
        st.warning("🍽 No cuisine data available for this accent yet.")
        return

    st.write("### 🍽 Personalized Cuisine Suggestions")
    
    for category, items in region.items():
        st.markdown(f"**{category.capitalize()}**")
        st.write("\n".join([f"- {item}" for item in items]))
        st.write("")  # spacing


# ------------------------------
# UI
# ------------------------------
st.title("🎙️ Indian English Accent Identifier")
st.caption("Upload audio or record live — Model: HuBERT + BiLSTM")

# ------------------------------
# Upload Option
# ------------------------------
st.subheader("Upload Audio")
uploaded = st.file_uploader("Upload .wav/.mp3/.flac", type=["wav", "mp3", "flac"])

if uploaded:
    data = uploaded.read()
    st.audio(data)
    if st.button("Predict from uploaded audio"):
        with st.spinner("Extracting features..."):
            y = read_audio_bytes_to_np(data)
            emb = extract_hubert_embedding(y)
            label, conf = predict_accent(emb)
        st.session_state.upload_prediction = (label, conf)

if st.session_state.upload_prediction:
    label, conf = st.session_state.upload_prediction
    st.success(f"**Accent:** {label.title()}  (Confidence: {conf:.3f})")
    label, conf = predict_accent(emb)
    if conf < 0.55:
        st.warning("Low confidence prediction. Please speak a longer sentence (3–6 seconds).")
    else:
        st.success(f"Accent: {label.title()} ({conf:.2f})")

    show_recommendations(label)

    conf_percent = int(conf * 100)

    st.write(f"### Confidence: **{conf_percent}%**")

    progress_text = st.empty()
    progress_bar = st.progress(0)

    for i in range(conf_percent + 1):
        progress_text.write(f"**{i}%**")
        progress_bar.progress(i)
        time.sleep(0.01)  # controls animation speed

# ------------------------------
# Live Microphone Option (Reliable Mode)
# ------------------------------
import sounddevice as sd

st.subheader("Record Using Microphone")

DURATION = 6  # seconds

if st.button("🎤 Record"):
    st.write("Recording... Speak now.")
    recording = sd.rec(int(DURATION * TARGET_SR), samplerate=TARGET_SR, channels=1, dtype='float32')
    sd.wait()
    audio = recording.flatten()

    # Playback (Now Audible ✅)
    wav_bytes = np_audio_to_wav_bytes(audio, TARGET_SR)
    st.audio(wav_bytes, format="audio/wav")

    with st.spinner("Extracting & Predicting..."):
        emb = extract_hubert_embedding(audio)
        label, conf = predict_accent(emb)

    st.session_state.mic_prediction = (label, conf)

if st.session_state.mic_prediction:
    label, conf = st.session_state.mic_prediction
    st.success(f"**Accent:** {label.title()} (Confidence: {conf:.3f})")
    show_recommendations(label)

st.caption("Tip: Speak a full sentence for best results.")
