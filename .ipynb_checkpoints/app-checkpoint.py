import io
import os
import time
import numpy as np
import streamlit as st
import sounddevice as sd
import soundfile as sf
import librosa
import joblib
import tensorflow as tf
import torch
from transformers import Wav2Vec2FeatureExtractor, HubertModel

# Local modules
from auth.auth_service import create_user, login_user
from utils.storage import save_audio_and_prediction

# Config
MODEL_PATH = os.path.join("models", "accent_bilstm_model.h5")
LABEL_ENC_PATH = os.path.join("models", "label_encoder.pkl")
HUBERT_NAME = "facebook/hubert-base-ls960"
TARGET_SR = 16000
DURATION = 6
CONF_THRESHOLD = 0.55

# Cuisine map (kept concise)
CUISINE_MAP = {
    "hindi": {"breakfast": ["Aloo Paratha", "Poha"], "lunch": ["Dal Makhani"], "snacks": ["Samosa"], "dinner": ["Butter Chicken"]},
    "gujarati": {"breakfast": ["Thepla"], "lunch": ["Gujarati Thali"]},
    "kannada": {"breakfast": ["Bisi Bele Bath"], "lunch": ["Mysore Rasam"]},
    "malayalam": {"breakfast": ["Appam with Stew"]},
    "tamil": {"breakfast": ["Pongal"]},
    "telugu": {"breakfast": ["Pesarattu"]}
}

# Helpers
def np_audio_to_wav_bytes(y, sr=TARGET_SR):
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format="WAV")
    return buffer.getvalue()

@st.cache_resource(show_spinner=True)
def load_hubert():
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_NAME)
    hubert = HubertModel.from_pretrained(HUBERT_NAME)
    hubert.eval()
    return extractor, hubert

@st.cache_resource(show_spinner=True)
def load_accent_model():
    try:
        from keras.layers import LSTM
        def lstm_patched(**kwargs):
            kwargs.pop("time_major", None)
            return LSTM(**kwargs)
        tf.keras.utils.get_custom_objects().update({"LSTM": lstm_patched})
    except Exception:
        pass
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.build(input_shape=(None, 768))
    le = joblib.load(LABEL_ENC_PATH)
    return model, le

# Initialize
feature_extractor, hubert = load_hubert()
tf_model, label_enc = load_accent_model()

device = torch.device("cpu")
hubert.to(device)

# Audio helpers
import soundfile as sf

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
        for item in items:
            st.write(f"- {item}")
        st.write("")

# Page + styles
st.set_page_config(page_title="AccentSense", page_icon="🎙️", layout="wide")

st.markdown("""
<style>
.auth-card {max-width:700px; margin:20px auto; padding:24px; border-radius:12px; background:linear-gradient(135deg,#0f172a,#111827); color:#fff}
.card {background:#ffffff; color:#111827; padding:18px; border-radius:10px; box-shadow:0 6px 18px rgba(0,0,0,0.08)}
.result-card {background:linear-gradient(180deg,#fff,#f7fafc); padding:18px; border-radius:10px}
.center {display:flex; justify-content:center}
.small {font-size:0.9rem}
</style>
""", unsafe_allow_html=True)

# Session state
if "user" not in st.session_state:
    st.session_state.user = None
if "upload_prediction" not in st.session_state:
    st.session_state.upload_prediction = None
if "mic_prediction" not in st.session_state:
    st.session_state.mic_prediction = None

# AUTH CARD
if not st.session_state.user:
    st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
    st.title("AccentSense")
    st.caption("Detect regional Indian English accents and get cuisine suggestions.")
    tabs = st.tabs(["Signup","Login"])
    with tabs[1]:
        st.subheader("Login")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", use_container_width=True):
            success, result = login_user(email, password)
            if success:
                st.session_state.user = result
                st.session_state.upload_prediction = None
                st.session_state.mic_prediction = None
                st.experimental_rerun()
            else:
                st.error(result)
    with tabs[0]:
        st.subheader("Create account")
        full_name = st.text_input("Full Name", key="signup_name")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")
        if st.button("Create Account", use_container_width=True):
            success, result = create_user(email, password, full_name)
            if success:
                st.success("Account created. Please login.")
            else:
                st.error(result)
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# Sidebar
with st.sidebar:
    st.write(f"Logged in as: {st.session_state.user.get('email')}")
    if st.button("Logout"):
        for k in ["user","upload_prediction","mic_prediction"]:
            if k in st.session_state:
                del st.session_state[k]
        st.experimental_rerun()
    st.markdown("---")
    st.write("Tips")
    st.write("• Speak a full sentence (3–6s)\n• Use quiet background")

# Layout: two columns (controls | results)
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Analyze Audio")
    st.subheader("Upload")
    uploaded = st.file_uploader("Upload .wav/.mp3/.flac", type=["wav","mp3","flac"])
    st.write("")
    st.subheader("Record")
    st.write("Duration: 3–6 seconds")
    if st.button("🎤 Record"):
        st.write("Recording... Speak now.")
        recording = sd.rec(int(DURATION * TARGET_SR), samplerate=TARGET_SR, channels=1, dtype='float32')
        sd.wait()
        audio = recording.flatten()
        st.audio(np_audio_to_wav_bytes(audio), format="audio/wav")
        with st.spinner("Processing..."):
            emb = extract_hubert_embedding(audio)
            label, conf = predict_accent(emb)
        st.session_state.mic_prediction = (label, conf)

    if uploaded:
        data = uploaded.read()
        st.audio(data)
        if st.button("Predict from uploaded audio"):
            with st.spinner("Processing..."):
                y = read_audio_bytes_to_np(data)
                emb = extract_hubert_embedding(y)
                label, conf = predict_accent(emb)
            st.session_state.upload_prediction = (label, conf)

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # Result card
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.subheader("Result")
    if st.session_state.upload_prediction or st.session_state.mic_prediction:
        label, conf = st.session_state.upload_prediction or st.session_state.mic_prediction
        conf_percent = int(float(conf) * 100)
        st.metric(label="Predicted Accent", value=label.title())
        st.metric(label="Confidence", value=f"{conf_percent}%")
        st.progress(conf_percent)
        if float(conf) >= CONF_THRESHOLD:
            st.success(f"Accent: {label.title()} ({conf:.2f})")
            show_recommendations(label)
            try:
                # attempt save
                if st.session_state.upload_prediction:
                    save_audio_and_prediction(audio_np=read_audio_bytes_to_np(uploaded.getvalue()), label=label, confidence=conf, user_id=st.session_state.user.get("_id"))
                else:
                    save_audio_and_prediction(audio_np=audio, label=label, confidence=conf, user_id=st.session_state.user.get("_id"))
            except Exception:
                pass
        else:
            st.warning("Low confidence. Try a longer sentence or quieter environment.")
    else:
        st.info("No prediction yet. Upload or record audio to get started.")
    st.markdown("</div>", unsafe_allow_html=True)

st.caption("Built with HuBERT + BiLSTM — AccentSense")
