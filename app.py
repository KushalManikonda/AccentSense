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

# Cuisine map (partial - keep as in remote)
CUISINE_MAP = {
    "hindi": {"breakfast": ["Aloo Paratha", "Poha", "Bedmi Puri"], "lunch": ["Dal Makhani", "Chole Bhature", "Rajma Chawal"], "snacks": ["Samosa"], "dinner": ["Butter Chicken"], "dessert": ["Gulab Jamun"], "drink": ["Masala Chai"], "seasonal_fruit": ["Mango"]},
    "gujarati": {"breakfast": ["Thepla"], "lunch": ["Gujarati Thali"], "snacks": ["Fafda Jalebi"], "dinner": ["Bajra Rotla"], "dessert": ["Basundi"], "drink": ["Chaas (Buttermilk)"], "seasonal_fruit": ["Chiku"]},
    "kannada": {"breakfast": ["Bisi Bele Bath"], "lunch": ["Mysore Rasam"], "snacks": ["Maddur Vada"], "dinner": ["Akki Roti"], "dessert": ["Mysore Pak"], "drink": ["Filter Coffee"], "seasonal_fruit": ["Jackfruit"]},
    "malayalam": {"breakfast": ["Appam with Stew"], "lunch": ["Karimeen Fry"], "snacks": ["Banana Chips"], "dinner": ["Malabar Parotta with Beef Curry"], "dessert": ["Payasam"], "drink": ["Tender Coconut Water"], "seasonal_fruit": ["Banana"]},
    "tamil": {"breakfast": ["Pongal"], "lunch": ["South Indian Thali"], "snacks": ["Murukku"], "dinner": ["Dosa with Chutney"], "dessert": ["Kesari"], "drink": ["Filter Coffee"], "seasonal_fruit": ["Banana"]},
    "telugu": {"breakfast": ["Pesarattu"], "lunch": ["Gongura Pachadi"], "snacks": ["Punugulu"], "dinner": ["Hyderabadi Biryani"], "dessert": ["Pootharekulu"], "drink": ["Irani Chai"], "seasonal_fruit": ["Mango (Banganapalli)"]}
}

# Utils
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
    # Patch LSTM to ignore unsupported config keys if needed
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

# Initialize models
feature_extractor, hubert = load_hubert()
tf_model, label_enc = load_accent_model()

device = torch.device("cpu")
hubert.to(device)

# Audio helpers
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

# Session state
if "user" not in st.session_state:
    st.session_state.user = None
if "upload_prediction" not in st.session_state:
    st.session_state.upload_prediction = None
if "mic_prediction" not in st.session_state:
    st.session_state.mic_prediction = None

# Page config
st.set_page_config(page_title="Indian English Accent ID", page_icon="🎙️", layout="centered")

# AUTH UI
if not st.session_state.user:
    st.markdown("""
        <style>
        .auth-card { max-width: 420px; margin: auto; padding: 30px; border-radius: 15px; background-color: #111; color: #fff; }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
    st.title("Welcome to AccentSense")
    st.caption("Analyze spoken audio to detect regional English accents and receive tailored cuisine recommendations in real time.")
    tab2, tab1 = st.tabs(["Signup", "Login"]) 
    with tab1:
        st.subheader("Welcome back")
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
    with tab2:
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
st.sidebar.text(f"Logged in as: {st.session_state.user.get('email')}")
if st.sidebar.button("Logout"):
    for key in ["user", "upload_prediction", "mic_prediction"]:
        if key in st.session_state:
            del st.session_state[key]
    st.experimental_rerun()

# Main UI
st.title("🎙️ Indian English Accent Identifier")
st.caption("Upload audio or record live — Model: HuBERT + BiLSTM")

# Upload
st.subheader("Upload Audio")
uploaded = st.file_uploader("Upload .wav/.mp3/.flac", type=["wav", "mp3", "flac"])

if uploaded:
    data = uploaded.read()
    st.audio(data)
    if st.button("Predict from uploaded audio"):
        with st.spinner("Processing..."):
            y = read_audio_bytes_to_np(data)
            emb = extract_hubert_embedding(y)
            # model expects batch
            label, conf = predict_accent(emb)
        st.session_state.upload_prediction = (label, conf)

if st.session_state.upload_prediction:
    label, conf = st.session_state.upload_prediction
    conf = float(conf)
    conf_percent = int(conf * 100)
    if conf < CONF_THRESHOLD:
        st.warning("⚠️ Accent could not be confidently predicted. This may be a foreign accent.")
    else:
        st.success(f"Accent: {label.title()} (Confidence: {conf:.2f})")
        show_recommendations(label)
        # Save prediction if user exists
        try:
            save_audio_and_prediction(audio_np=read_audio_bytes_to_np(uploaded.getvalue()), label=label, confidence=conf, user_id=st.session_state.user.get("_id"))
        except Exception:
            pass
    st.write(f"### Confidence: **{conf_percent}%**")
    progress_bar = st.progress(0)
    for i in range(conf_percent + 1):
        progress_bar.progress(i)
        time.sleep(0.01)

# Microphone
st.subheader("Record Using Microphone")
if st.button("🎤 Record"):
    st.write("Recording... Speak now.")
    recording = sd.rec(int(DURATION * TARGET_SR), samplerate=TARGET_SR, channels=1, dtype='float32')
    sd.wait()
    audio = recording.flatten()
    wav_bytes = np_audio_to_wav_bytes(audio)
    st.audio(wav_bytes, format="audio/wav")
    with st.spinner("Processing..."):
        emb = extract_hubert_embedding(audio)
        label, conf = predict_accent(emb)
    st.session_state.mic_prediction = (label, conf)

if st.session_state.mic_prediction:
    label, conf = st.session_state.mic_prediction
    if conf >= CONF_THRESHOLD:
        st.success(f"Accent: {label.title()} ({conf:.2f})")
        show_recommendations(label)
        try:
            save_audio_and_prediction(audio_np=audio, label=label, confidence=conf, user_id=st.session_state.user.get("_id"))
        except Exception:
            pass
    else:
        st.warning("Low confidence prediction. Please speak a longer sentence (3–6 seconds).")

st.caption("Tip: Speak a full sentence for best results.")
