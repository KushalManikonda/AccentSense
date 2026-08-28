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
from pymongo import MongoClient
from datetime import datetime
import uuid
import os
from bson import ObjectId
import bcrypt

# ---- FORCE DEFINE AT TOP ----
def save_audio_and_prediction(audio_np, label, confidence, user_id="user_id"):
    import os
    from datetime import datetime
    import soundfile as sf

    UPLOAD_DIR = "uploads"
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    timestamp = int(datetime.utcnow().timestamp())
    filename = f"{user_id}_{timestamp}.wav"
    filepath = os.path.join(UPLOAD_DIR, filename)

    sf.write(filepath, audio_np, 16000)

    print("📦 Writing to MongoDB...")
    history_collection.insert_one({
        "user_id": st.session_state.user["_id"],
        "accent": label,
        "audio_path": filepath,
        "confidence": float(confidence),
        "timestamp": datetime.utcnow()
    })

MONGO_URI = "mongodb+srv://kushalmanikonda_db_user:V3klf7MlSpOi1sUQ@accentsense.izufdym.mongodb.net/?appName=AccentSense"

client = MongoClient(MONGO_URI)
db = client["accentsense"]

cuisine_collection = db["cuisines"]
history_collection = db["accent_history"]
users_collection = db["users"]

if "user" not in st.session_state:
    st.session_state.user = None

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
    import tensorflow as tf
    import joblib

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

def get_cuisine_from_db(accent):
    data = cuisine_collection.find_one({"accent": accent.lower()})
    return data["categories"] if data else None

def show_recommendations(label):
    region = get_cuisine_from_db(label)
    if not region:
        st.warning("🍽 No cuisine data available for this accent yet.")
        return

    st.write("### 🍽 Personalized Cuisine Suggestions")
    
    for category, items in region.items():
        st.markdown(f"**{category.capitalize()}**")
        st.write("\n".join([f"- {item}" for item in items]))
        st.write("")

# ------------------------------
# AUTH HELPERS
# ------------------------------

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

def create_user(email, password, full_name):
    from datetime import datetime
    import uuid

    if users_collection.find_one({"email": email}):
        return False, "User already exists"

    user = {
        "username": email.split("@")[0] + "_" + str(uuid.uuid4())[:5],
        "email": email,
        "password": hash_password(password),
        "auth_provider": "local",
        "full_name": full_name,
        "is_verified": True,
        "roles": ["user"],
        "preferences": {
            "cuisine_types": []
        },
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }

    print("📦 USER OBJECT:", user)   # DEBUG

    result = users_collection.insert_one(user)

    print("✅ USER INSERTED:", result.inserted_id)  # DEBUG

    user["_id"] = result.inserted_id
    return True, user

def login_user(email, password):
    user = users_collection.find_one({"email": email})
    if not user:
        return False, "User not found"

    if not verify_password(password, user["password"]):
        return False, "Invalid password"

    return True, user

# ------------------------------
# AUTH UI
# ------------------------------
if "user" not in st.session_state or not st.session_state.user:

    st.markdown(
        """
        <style>
        .auth-card {
            max-width: 420px;
            margin: auto;
            padding: 30px;
            border-radius: 15px;
            background-color: #111;
            box-shadow: 0px 0px 20px rgba(0,0,0,0.5);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='auth-card'>", unsafe_allow_html=True)

    st.title("Welcome to AccentSense")
    st.caption("Analyze spoken audio to detect regional English accents and receive tailored cuisine recommendations in real time.")

    tab2, tab1 = st.tabs(["Signup", "Login"])

    # -------- LOGIN --------
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
                st.rerun()
            else:
                st.error(result)

    # -------- SIGNUP --------
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

st.sidebar.text(f"Logged in as: {st.session_state.user['email']}")

if st.sidebar.button("Logout"):
    keys_to_clear = [
        "user",
        "upload_prediction",
        "mic_prediction"
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    st.rerun()

# ------------------------------
# UI
# ------------------------------
st.title("🎙️ Indian English Accent Identifier")
st.caption("Upload audio or record live — Model: HuBERT + BiLSTM")

# ------------------------------
# Upload Option
# ------------------------------
st.subheader("Upload Audio")
uploaded = st.file_uploader("Upload .wav/.mp3", type=["wav", "mp3"])

CONF_THRESHOLD = 0.85

if uploaded:
    data = uploaded.read()
    st.audio(data)

    if st.button("Predict from uploaded audio"):
        with st.spinner("Extracting features..."):
            y = read_audio_bytes_to_np(data)
            emb = extract_hubert_embedding(y)
            label, conf = predict_accent(emb)

        # store once (no duplicate calls)
        st.session_state.upload_prediction = (label, conf)

# ---- DISPLAY RESULT ----
if "upload_prediction" in st.session_state and st.session_state.upload_prediction:
    label, conf = st.session_state.upload_prediction

    try:
        conf = float(conf)
    except:
        conf = 0.0  # fallback to avoid crash

    conf_percent = int(conf * 100)

    if conf < CONF_THRESHOLD:
        st.warning(
            "⚠️ Accent could not be confidently predicted.\n"
            "This may be a foreign accent or unsupported language."
        )
    else:
        st.success(f"Accent: {label.title()} (Confidence: {conf:.2f})")

        # show recommendations ONLY if valid
        show_recommendations(label)

    # progress bar (kept outside so always visible)
    st.write(f"### Confidence: **{conf_percent}%**")

    progress_text = st.empty()
    progress_bar = st.progress(0)

    for i in range(conf_percent + 1):
        progress_text.write(f"**{i}%**")
        progress_bar.progress(i)
        time.sleep(0.01)

# ------------------------------
# Live Microphone Option
# ------------------------------
import sounddevice as sd

st.subheader("Record Using Microphone")

DURATION = 6

if st.button("🎤 Record"):
    st.write("Recording... Speak now.")
    recording = sd.rec(int(DURATION * TARGET_SR), samplerate=TARGET_SR, channels=1, dtype='float32')
    sd.wait()
    audio = recording.flatten()

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