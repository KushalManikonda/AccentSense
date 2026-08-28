import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import time

try:
    import sounddevice as sd
    HAS_SD = True
except OSError:
    HAS_SD = False

# ---- IMPORTS FROM YOUR MODULES ----
from auth.auth_service import create_user, login_user
from services.audio_service import read_audio_bytes_to_np, np_audio_to_wav_bytes
from services.recommendation import get_cuisine
from ml.model import load_models, extract_embedding, predict
from utils.storage import save_audio_and_prediction

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Indian English Accent ID", page_icon="🎙️", layout="centered")

# ------------------------------
# LOAD MODELS (ONCE)
# ------------------------------
@st.cache_resource
def init_models():
    return load_models()

extractor, hubert, tf_model, label_enc = init_models()

# ------------------------------
# SESSION STATE INIT
# ------------------------------
if "user" not in st.session_state:
    st.session_state.user = None

if "upload_prediction" not in st.session_state:
    st.session_state.upload_prediction = None

if "mic_prediction" not in st.session_state:
    st.session_state.mic_prediction = None

# ------------------------------
# AUTH UI
# ------------------------------
if not st.session_state.user:

    st.markdown("""
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
    """, unsafe_allow_html=True)

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

# ------------------------------
# SIDEBAR
# ------------------------------
st.sidebar.text(f"Logged in as: {st.session_state.user['email']}")

if st.sidebar.button("Logout"):
    for key in ["user", "upload_prediction", "mic_prediction"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# ------------------------------
# MAIN UI
# ------------------------------
st.title("🎙️ Indian English Accent Identifier")
st.caption("Upload audio or record live — Model: HuBERT + BiLSTM")

CONF_THRESHOLD = 0.70

# ------------------------------
# CUISINE DISPLAY
# ------------------------------
def show_recommendations(label):
    region = get_cuisine(label)

    if not region:
        st.warning("🍽 No cuisine data available.")
        return

    st.write("### 🍽 Personalized Cuisine Suggestions")

    for category, items in region.items():
        st.markdown(f"**{category.capitalize()}**")
        for item in items:
            st.write(f"- {item}")
        st.write("")

# ------------------------------
# FILE UPLOAD
# ------------------------------
st.subheader("Upload Audio")
uploaded = st.file_uploader("Upload .wav/.mp3", type=["wav", "mp3"])

if uploaded:
    data = uploaded.read()
    st.audio(data)

    if st.button("Predict from uploaded audio"):
        with st.spinner("Processing..."):
            y = read_audio_bytes_to_np(data)
            emb = extract_embedding(y, extractor, hubert)
            label, conf = predict(emb, tf_model, label_enc)

        st.session_state.upload_prediction = (label, conf)

# ------------------------------
# DISPLAY RESULT
# ------------------------------
if st.session_state.upload_prediction:
    label, conf = st.session_state.upload_prediction

    conf = float(conf)
    conf_percent = int(conf * 100)

    if conf < CONF_THRESHOLD:
        st.warning("⚠️ Accent could not be confidently predicted.\nThis may be a foreign accent.")
    else:
        st.success(f"Accent: {label.title()} (Confidence: {conf:.2f})")
        show_recommendations(label)

        # SAVE ONLY VALID PREDICTIONS
        save_audio_and_prediction(
            audio_np=read_audio_bytes_to_np(uploaded.getvalue()),
            label=label,
            confidence=conf,
            user_id=st.session_state.user["_id"]
        )

    # progress bar
    st.write(f"### Confidence: **{conf_percent}%**")

    progress_bar = st.progress(0)
    for i in range(conf_percent + 1):
        progress_bar.progress(i)
        time.sleep(0.01)

# ------------------------------
# MICROPHONE
# ------------------------------
st.subheader("Record Using Microphone")

DURATION = 10
TARGET_SR = 16000

if not HAS_SD:
    st.warning("Microphone recording is not available in this cloud environment. Please use the file upload option above.")
else:
    if st.button("🎤 Record"):
        st.write("Recording... Speak now.")
        recording = sd.rec(int(DURATION * TARGET_SR), samplerate=TARGET_SR, channels=1, dtype='float32')
        sd.wait()

        audio = recording.flatten()
        st.session_state.mic_audio = audio
        wav_bytes = np_audio_to_wav_bytes(audio)

        st.audio(wav_bytes)

        with st.spinner("Processing..."):
            emb = extract_embedding(audio, extractor, hubert)
            label, conf = predict(emb, tf_model, label_enc)

        st.session_state.mic_prediction = (label, conf)

# ------------------------------
# MIC RESULT
# ------------------------------
if st.session_state.mic_prediction:
    label, conf = st.session_state.mic_prediction

    if conf >= CONF_THRESHOLD:
        st.success(f"Accent: {label.title()} ({conf:.2f})")
        show_recommendations(label)

        if "mic_audio" in st.session_state:
            save_audio_and_prediction(
                audio_np=st.session_state.mic_audio,
                label=label,
                confidence=conf,
                user_id=st.session_state.user["_id"]
            )
    else:
        st.warning("Low confidence prediction.")

st.caption("Tip: Speak a full sentence for best results.")