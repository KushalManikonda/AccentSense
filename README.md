# AccentSense: Indian Accent Recognition & Cuisine Recommendation System

A real-time speech-based system that identifies the **native language accent** of Indian English speakers and recommends region-specific cuisines.  
Powered by **HuBERT embeddings**, **BiLSTM deep learning**, and a **Streamlit interactive interface**.

---

## 🚀 Features

- 🎤 Live microphone-based accent detection  
- 📁 Upload audio files (.wav, .flac, .mp3)  
- 🧠 MFCC + Random Forest baseline model  
- 🤖 HuBERT + BiLSTM deep learning model  
- 🍽 Accent-aware personalized cuisine recommendations  

---

## 🧪 Dataset Overview

- Total Samples: **8,116**
- Accent Classes: **6**
- Categories: Hindi, Gujarati, Kannada, Malayalam, Tamil, Telugu  
- Speech Types:  
  - **Word-level** recordings  
  - **Sentence-level** recordings  
- Sampling Rate Standardized to: **16 kHz**

---

## 📈 Model Comparison

Both MFCC and HuBERT representations were tested across models.  
Final best performing model: **HuBERT + BiLSTM**

| Feature Type | Model                   | Accuracy (%) | Notes                                 |
|--------------|-------------------------|--------------|---------------------------------------|
| MFCC         | Random Forest           | 99.01        | Lightweight baseline                  |
| HuBERT       | BiLSTM                  | 99.50        | Best performance & generalization     |

---

## 🧪 Word-Level vs Sentence-Level Evaluation

| Category        | Accuracy (%) | Robustness Level | Interpretability |
|-----------------|--------------|------------------|------------------|
| Word-Level      | 99.45        | Medium           | Low              |
| Sentence-Level  | 99.77        | High             | High             |

> Insight: Accent patterns emerge more clearly in **continuous speech**, improving model confidence and interpretability.

---

## 🔧 Tech Stack

| Layer            | Technology Used                          |
|------------------|------------------------------------------|
| Language         | Python                                   |
| ML/DL            | TensorFlow, PyTorch, Scikit-Learn        |
| Audio Processing | Librosa, torchaudio                      |
| Deployment       | Streamlit, streamlit-webrtc              |
| Embeddings       | HuBERT (Facebook AI), Wav2Vec2 processor |

---

## 🏗 System Architecture

1. Audio Input (Upload / Microphone)  
2. Preprocessing (Resampling, normalization)  
3. Feature Extraction: MFCC or HuBERT  
4. Classification (Random Forest / BiLSTM)  
5. Accent Prediction  
6. Regional Cuisine Recommendation Engine  

---

## ▶️ How to Run

```bash
conda create -n accentid python=3.10
conda activate accentid
pip install -r requirements.txt
streamlit run app.py
```

---

## 📄 Full Project Report → `AccentSense_Final_Report.docx`

👉 [Click here to view the report](https://github.com/KushalManikonda/AccentSense/blob/main/AccentSense_Final_Report.docx)

---

## 💻 GitHub Link → `AccentSense`

👉 [Click here to view the Git Repository](https://github.com/KushalManikonda/AccentSense)

---
