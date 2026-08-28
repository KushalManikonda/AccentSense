# AccentSense: Native Language Identification & Accent-Aware Recommendation

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![MongoDB](https://img.shields.io/badge/MongoDB-4EA94B?style=for-the-badge&logo=mongodb&logoColor=white)

**AccentSense** is a real-time speech-based intelligent system designed to predict the native language of Indian English speakers by analyzing their distinct regional accents. It addresses the rich accent diversity in India (e.g., Telugu-English, Tamil-English, Gujarati-English) and goes a step further by providing accent-aware personalization through culturally relevant cuisine and travel recommendations.

---

## ⚙️ System Workflow
1. **Audio Acquisition:** Users provide audio via .wav/.mp3 upload or live microphone recording.
2. **Audio Preprocessing:** The system normalizes amplitude, trims silence, converts to mono, and resamples the audio to 16 kHz.
3. **Feature Extraction:** Extracts deep contextual embeddings using a pre-trained **HuBERT** model.
4. **Accent Classification:** A **BiLSTM** network predicts one of 6 regional Indian accents (Hindi, Tamil, Telugu, Malayalam, Kannada, Gujarati).
5. **Personalization Engine:** Maps the detected accent to a specific Indian region and outputs tailored cultural and cuisine recommendations.

---

## 🛠️ Technology Stack & Rationale

| Layer | Technology | Why it was chosen |
| :--- | :--- | :--- |
| **Language** | Python 3.9+ | Industry standard for ML/AI and rapid prototyping. |
| **Deep Learning** | PyTorch, Hugging Face | **HuBERT** effectively captures high-level phonetic and prosodic accent cues. **BiLSTM** was chosen over static models to capture forward/backward temporal articulation and pronunciation flow. |
| **Audio Processing** | Librosa, Soundfile | Standard, highly optimized libraries for audio resampling, trimming, and MFCC extraction. |
| **Frontend UI** | Streamlit | Lightweight framework providing a real-time, user-friendly interface without needing a complex separate frontend. |
| **Database** | MongoDB | Highly flexible NoSQL document storage for lazy-loaded user authentication and interaction logging. |

---

## 📊 Machine Learning: Comparisons & Evaluation

### Models Compared
The project evaluated traditional handcrafted acoustic features against modern self-supervised transformer representations:
1. **Baseline Model:** Random Forest Classifier trained on low-level **MFCC** (Mel-Frequency Cepstral Coefficients) features.
2. **Primary Model:** **BiLSTM** sequence model trained on 768-dimensional contextual **HuBERT** embeddings.

### Evaluation Metrics Used
The models were evaluated using the following metrics:
* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **Confusion Matrix** (to analyze misclassifications among acoustically similar languages like Tamil-Malayalam).

### Key Experimental Findings
* **Accuracy Comparison:** The HuBERT+BiLSTM architecture achieved **99.50%** accuracy, outperforming the MFCC+Random Forest baseline (**99.01%**).
* **Word-level vs. Sentence-level:** Sentence-level speech yielded higher stability (**99.63%**) compared to isolated word inputs (**99.31%**) due to richer rhythmic and intonation cues.
* **Transformer Layer Analysis:** Evaluating individual transformer layers revealed that **HuBERT Layers 9–11** preserve the strongest accent-sensitive information, outperforming the final layers.

---

## 🧪 Software Testing

To ensure reliability, robustness, and exception safety, a multi-layered automated and manual testing strategy was employed. The following types of testing were performed:

* **Unit Testing** (Isolated validation of audio preprocessing, prediction, and DB utilities)
* **Integration Testing** (End-to-end pipeline verification from UI to MongoDB)
* **Functional Testing** (Upload functionality, UI responsiveness, prediction rendering)
* **System Testing**
* **Exception Handling Testing** (Invalid file formats, missing audio)
* **Edge-Case Testing** (Silent audio, exceptionally long/short audio inputs)
* **Browser Compatibility Testing**
* **Machine Learning Evaluation Testing**
* **User Acceptance Testing (UAT)**

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/KushalManikonda/AccentSense.git
cd AccentSense

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

> **Note:** Requires `tf-nightly` if running on Python 3.14+.

### 3. Setup Environment Variables

Create a `.env` file in the root directory and add your MongoDB connection string:

```env
MONGO_URI=mongodb+srv://<username>:<password>@<cluster-url>/?appName=AccentSense

```

### 4. Run the Application

```bash
streamlit run app.py

```

Access the web app at [http://localhost:8501](http://localhost:8501).

---

## 📂 Project Structure

```text
AccentSense/
├── app.py                  # Main Streamlit application entry point
├── auth/                   # User authentication logic (bcrypt)
├── db/                     # MongoDB lazy-connection configuration
├── ml/                     # ML pipeline: HuBERT extraction & BiLSTM prediction
├── models/                 # Saved weights and label encoders
├── services/               # Audio preprocessing & recommendation mapping
├── tests/                  # Comprehensive Pytest suite
├── utils/                  # Storage and helper functions
├── requirements.txt        # Project dependencies
└── .gitignore              # Git ignore rules
```

---

## 🔮 Future Enhancements
* Expansion of datasets across different age groups and dialect variations.
* Integration of **HuBERT-Large** and **Whisper-based** architectures.
* Cloud-hosted REST API deployment with secure authentication.
* Docker-based scalable microservices infrastructure.
* Multilingual user interface support.
* Dynamic, AI-driven recommendation personalization based on user preferences.
* Foreign accent detection capabilities.
