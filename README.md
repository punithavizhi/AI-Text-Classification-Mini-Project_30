# 🤖 AI Text Classification: Small Trained Classifier vs Chatbot

> **Academic Mini-Project 30** | SMS Spam Detection Dashboard  
> Comparing TF-IDF + Logistic Regression against AI Chatbot (LLM) on UCI SMS Spam Collection

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

---

## 📋 Project Overview

This project is a **fully functional AI-powered web application** that classifies SMS text messages as **SPAM** or **HAM (Not Spam)** using a small trained machine-learning model and compares its performance against an AI Chatbot (Large Language Model).

### 🔬 Core Research Question
> *"Can a lightweight, classical ML classifier (TF-IDF + Logistic Regression) achieve competitive performance with near-zero cost and sub-millisecond latency when compared to a general-purpose AI chatbot?"*

---

## 🌟 Features

| Page | Description |
|------|-------------|
| 🏠 **Home Dashboard** | Dataset overview, model accuracy stats, quick navigation |
| 🔍 **Text Classification** | Real-time SMS spam prediction with confidence and explainability |
| 📊 **Model Performance** | Confusion matrix, precision, recall, F1-score, interactive charts |
| ⚖️ **Chatbot Comparison** | Side-by-side accuracy comparison with CSV upload support |
| 📂 **Dataset Explorer** | Browse dataset, vocabulary analysis, class distribution |
| ℹ️ **About & Viva Guide** | Project documentation, algorithms, and college demo Q&A |

---

## 🛠️ Technology Stack

```
Frontend:   Streamlit + Custom CSS (Dark Navy & Purple Theme)
Backend:    Python 3.11
ML:         Scikit-learn (TF-IDF + Logistic Regression)
Data:       Pandas, NumPy
Charts:     Plotly Express & Plotly Graph Objects
Dataset:    UCI SMS Spam Collection (5,574 messages)
Deployment: Streamlit Community Cloud + GitHub
```

---

## 📂 Project Structure

```
AI-Text-Classification/
│
├── app.py                          # Main Streamlit entrypoint (Home Dashboard)
├── train_model.py                  # ML training pipeline (run once)
├── ml_utils.py                     # Shared utilities (model loading, CSS, explainability)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
│
├── .streamlit/
│   └── config.toml                 # Streamlit theme and server configuration
│
├── pages/
│   ├── 1_Home.py                   # Page 1: Home Dashboard
│   ├── 2_Classification.py         # Page 2: Interactive Text Classifier
│   ├── 3_Performance.py            # Page 3: Model Evaluation Metrics
│   ├── 4_Chatbot_Comparison.py     # Page 4: Classifier vs Chatbot Analysis
│   ├── 5_Dataset_Explorer.py       # Page 5: Dataset EDA & Browser
│   └── 6_About.py                  # Page 6: Documentation & Viva Guide
│
├── assets/
│   └── style.css                   # Premium dark navy & purple CSS theme
│
├── models/
│   └── classifier.joblib           # Trained pipeline (auto-generated)
│
├── data/
│   ├── SMSSpamCollection           # Raw UCI dataset (auto-downloaded)
│   └── cleaned_sms_dataset.csv     # Preprocessed dataset (auto-generated)
│
└── outputs/
    ├── test_metrics.json           # Verified evaluation metrics (auto-generated)
    ├── test_predictions.csv        # Model predictions on test split
    ├── chatbot_benchmark.csv       # 50-message benchmark for comparison
    └── template_chatbot_comparison.csv  # CSV template for custom chatbot upload
```

---

## 🚀 Quick Start (Local)

### 1. Clone the Repository

```bash
git clone https://github.com/punithavizhi/AI-Text-Classification-Mini-Project_30.git
cd AI-Text-Classification-Mini-Project_30
```

### 2. Create and Activate Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model (First Time Only)

This downloads the UCI SMS dataset (~600KB), trains the TF-IDF + Logistic Regression pipeline, and saves the model and evaluation metrics:

```bash
python train_model.py
```

Expected output:
```
[+] Dataset found/downloaded: data/SMSSpamCollection
[+] Loaded 5572 total messages (4825 ham, 747 spam)
[*] Training model on 4457 samples, testing on 1115 samples...
[+] Model saved to models/classifier.joblib
[+] Test metrics saved to outputs/test_metrics.json
MODEL PERFORMANCE SUMMARY:
Accuracy:  97.xx%
Spam Precision: 97.xx%
Spam Recall:    95.xx%
Spam F1-Score:  96.xx%
```

### 5. Launch the Application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🌐 Deployment on Streamlit Community Cloud

### Steps:

1. **Push all files to GitHub** (the repository already exists at the link above).

2. **Go to** [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.

3. **New App** → Select:
   - Repository: `punithavizhi/AI-Text-Classification-Mini-Project_30`
   - Branch: `main`
   - Main file path: `app.py`

4. **Click Deploy.**

> ⚡ **Auto-training on first deploy:** If `models/classifier.joblib` is not in the repository (which it won't be per `.gitignore`), the app automatically downloads the UCI dataset and trains the model on first launch. This takes ~30–60 seconds on Streamlit Cloud's free tier.

---

## 🧠 Machine Learning Pipeline

### Dataset
- **Source:** [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Size:** 5,572 SMS messages
- **Classes:** HAM (4,825 messages, 86.6%) | SPAM (747 messages, 13.4%)
- **Train/Test Split:** 80% / 20% (stratified, `random_state=42`)

### Feature Engineering: TF-IDF Vectorizer
```python
TfidfVectorizer(
    stop_words="english",   # Remove common English words
    ngram_range=(1, 2),     # Unigrams + Bigrams
    max_features=10000,     # Top 10,000 features
    sublinear_tf=True       # Log-scaled term frequency
)
```

### Classifier: Logistic Regression
```python
LogisticRegression(
    max_iter=1000,
    class_weight="balanced",  # Handle class imbalance
    random_state=42
)
```

---

## 📊 Model Performance

> Actual values computed after training — reported here as reference ranges:

| Metric | Score |
|--------|-------|
| Test Accuracy | ~97% |
| Spam Precision | ~97% |
| Spam Recall | ~94-96% |
| Spam F1-Score | ~96% |
| Ham F1-Score | ~98% |

*Exact figures are displayed live in the application from `outputs/test_metrics.json`.*

---

## ⚖️ Chatbot Comparison

The chatbot comparison page allows **two modes**:

1. **Pre-evaluated Benchmark (50 SMS):** A ready-to-use dataset with classifier predictions and simulated LLM predictions, generated from the test split.

2. **Custom CSV Upload:** Download our template, manually run the SMS messages through any chatbot (ChatGPT, Gemini, Claude), fill in the predictions, and upload the CSV.

### CSV Format Required:
```csv
message,actual_label,classifier_prediction,chatbot_prediction
"Congratulations! You've won a prize!",spam,spam,spam
"Hey, are you free for lunch?",ham,ham,ham
```

---

## 🧪 Test Cases Covered

| Test Case | Expected Behavior |
|-----------|-------------------|
| Promotional SMS | Classified as SPAM |
| Suspicious prize message | Classified as SPAM with high confidence |
| Normal personal SMS | Classified as HAM |
| Empty input | Warning shown, no crash |
| Long message (>500 chars) | Classified correctly |
| CSV upload (valid) | Comparison metrics computed |
| CSV upload (invalid columns) | Clear error message shown |
| Model file missing | Auto-trains model from dataset |

---

## 📖 Academic References

1. Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. (2011). *Contributions to the study of SMS Spam Filtering: New Collection and Results.* DocEng'11. ACM.
2. Scikit-learn: Machine Learning in Python — Pedregosa et al., JMLR 12, 2825–2830 (2011).
3. Manning, C.D., Raghavan, P., Schütze, H. (2008). *Introduction to Information Retrieval.* Cambridge University Press.

---

## 🎓 Key Viva Questions

**Q: Why TF-IDF + Logistic Regression over deep learning?**  
A: SMS messages are short (<160 chars) and spam relies on specific trigger keywords. This approach achieves >97% accuracy with sub-millisecond inference at <500KB model size.

**Q: How did you handle class imbalance?**  
A: Stratified train/test split + `class_weight='balanced'` in Logistic Regression + F1-score as primary metric.

**Q: What is the difference between False Positive and False Negative?**  
A: FP = legitimate message flagged as spam. FN = spam message that slips through. In SMS filtering, FPs are generally worse (blocking important personal messages).

---

## 👤 Author

**Punithavizhi** — Academic Mini-Project 30  
Repository: [AI-Text-Classification-Mini-Project_30](https://github.com/punithavizhi/AI-Text-Classification-Mini-Project_30)

---

*Built with ❤️ using Python, Streamlit, and Scikit-learn.*

