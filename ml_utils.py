"""
Shared Utilities for AI Text Classification Streamlit App
Handles model loading, dataset caching, CSS injection, and explainability.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
ASSETS_DIR = BASE_DIR / "assets"
MODEL_PATH = MODELS_DIR / "classifier.joblib"
METRICS_PATH = OUTPUTS_DIR / "test_metrics.json"
DATASET_PATH = DATA_DIR / "cleaned_sms_dataset.csv"
CSS_PATH = ASSETS_DIR / "style.css"


def load_custom_css():
    """Inject modern dark navy and purple CSS styling into Streamlit."""
    if CSS_PATH.exists():
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Load preprocessed SMS dataset, or download and clean if missing."""
    if not DATASET_PATH.exists():
        from train_model import download_dataset, load_and_preprocess_data
        raw_path = download_dataset()
        df = load_and_preprocess_data(raw_path)
        return df
    
    df = pd.read_csv(DATASET_PATH)
    # Ensure columns and types
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["message"] = df["message"].astype(str).str.strip()
    return df


@st.cache_resource
def load_model():
    """Load pre-trained Pipeline model or train it if not found."""
    if not MODEL_PATH.exists():
        from train_model import download_dataset, load_and_preprocess_data, train_and_evaluate
        raw_path = download_dataset()
        df = load_and_preprocess_data(raw_path)
        model, _ = train_and_evaluate(df)
        return model
    
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics() -> dict:
    """Load verified evaluation metrics from outputs/test_metrics.json."""
    if not METRICS_PATH.exists():
        from train_model import download_dataset, load_and_preprocess_data, train_and_evaluate
        raw_path = download_dataset()
        df = load_and_preprocess_data(raw_path)
        _, metrics = train_and_evaluate(df)
        return metrics

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def explain_prediction(model, text: str, top_n: int = 5):
    """
    Explain model decision by extracting top TF-IDF features and their Logistic Regression coefficients.
    Returns: label, confidence, spam_prob, ham_prob, top_indicators
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        return None, 0.0, 0.0, 0.0, []

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    classes = list(classifier.classes_)

    # Prediction & probability
    probs = model.predict_proba([cleaned_text])[0]
    pred_label = model.predict([cleaned_text])[0]
    confidence = float(probs[classes.index(pred_label)])

    spam_idx = classes.index("spam")
    ham_idx = classes.index("ham")
    spam_prob = float(probs[spam_idx])
    ham_prob = float(probs[ham_idx])

    # Extract active features
    feature_names = vectorizer.get_feature_names_out()
    coef = classifier.coef_[0]  # Positive coef = spam, Negative coef = ham
    
    X_vec = vectorizer.transform([cleaned_text])
    nonzero_indices = X_vec.nonzero()[1]

    indicators = []
    for idx in nonzero_indices:
        token = feature_names[idx]
        tf_idf_val = X_vec[0, idx]
        weight = coef[idx]
        score = tf_idf_val * weight
        indicators.append({
            "token": token,
            "weight": weight,
            "score": score,
            "influence": "SPAM" if score > 0 else "HAM"
        })

    # Sort by absolute impact
    indicators.sort(key=lambda x: abs(x["score"]), reverse=True)
    return pred_label, confidence, spam_prob, ham_prob, indicators[:top_n]


def compute_comparison_metrics(df: pd.DataFrame) -> dict:
    """
    Compute verified comparison metrics between classifier and chatbot on the same test set.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Normalize inputs
    clean_df = df.copy()
    clean_df["actual_label"] = clean_df["actual_label"].astype(str).str.strip().str.lower()
    clean_df["classifier_prediction"] = clean_df["classifier_prediction"].astype(str).str.strip().str.lower()
    clean_df["chatbot_prediction"] = clean_df["chatbot_prediction"].astype(str).str.strip().str.lower()

    # Filter strictly to ham/spam
    valid_mask = (
        clean_df["actual_label"].isin(["ham", "spam"]) &
        clean_df["classifier_prediction"].isin(["ham", "spam"]) &
        clean_df["chatbot_prediction"].isin(["ham", "spam"])
    )
    valid_df = clean_df[valid_mask].copy()

    if len(valid_df) == 0:
        raise ValueError("No valid rows found with labels in ['ham', 'spam'].")

    y_true = valid_df["actual_label"]
    y_cls = valid_df["classifier_prediction"]
    y_bot = valid_df["chatbot_prediction"]

    results = {
        "total_evaluated": len(valid_df),
        "classifier": {
            "accuracy": float(accuracy_score(y_true, y_cls)),
            "precision_spam": float(precision_score(y_true, y_cls, pos_label="spam", zero_division=0)),
            "recall_spam": float(recall_score(y_true, y_cls, pos_label="spam", zero_division=0)),
            "f1_spam": float(f1_score(y_true, y_cls, pos_label="spam", zero_division=0)),
            "precision_ham": float(precision_score(y_true, y_cls, pos_label="ham", zero_division=0)),
            "recall_ham": float(recall_score(y_true, y_cls, pos_label="ham", zero_division=0)),
            "f1_ham": float(f1_score(y_true, y_cls, pos_label="ham", zero_division=0)),
        },
        "chatbot": {
            "accuracy": float(accuracy_score(y_true, y_bot)),
            "precision_spam": float(precision_score(y_true, y_bot, pos_label="spam", zero_division=0)),
            "recall_spam": float(recall_score(y_true, y_bot, pos_label="spam", zero_division=0)),
            "f1_spam": float(f1_score(y_true, y_bot, pos_label="spam", zero_division=0)),
            "precision_ham": float(precision_score(y_true, y_bot, pos_label="ham", zero_division=0)),
            "recall_ham": float(recall_score(y_true, y_bot, pos_label="ham", zero_division=0)),
            "f1_ham": float(f1_score(y_true, y_bot, pos_label="ham", zero_division=0)),
        },
        "agreement_rate": float((y_cls == y_bot).mean()),
        "valid_df": valid_df
    }
    return results
