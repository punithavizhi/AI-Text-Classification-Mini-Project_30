"""
AI Text Classification: Small Trained Classifier vs Chatbot
Main Streamlit Application Entrypoint
Author: Academic Mini-Project 30
Dataset: UCI SMS Spam Collection
"""

import streamlit as st
from ml_utils import load_custom_css, load_dataset, load_metrics

# Page configuration
st.set_page_config(
    page_title="AI Text Classification: Classifier vs Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS theme
load_custom_css()

# Sidebar branding & quick info
with st.sidebar:
    st.markdown("### 🤖 Project Navigation")
    st.markdown("""
    <div style="background: rgba(30,27,75,0.45); border: 1px solid rgba(139,92,246,0.3); padding: 0.85rem; border-radius: 10px; margin-bottom: 1rem;">
        <span style="font-weight: 700; color: #c084fc;">AI Text Classification</span><br>
        <span style="font-size: 0.82rem; color: #94a3b8;">Mini-Project 30</span>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Use the pages menu above to navigate between features.")
    st.divider()
    st.markdown("""
    **Tech Stack:**
    - Python 3.11
    - Scikit-learn
    - TF-IDF Vectorizer
    - Logistic Regression
    - Streamlit & Plotly
    """)

# Load dataset and metrics
try:
    df = load_dataset()
    metrics = load_metrics()
except Exception as e:
    st.error(f"Error loading model or dataset: {e}")
    st.info("Please run `python train_model.py` to prepare dataset and model weights.")
    st.stop()

# Render Hero Banner
st.markdown("""
<div class="hero-card">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
        <div>
            <div class="pill pill-purple" style="margin-bottom: 0.75rem;">Academic Mini-Project 30</div>
            <div class="hero-title">AI Text Classification Dashboard</div>
            <div class="hero-subtitle">
                A modern comparative analysis between a small trained machine-learning text classifier 
                (TF-IDF + Logistic Regression) and an AI Chatbot (LLM) on SMS Spam and Ham detection.
            </div>
        </div>
        <div style="text-align: right;">
            <span class="pill pill-cyan">UCI SMS Spam Collection</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Overview Metric Cards
total_msgs = metrics.get("dataset_total", len(df))
ham_count = metrics.get("dataset_ham", sum(df["label"] == "ham"))
spam_count = metrics.get("dataset_spam", sum(df["label"] == "spam"))
accuracy = metrics.get("accuracy", 0.0) * 100
spam_prec = metrics.get("spam_precision", 0.0) * 100
spam_rec = metrics.get("spam_recall", 0.0) * 100

st.markdown(f"""
<div class="metric-grid">
    <div class="stat-card">
        <div class="label">Total Messages</div>
        <div class="value">{total_msgs:,}</div>
        <div class="detail">UCI SMS Spam Collection</div>
    </div>
    <div class="stat-card">
        <div class="label">Model Accuracy</div>
        <div class="value">{accuracy:.2f}%</div>
        <div class="detail">Held-out test set (20%)</div>
    </div>
    <div class="stat-card">
        <div class="label">Ham Messages</div>
        <div class="value">{ham_count:,}</div>
        <div class="detail">{ham_count / total_msgs * 100:.1f}% legitimate texts</div>
    </div>
    <div class="stat-card">
        <div class="label">Spam Messages</div>
        <div class="value">{spam_count:,}</div>
        <div class="detail">{spam_count / total_msgs * 100:.1f}% unwanted / fraud</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("### ⚡ Quick Navigation")
st.write("Jump straight into any page or explore the application using the buttons below:")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔍</div>
        <div class="feature-title">Live Classification</div>
        <div class="feature-desc">Test real-time SMS messages with prediction confidence and key feature influences.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Text Classification ➔", key="main_nav_classify", use_container_width=True):
        st.switch_page("pages/2_Classification.py")

with c2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Model Performance</div>
        <div class="feature-desc">Inspect verified confusion matrix, precision, recall, and F1-score evaluations.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("View Model Performance ➔", key="main_nav_perf", use_container_width=True):
        st.switch_page("pages/3_Performance.py")

with c3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">⚖️</div>
        <div class="feature-title">Chatbot Comparison</div>
        <div class="feature-desc">Compare the trained classifier head-to-head with AI chatbot predictions on identical tests.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Chatbot Comparison ➔", key="main_nav_comp", use_container_width=True):
        st.switch_page("pages/4_Chatbot_Comparison.py")

st.markdown("<br>", unsafe_allow_html=True)

c4, c5 = st.columns(2)

with c4:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📂</div>
        <div class="feature-title">Dataset Explorer</div>
        <div class="feature-desc">Browse SMS messages, inspect class distributions, and search text samples.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Explore Dataset ➔", key="main_nav_data", use_container_width=True):
        st.switch_page("pages/5_Dataset_Explorer.py")

with c5:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">ℹ️</div>
        <div class="feature-title">About & College Viva Guide</div>
        <div class="feature-desc">Architecture details, algorithms, trade-off analysis, and viva demonstration questions.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Read About Project ➔", key="main_nav_about", use_container_width=True):
        st.switch_page("pages/6_About.py")

st.divider()

st.markdown("""
#### 🎯 Core Research Question
> **"How does a small, domain-specialized machine learning classifier compare with a general-purpose AI chatbot?"**
- **Small Classifier (TF-IDF + Logistic Regression)**: Low latency (<2ms), zero API cost, deterministic, private, runable on edge devices.
- **AI Chatbot (LLM)**: Rich contextual comprehension, zero-shot generalization, but higher latency, per-token API cost, and variable outputs.
""")
