"""
Page 1: Home Dashboard
AI Text Classification: Small Trained Classifier vs Chatbot
"""

import streamlit as st
from ml_utils import load_custom_css, load_dataset, load_metrics

st.set_page_config(
    page_title="AI Text Classification | Home",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load styling
load_custom_css()

# Load real dataset and verified metrics
try:
    df = load_dataset()
    metrics = load_metrics()
except Exception as e:
    st.error(f"Error loading model or dataset: {e}")
    st.stop()

# Header Hero Card
st.markdown("""
<div class="hero-card">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
        <div>
            <div class="pill pill-purple" style="margin-bottom: 0.75rem;">Academic Mini-Project 30</div>
            <div class="hero-title">AI Text Classification Dashboard</div>
            <div class="hero-subtitle">
                A comparative analysis between a small trained machine-learning classifier (TF-IDF + Logistic Regression) 
                and an AI Chatbot (LLM) on SMS Spam and Ham classification.
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
        <div class="detail">SMS dataset entries</div>
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
st.write("Explore different sections of this mini-project using the navigation buttons below or the sidebar:")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔍</div>
        <div class="feature-title">Live Classification</div>
        <div class="feature-desc">Test real-time SMS messages with prediction confidence and key feature influences.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Text Classification ➔", key="nav_classify", use_container_width=True):
        st.switch_page("pages/2_Classification.py")

with c2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Model Performance</div>
        <div class="feature-desc">Inspect verified confusion matrix, precision, recall, and F1-score evaluations.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("View Model Performance ➔", key="nav_perf", use_container_width=True):
        st.switch_page("pages/3_Performance.py")

with c3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">⚖️</div>
        <div class="feature-title">Chatbot Comparison</div>
        <div class="feature-desc">Compare the trained classifier head-to-head with AI chatbot predictions on identical tests.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Chatbot Comparison ➔", key="nav_comp", use_container_width=True):
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
    if st.button("Explore Dataset ➔", key="nav_data", use_container_width=True):
        st.switch_page("pages/5_Dataset_Explorer.py")

with c5:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">ℹ️</div>
        <div class="feature-title">About & College Viva Guide</div>
        <div class="feature-desc">Architecture details, algorithms, trade-off analysis, and viva demonstration questions.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Read About Project ➔", key="nav_about", use_container_width=True):
        st.switch_page("pages/6_About.py")

st.divider()

# Core Architecture Callout
st.markdown("""
#### 🎯 Core Research Question
> **"How does a small, domain-specialized machine learning classifier compare with a general-purpose AI chatbot?"**
- **Small Classifier (TF-IDF + Logistic Regression)**: Low latency (<5ms), zero cloud API cost, deterministic, runable entirely on edge devices.
- **AI Chatbot (LLM)**: Rich contextual comprehension, zero-shot generalization, but higher latency, higher API expense, and possible hallucinations or prompt sensitivity.
""")
