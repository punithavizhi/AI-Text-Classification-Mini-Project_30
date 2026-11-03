"""
Page 6: About Project
Comprehensive academic documentation, algorithms, trade-offs, and college viva demonstration guide.
"""

import streamlit as st
from ml_utils import load_custom_css

st.set_page_config(
    page_title="AI Text Classification | About",
    page_icon="ℹ️",
    layout="wide"
)

# Load styling
load_custom_css()

# Header
st.markdown("""
<div class="hero-card" style="padding: 1.5rem 1.75rem; margin-bottom: 1.5rem;">
    <div class="pill pill-purple" style="margin-bottom: 0.5rem;">Documentation & Architecture</div>
    <div class="hero-title" style="font-size: 2rem;">About AI Text Classification</div>
    <div class="hero-subtitle">
        Academic mini-project comparing a small trained machine-learning model against conversational AI chatbots (LLMs)
        for SMS Spam and Ham classification.
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📖 Project Overview",
    "⚙️ Algorithms & Tech Stack",
    "⚖️ Advantages & Limitations",
    "🚀 Future Enhancements",
    "🎓 College Viva / Demo Guide"
])

with tab1:
    st.markdown("""
    ### 1. Project Introduction
    Unsolicited, deceptive, and fraudulent text messages (SMS spam) pose serious security and financial risks to mobile users 
    worldwide. Modern telecommunication platforms require automated, high-throughput text filtering mechanisms to detect spam in real time.
    
    ### 2. Problem Statement
    While Large Language Models (LLMs) like GPT-4, Claude, and Gemini demonstrate remarkable general language understanding, 
    deploying cloud-based LLM APIs for every incoming SMS across billions of users is cost-prohibitive, introduces network latency, 
    and raises privacy concerns. 
    
    **This research project investigates:**
    *Can a lightweight, classical machine-learning model (TF-IDF + Logistic Regression) achieve competitive classification performance 
    with near-zero operational cost and sub-millisecond inference speeds when compared to a conversational chatbot?*

    ### 3. Key Objectives
    1. **Build a working text classification system** capable of categorizing SMS into `SPAM` or `HAM`.
    2. **Train a machine-learning pipeline** using the standard UCI SMS Spam Collection dataset.
    3. **Create an interactive, responsive web application** using Streamlit and modern dark-themed UI.
    4. **Empirically compare the trained classifier with AI chatbot predictions** on identical test messages.
    5. **Display verified performance metrics**, interactive confusion matrices, and transparent explanations.
    6. **Publish a clean, reproducible GitHub repository** deployable to Streamlit Community Cloud.
    """)

with tab2:
    st.markdown("""
    ### 🧠 Machine Learning Algorithms

    #### 1. Text Representation: TF-IDF Vectorization
    Term Frequency-Inverse Document Frequency (TF-IDF) converts raw text strings into numerical sparse feature vectors:
    - **Term Frequency (TF):** Measures how frequently a word appears in a specific SMS message:
      $$\\text{TF}(t, d) = \\frac{f_{t,d}}{\\sum_{t' \\in d} f_{t',d}}$$
    - **Inverse Document Frequency (IDF):** Penalizes ubiquitous words (like *"the"*, *"is"*) that carry little discriminatory signal:
      $$\\text{IDF}(t, D) = \\log \\left( \\frac{1 + |D|}{1 + |\\{d \\in D : t \\in d\\}|} \\right) + 1$$
    - **Configuration in this project:**
      - Sublinear TF scaling (`sublinear_tf=True`) to dampen the effect of repeated tokens.
      - N-gram range `(1, 2)` capturing both unigrams (*"free"*, *"prize"*) and bigrams (*"claim now"*, *"gift card"*).
      - English stop-words filtering and top 10,000 feature limit.

    #### 2. Classification Algorithm: Logistic Regression
    Logistic Regression models the posterior probability of a message being spam using the logistic sigmoid function:
    $$P(y = \\text{spam} \\mid \\mathbf{x}) = \\sigma(\\mathbf{w}^T \\mathbf{x} + b) = \\frac{1}{1 + e^{-(\\mathbf{w}^T \\mathbf{x} + b)}}$$
    - **Class Weighting:** The SMS dataset is imbalanced (~86.6% Ham vs ~13.4% Spam). We apply `class_weight='balanced'` 
      so that errors on the minority spam class are penalized proportionally during loss optimization.
    - **Optimization:** L-BFGS solver with L2 regularization for rapid convergence and robust generalization.

    ---

    ### 💻 Technology Stack
    - **Frontend UI:** Streamlit 1.30+, Custom CSS3 with Dark Navy & Purple glassmorphism aesthetics.
    - **Machine Learning & Data Science:** Scikit-learn, Pandas, NumPy, Joblib.
    - **Interactive Visualizations:** Plotly Express & Plotly Graph Objects.
    - **Benchmark Source:** UCI Machine Learning Repository (SMS Spam Collection, 5,574 messages).
    - **Deployment Platform:** Streamlit Community Cloud / GitHub.
    """)

with tab3:
    st.markdown("### ⚖️ Architectural Comparison: Small Classifier vs AI Chatbot")

    c_adv, c_lim = st.columns(2)

    with c_adv:
        st.markdown("""
        #### ✅ Advantages of Small Trained Classifier
        - **⚡ Blazing Speed:** Sub-millisecond latency (<2ms per message) vs 500-2000ms for LLMs.
        - **💰 Zero API Expense:** No recurring per-token or per-call costs. Runs offline indefinitely.
        - **🔒 Total Data Privacy:** SMS messages are processed locally without transmitting personal texts to third-party servers.
        - **🎯 100% Deterministic:** Same text always produces the exact same classification and confidence score.
        - **🔬 Transparent & Explainable:** Directly inspectable feature weights and decision boundary.
        - **📱 Ultra-Lightweight:** Model file is <500KB; easily embeds in iOS/Android apps or carrier gateway firewalls.
        """)

    with c_lim:
        st.markdown("""
        #### ⚠️ Limitations & Chatbot Counterparts
        - **Vocabulary Boundary:** The TF-IDF model struggles when spammers use creative character substitution (`FR33`, `C!a1m`, zero-width spaces) unless explicitly preprocessed.
        - **Semantic Nuance:** Chatbots understand context, irony, idioms, and multi-sentence conversational flow that bag-of-words models miss.
        - **Cold-Start on New Domains:** Small models trained on SMS cannot classify phishing emails or Discord spam without retraining on those distributions.
        - **Chatbot Drawbacks:** LLMs suffer from high compute footprint, cloud dependency, potential hallucinations, and prompt sensitivity.
        """)

with tab4:
    st.markdown("""
    ### 🚀 Future Enhancements
    1. **Adversarial Robustness & Levenshtein Normalization:** Implement phonetic/character-level normalization (e.g. converting `p-r-i-z-e` or `w!n` to canonical spellings) before vectorization.
    2. **Hybrid / Cascaded Pipeline:** Use the small classifier as a first-line filter for 95% of traffic (<1ms), and escalate ambiguous/low-confidence messages (40%-60% confidence) to an LLM.
    3. **Real-Time Model Feedback Loop:** Add a user feedback button ("Report False Positive / False Negative") to continuously retrain and fine-tune model weights.
    4. **Multilingual Spam Detection:** Expand vocabulary and training set to handle multilingual SMS spam across regional dialects (Hinglish, Spanish, etc.).
    5. **Browser Extension & Carrier Hook:** Package the classifier into a lightweight Chrome extension or mobile companion app.
    """)

with tab5:
    st.markdown("""
    ### 🎓 College Viva & Demonstration Guide

    > **Tip for students:** Examiners love to test whether you truly understand the machine learning principles behind your project rather than just running code. Here are the most frequently asked questions and how to answer them:

    #### Q1: Why did you choose TF-IDF + Logistic Regression over deep learning (like LSTM or BERT)?
    * **Answer:** SMS messages are concise (typically <160 characters) and spam patterns rely heavily on distinct trigger keywords (*"free"*, *"claim"*, *"urgent"*, *"prize"*). For this domain, TF-IDF + Logistic Regression achieves >97% accuracy with sub-millisecond inference and tiny memory footprint (<500KB). A heavy Transformer model would be overkill, 1000x slower, and consume gigabytes of VRAM for negligible accuracy gain.

    #### Q2: How did you handle class imbalance in the dataset?
    * **Answer:** In the UCI SMS dataset, ~86.6% are Ham and only ~13.4% are Spam. If a naive model always predicts "Ham", it would achieve 86.6% accuracy but be useless. We addressed this by:
      1. Using **Stratified 80/20 train/test splitting** to preserve class proportions in both sets.
      2. Setting `class_weight='balanced'` in Logistic Regression, which automatically assigns higher penalty weights to misclassifying the minority Spam class.
      3. Focusing our evaluation on **Precision, Recall, and F1-score**, rather than relying solely on raw Accuracy.

    #### Q3: What is the difference between False Positive and False Negative in this context? Which is worse?
    * **Answer:** 
      - **False Positive (FP / Type I error):** A legitimate Ham message is wrongly flagged as Spam.
      - **False Negative (FN / Type II error):** A spam message is missed and delivered to the inbox.
      - *In SMS filtering, False Positives are generally considered worse* because blocking an important personal, medical, or two-factor authentication message causes user frustration and loss of trust.

    #### Q4: How is the Chatbot Comparison conducted without fabricating numbers?
    * **Answer:** We take an identical held-out test sample of SMS messages. We run the messages through the trained classifier, and evaluate the exact same messages with the chatbot under a zero-shot prompt template without showing true labels. We compute standard metrics (Accuracy, Precision, Recall, F1) on both outputs using identical scikit-learn evaluation functions. Users can also download our template CSV, test their own chatbot, and upload the results.
    """)
