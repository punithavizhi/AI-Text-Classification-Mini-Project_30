"""
Page 2: Text Classification
Interactive SMS Spam and Ham prediction with explainability and model confidence.
"""

import streamlit as st
import pandas as pd
from ml_utils import load_custom_css, load_model, explain_prediction

st.set_page_config(
    page_title="AI Text Classification | Predict",
    page_icon="🔍",
    layout="wide"
)

# Load styling
load_custom_css()

# Load model
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading trained model: {e}")
    st.stop()

# Header
st.markdown("""
<div class="hero-card" style="padding: 1.5rem 1.75rem; margin-bottom: 1.5rem;">
    <div class="pill pill-purple" style="margin-bottom: 0.5rem;">Real-Time Inference</div>
    <div class="hero-title" style="font-size: 2rem;">Interactive SMS Classifier</div>
    <div class="hero-subtitle">
        Enter an SMS text below to classify it as <strong>SPAM</strong> or <strong>HAM</strong> using 
        the trained TF-IDF + Logistic Regression pipeline.
    </div>
</div>
""", unsafe_allow_html=True)

# Preset Samples
PRESET_SAMPLES = {
    "Suspicious Prize": "Congratulations! You have won a free $1000 cash prize or Walmart gift card. Claim now at http://win-claim.xyz!",
    "Promotional Offer": "Special 50% discount on all mobile accessories today only! Reply STOP to opt out.",
    "Normal Personal": "Hey mate, are we still meeting up for coffee after class today?",
    "Bank / Account Alert": "URGENT: Your bank debit card ending in 4120 has been temporarily locked. Call 0800-999-123 immediately.",
    "Colleague / Study": "Can you please share the machine learning assignment notes from yesterday's lecture?"
}

st.markdown("##### 💡 Try Example SMS Messages:")
sample_cols = st.columns(len(PRESET_SAMPLES))

# Initialize session state for text input
if "input_message" not in st.session_state:
    st.session_state["input_message"] = "Congratulations! You have won a free prize. Claim now!"

for idx, (label, text) in enumerate(PRESET_SAMPLES.items()):
    with sample_cols[idx]:
        if st.button(label, key=f"sample_{idx}", use_container_width=True):
            st.session_state["input_message"] = text

# Text input form
with st.container():
    user_input = st.text_area(
        "SMS Message Text",
        value=st.session_state["input_message"],
        height=140,
        placeholder="Type or paste any SMS message here...",
        help="Paste SMS text. Empty or whitespace-only messages will be flagged."
    )

    col_btn1, col_btn2, _ = st.columns([1.5, 1, 4])
    with col_btn1:
        classify_clicked = st.button("🚀 Classify Message", type="primary", use_container_width=True)
    with col_btn2:
        clear_clicked = st.button("🔄 Clear", type="secondary", use_container_width=True)

if clear_clicked:
    st.session_state["input_message"] = ""
    st.rerun()

# Run classification
if classify_clicked:
    if not user_input.strip():
        st.warning("⚠️ Please enter a non-empty SMS message to classify.")
    else:
        with st.spinner("Analyzing text features and computing prediction..."):
            pred_label, confidence, spam_prob, ham_prob, indicators = explain_prediction(model, user_input)

        st.markdown("<br>", unsafe_allow_html=True)

        res_col1, res_col2 = st.columns([1.2, 1.8])

        with res_col1:
            st.markdown("### Classification Result")
            if pred_label == "spam":
                st.markdown("""
                <div class="badge-spam">
                    🚨 PREDICTION: SPAM
                </div>
                """, unsafe_allow_html=True)
                st.caption("This message contains vocabulary and patterns typical of unsolicited or fraudulent SMS.")
            else:
                st.markdown("""
                <div class="badge-ham">
                    ✅ PREDICTION: HAM (NOT SPAM)
                </div>
                """, unsafe_allow_html=True)
                st.caption("This message appears legitimate and conversational with low deceptive markers.")

            st.markdown("<br>", unsafe_allow_html=True)
            st.metric("Model Confidence", f"{confidence * 100:.1f}%")

            # Probability progress bars
            st.write("**Probability Distribution:**")
            st.write(f"Spam probability: `{spam_prob * 100:.2f}%`")
            st.progress(spam_prob)
            st.write(f"Ham probability: `{ham_prob * 100:.2f}%`")
            st.progress(ham_prob)

        with res_col2:
            st.markdown("### 🔬 Model Explanation & Influence")
            st.write("Why did the classifier make this decision? Key tokens identified by TF-IDF & Logistic Regression:")

            if indicators:
                ind_df = pd.DataFrame(indicators)
                ind_df["Token"] = ind_df["token"]
                ind_df["Influence"] = ind_df["influence"].apply(
                    lambda x: "🚨 Spam Indicator" if x == "SPAM" else "✅ Ham Indicator"
                )
                ind_df["Weight"] = ind_df["score"].round(4)
                
                st.dataframe(
                    ind_df[["Token", "Influence", "Weight"]],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Explanation summary text
                top_word = indicators[0]["token"]
                top_influence = indicators[0]["influence"]
                st.info(
                    f"💡 **Key insight:** The token **`{top_word}`** had the strongest impact in pushing the prediction towards **{top_influence}**."
                )
            else:
                st.info("No strong single keywords found in vocabulary; classification was determined by intercept baseline.")

            st.caption("TF-IDF Vectorizer (ngram 1-2, sublinear scaling) + Logistic Regression balanced weights.")
