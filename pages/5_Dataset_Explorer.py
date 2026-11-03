"""
Page 5: Dataset Explorer
Exploratory Data Analysis (EDA) of the UCI SMS Spam Collection dataset.
"""

from collections import Counter
import re
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ml_utils import load_custom_css, load_dataset

st.set_page_config(
    page_title="AI Text Classification | Dataset Explorer",
    page_icon="📂",
    layout="wide"
)

# Load styling
load_custom_css()

try:
    df = load_dataset()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Ensure message length features
if "char_length" not in df.columns:
    df["char_length"] = df["message"].str.len()
if "word_count" not in df.columns:
    df["word_count"] = df["message"].apply(lambda x: len(str(x).split()))

# Header
st.markdown("""
<div class="hero-card" style="padding: 1.5rem 1.75rem; margin-bottom: 1.5rem;">
    <div class="pill pill-purple" style="margin-bottom: 0.5rem;">Dataset Exploration</div>
    <div class="hero-title" style="font-size: 2rem;">UCI SMS Spam Collection Explorer</div>
    <div class="hero-subtitle">
        Analyze class distributions, message lengths, vocabulary frequency, and inspect raw dataset records.
    </div>
</div>
""", unsafe_allow_html=True)

# Overview Metrics
total_msgs = len(df)
ham_count = (df["label"] == "ham").sum()
spam_count = (df["label"] == "spam").sum()
ham_pct = (ham_count / total_msgs) * 100
spam_pct = (spam_count / total_msgs) * 100
avg_ham_len = df[df["label"] == "ham"]["char_length"].mean()
avg_spam_len = df[df["label"] == "spam"]["char_length"].mean()

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="label">Total Messages</div>
        <div class="value">{total_msgs:,}</div>
        <div class="detail">Cleaned dataset rows</div>
    </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="label">Ham (Legitimate)</div>
        <div class="value" style="color: #10b981;">{ham_count:,}</div>
        <div class="detail">{ham_pct:.1f}% majority class</div>
    </div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="label">Spam (Fraud / Ads)</div>
        <div class="value" style="color: #ef4444;">{spam_count:,}</div>
        <div class="detail">{spam_pct:.1f}% minority class</div>
    </div>
    """, unsafe_allow_html=True)
with m4:
    st.markdown(f"""
    <div class="stat-card">
        <div class="label">Avg Message Length</div>
        <div class="value">{avg_spam_len:.0f} <span style="font-size: 1rem; color: #94a3b8;">vs {avg_ham_len:.0f}</span></div>
        <div class="detail">Spam vs Ham chars</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Visualizations: Class distribution donut & length histogram
v_col1, v_col2 = st.columns([1, 1.4])

with v_col1:
    st.markdown("### 🍩 Class Imbalance Distribution")
    fig_donut = px.pie(
        names=["Ham (Legitimate)", "Spam (Unwanted)"],
        values=[ham_count, spam_count],
        color=["Ham (Legitimate)", "Spam (Unwanted)"],
        color_discrete_map={
            "Ham (Legitimate)": "#10b981",
            "Spam (Unwanted)": "#ef4444"
        },
        hole=0.55
    )
    fig_donut.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=20),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_donut, use_container_width=True)
    st.caption("Notice the natural class imbalance: ~86.6% Ham vs ~13.4% Spam. Balanced class weighting is applied during training.")

with v_col2:
    st.markdown("### 📊 Message Length Distribution (Characters)")
    fig_hist = px.histogram(
        df,
        x="char_length",
        color="label",
        nbins=50,
        barmode="overlay",
        color_discrete_map={"ham": "#10b981", "spam": "#ef4444"},
        labels={"char_length": "Character Length", "label": "Class"}
    )
    fig_hist.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=20),
        height=320,
        xaxis=dict(range=[0, 300]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption("Spam messages cluster heavily around 140-160 characters (maximizing standard 160-char SMS limits), while Ham messages are typically much shorter.")

st.markdown("<br>", unsafe_allow_html=True)

# Top Vocabulary Frequency in Ham vs Spam
st.markdown("### 🔤 Most Frequent Words (Excluding Stopwords)")

def get_top_words(texts, n=10):
    stop_words = {
        "i", "me", "my", "myself", "we", "our", "ours", "you", "your", "he", "him", "his",
        "she", "her", "it", "its", "they", "them", "what", "which", "who", "whom", "this",
        "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "a", "an", "the", "and", "but", "if",
        "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
        "against", "between", "into", "through", "during", "before", "after", "above", "below",
        "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
        "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ur", "u", "ok", "got", "get"
    }
    words = []
    for text in texts:
        tokens = re.findall(r'\b[a-zA-Z]{3,}\b', str(text).lower())
        words.extend([t for t in tokens if t not in stop_words])
    return Counter(words).most_common(n)

w_col1, w_col2 = st.columns(2)

with w_col1:
    st.markdown("##### 🚨 Top 10 Words in Spam Messages")
    top_spam = get_top_words(df[df["label"] == "spam"]["message"], n=10)
    spam_word_df = pd.DataFrame(top_spam, columns=["Word", "Frequency"])
    fig_w_spam = px.bar(
        spam_word_df,
        x="Frequency",
        y="Word",
        orientation="h",
        color_discrete_sequence=["#ef4444"]
    )
    fig_w_spam.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig_w_spam, use_container_width=True)

with w_col2:
    st.markdown("##### ✅ Top 10 Words in Ham Messages")
    top_ham = get_top_words(df[df["label"] == "ham"]["message"], n=10)
    ham_word_df = pd.DataFrame(top_ham, columns=["Word", "Frequency"])
    fig_w_ham = px.bar(
        ham_word_df,
        x="Frequency",
        y="Word",
        orientation="h",
        color_discrete_sequence=["#10b981"]
    )
    fig_w_ham.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig_w_ham, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Interactive Dataset Table & Search
st.markdown("### 🔎 Interactive Dataset Browser")

f_col1, f_col2 = st.columns([1, 2])
with f_col1:
    class_filter = st.selectbox("Filter by Class:", ["All", "Ham Only", "Spam Only"])
with f_col2:
    search_query = st.text_input("Search message text:", placeholder="e.g. prize, win, free, lecture, call...")

filtered_df = df.copy()
if class_filter == "Ham Only":
    filtered_df = filtered_df[filtered_df["label"] == "ham"]
elif class_filter == "Spam Only":
    filtered_df = filtered_df[filtered_df["label"] == "spam"]

if search_query.strip():
    filtered_df = filtered_df[filtered_df["message"].str.contains(search_query.strip(), case=False, na=False)]

st.write(f"Showing **{len(filtered_df):,}** matching messages:")
st.dataframe(
    filtered_df[["label", "message", "char_length", "word_count"]],
    use_container_width=True,
    height=320,
    hide_index=True
)

# Random Sample Generator
st.markdown("##### 🎲 Generate Random Samples:")
if st.button("Shuffle 5 Random Messages"):
    st.table(df.sample(5)[["label", "message"]])
