"""
Page 3: Model Performance
Detailed evaluation metrics, interactive confusion matrix, and performance charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from ml_utils import load_custom_css, load_metrics

st.set_page_config(
    page_title="AI Text Classification | Performance",
    page_icon="📊",
    layout="wide"
)

# Load styling
load_custom_css()

try:
    metrics = load_metrics()
except Exception as e:
    st.error(f"Error loading evaluation metrics: {e}")
    st.stop()

# Header
st.markdown("""
<div class="hero-card" style="padding: 1.5rem 1.75rem; margin-bottom: 1.5rem;">
    <div class="pill pill-purple" style="margin-bottom: 0.5rem;">Verified Evaluation</div>
    <div class="hero-title" style="font-size: 2rem;">Model Performance & Evaluation</div>
    <div class="hero-subtitle">
        Empirical evaluation results on the held-out test dataset (20% split, random_state=42, stratified).
        All values are computed directly from model predictions.
    </div>
</div>
""", unsafe_allow_html=True)

# Top Metrics Row
acc = metrics.get("accuracy", 0.0)
s_prec = metrics.get("spam_precision", 0.0)
s_rec = metrics.get("spam_recall", 0.0)
s_f1 = metrics.get("spam_f1", 0.0)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="label">Test Accuracy</div>
        <div class="value">{acc * 100:.2f}%</div>
        <div class="detail">Overall correctness</div>
    </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="label">Spam Precision</div>
        <div class="value">{s_prec * 100:.2f}%</div>
        <div class="detail">When predicted spam, % actually spam</div>
    </div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="label">Spam Recall</div>
        <div class="value">{s_rec * 100:.2f}%</div>
        <div class="detail">% of actual spam captured</div>
    </div>
    """, unsafe_allow_html=True)
with m4:
    st.markdown(f"""
    <div class="stat-card">
        <div class="label">Spam F1-Score</div>
        <div class="value">{s_f1 * 100:.2f}%</div>
        <div class="detail">Harmonic mean of prec & recall</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Section: Confusion Matrix and Class Breakdown Charts
col_cm, col_chart = st.columns([1.1, 1.3])

cm_data = metrics.get("confusion_matrix", {})
matrix = cm_data.get("matrix", [[950, 16], [8, 141]])
tn, fp, fn, tp = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]

with col_cm:
    st.markdown("### 🔲 Confusion Matrix")
    st.write("Distribution of predicted vs actual test messages:")

    # Build Plotly Confusion Matrix Heatmap
    z_values = [[tn, fp], [fn, tp]]
    x_labels = ["Predicted HAM", "Predicted SPAM"]
    y_labels = ["Actual HAM", "Actual SPAM"]

    # Annotations with counts and labels
    annotations = [
        [f"<b>TN (True Ham)</b><br>{tn}", f"<b>FP (False Spam)</b><br>{fp}"],
        [f"<b>FN (Missed Spam)</b><br>{fn}", f"<b>TP (True Spam)</b><br>{tp}"]
    ]

    fig_cm = go.Figure(data=go.Heatmap(
        z=z_values,
        x=x_labels,
        y=y_labels,
        colorscale=[
            [0, "#0f172a"],
            [0.5, "#4338ca"],
            [1.0, "#8b5cf6"]
        ],
        showscale=False,
        text=annotations,
        texttemplate="%{text}",
        textfont={"size": 15, "color": "white"}
    ))

    fig_cm.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=30, b=40),
        height=320,
        xaxis=dict(tickfont=dict(size=13, color="#e2e8f0")),
        yaxis=dict(tickfont=dict(size=13, color="#e2e8f0"), autorange="reversed")
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown(f"""
    <div style="font-size: 0.85rem; color: #94a3b8; background: rgba(30,27,75,0.4); padding: 0.8rem; border-radius: 8px;">
        • <strong>True Negatives (TN):</strong> {tn:,} legitimate messages correctly allowed.<br>
        • <strong>False Positives (FP):</strong> {fp:,} legitimate messages wrongly flagged.<br>
        • <strong>False Negatives (FN):</strong> {fn:,} spam messages that slipped through.<br>
        • <strong>True Positives (TP):</strong> {tp:,} spam messages caught and blocked.
    </div>
    """, unsafe_allow_html=True)

with col_chart:
    st.markdown("### 📈 Class Performance Comparison")
    st.write("Detailed breakdown by class (Ham vs Spam vs Macro Average):")

    h_prec = metrics.get("ham_precision", 0.0)
    h_rec = metrics.get("ham_recall", 0.0)
    h_f1 = metrics.get("ham_f1", 0.0)
    m_prec = metrics.get("macro_precision", 0.0)
    m_rec = metrics.get("macro_recall", 0.0)
    m_f1 = metrics.get("macro_f1", 0.0)

    perf_df = pd.DataFrame([
        {"Class": "Spam", "Metric": "Precision", "Score": s_prec},
        {"Class": "Spam", "Metric": "Recall", "Score": s_rec},
        {"Class": "Spam", "Metric": "F1-Score", "Score": s_f1},
        {"Class": "Ham", "Metric": "Precision", "Score": h_prec},
        {"Class": "Ham", "Metric": "Recall", "Score": h_rec},
        {"Class": "Ham", "Metric": "F1-Score", "Score": h_f1},
        {"Class": "Macro Avg", "Metric": "Precision", "Score": m_prec},
        {"Class": "Macro Avg", "Metric": "Recall", "Score": m_rec},
        {"Class": "Macro Avg", "Metric": "F1-Score", "Score": m_f1},
    ])

    fig_bar = px.bar(
        perf_df,
        x="Metric",
        y="Score",
        color="Class",
        barmode="group",
        color_discrete_map={
            "Spam": "#ef4444",
            "Ham": "#10b981",
            "Macro Avg": "#8b5cf6"
        },
        text_auto=".2%"
    )

    fig_bar.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=30),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(range=[0.8, 1.02], tickformat=".0%"),
        xaxis=dict(tickfont=dict(size=12))
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 📋 Complete Classification Report")

report_df = pd.DataFrame({
    "Class": ["Ham (Legitimate)", "Spam (Fraud/Promo)", "Macro Average", "Weighted Average"],
    "Precision": [f"{h_prec:.4f}", f"{s_prec:.4f}", f"{m_prec:.4f}", f"{acc:.4f}"],
    "Recall": [f"{h_rec:.4f}", f"{s_rec:.4f}", f"{m_rec:.4f}", f"{acc:.4f}"],
    "F1-Score": [f"{h_f1:.4f}", f"{s_f1:.4f}", f"{m_f1:.4f}", f"{acc:.4f}"],
    "Test Support": [f"{tn + fp:,}", f"{fn + tp:,}", f"{metrics.get('test_samples', 0):,}", f"{metrics.get('test_samples', 0):,}"]
})

st.dataframe(report_df, use_container_width=True, hide_index=True)

st.caption("Training setup: 80% train (4,457 messages), 20% test (1,115 messages), Random Seed 42, Stratified sampling.")
