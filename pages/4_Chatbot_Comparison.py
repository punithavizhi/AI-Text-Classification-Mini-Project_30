"""
Page 4: Chatbot Comparison
Head-to-head empirical comparison between the Small Trained Classifier and an AI Chatbot (LLM).
"""

import io
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ml_utils import load_custom_css, compute_comparison_metrics, load_model

st.set_page_config(
    page_title="AI Text Classification | Chatbot Comparison",
    page_icon="⚖️",
    layout="wide"
)

# Load styling
load_custom_css()

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = BASE_DIR / "outputs"
BENCHMARK_PATH = OUTPUTS_DIR / "chatbot_benchmark.csv"
TEMPLATE_PATH = OUTPUTS_DIR / "template_chatbot_comparison.csv"

# Header
st.markdown("""
<div class="hero-card" style="padding: 1.5rem 1.75rem; margin-bottom: 1.5rem;">
    <div class="pill pill-purple" style="margin-bottom: 0.5rem;">Comparative Analysis</div>
    <div class="hero-title" style="font-size: 2rem;">Trained Classifier vs AI Chatbot</div>
    <div class="hero-subtitle">
        Evaluate how a lightweight domain-specific model (TF-IDF + Logistic Regression) performs against 
        a large conversational AI chatbot on identical test messages.
    </div>
</div>
""", unsafe_allow_html=True)

# Data source selection: Preloaded Benchmark vs Custom CSV Upload
st.markdown("### 📥 Select Comparison Data Source")
data_source = st.radio(
    "Choose evaluation dataset:",
    ["Use Pre-evaluated Test Benchmark (50 SMS Sample)", "Upload Your Own Chatbot Results CSV"],
    horizontal=True
)

comp_df = None

if data_source == "Use Pre-evaluated Test Benchmark (50 SMS Sample)":
    if BENCHMARK_PATH.exists():
        comp_df = pd.read_csv(BENCHMARK_PATH)
        st.success(f"✅ Loaded pre-evaluated benchmark dataset containing **{len(comp_df)}** SMS test messages.")
    else:
        # Fallback benchmark generator
        st.info("Generating standard benchmark dataset...")
        from train_model import generate_chatbot_artifacts
        test_pred_path = OUTPUTS_DIR / "test_predictions.csv"
        if test_pred_path.exists():
            tdf = pd.read_csv(test_pred_path)
            generate_chatbot_artifacts(tdf)
            comp_df = pd.read_csv(BENCHMARK_PATH)
        else:
            st.warning("Please run the training pipeline first to generate the test split.")

else:
    st.markdown("#### 📤 Upload Completed Chatbot Evaluation CSV")
    st.write(
        "Upload a CSV file containing your test messages and predictions. "
        "The CSV must contain four columns: `message`, `actual_label`, `classifier_prediction`, and `chatbot_prediction`."
    )

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    # Download Template Button
    st.markdown("##### Need the CSV Template?")
    template_content = ""
    if TEMPLATE_PATH.exists():
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            template_content = f.read()
    else:
        template_content = "message,actual_label,classifier_prediction,chatbot_prediction\n" \
                           "Hey are you free for lunch?,ham,ham,ham\n" \
                           "Win free gift card now!,spam,spam,spam\n"

    st.download_button(
        label="📥 Download CSV Comparison Template",
        data=template_content,
        file_name="chatbot_comparison_template.csv",
        mime="text/csv",
        use_container_width=False
    )

    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            req_cols = {"message", "actual_label", "classifier_prediction", "chatbot_prediction"}
            if not req_cols.issubset(set(user_df.columns)):
                st.error(f"❌ Uploaded CSV is missing required columns. Required: {', '.join(req_cols)}")
            else:
                comp_df = user_df
                st.success(f"✅ Successfully loaded {len(comp_df)} rows from uploaded CSV.")
        except Exception as e:
            st.error(f"❌ Error parsing CSV file: {e}")

# If we have valid comparison data
if comp_df is not None:
    try:
        results = compute_comparison_metrics(comp_df)
        valid_df = results["valid_df"]
        total_eval = results["total_evaluated"]
        cls_metrics = results["classifier"]
        bot_metrics = results["chatbot"]
        agreement = results["agreement_rate"]

        st.markdown("<br>", unsafe_allow_html=True)

        # Overview Metrics Side-by-Side
        st.markdown("### 📊 Head-to-Head Metric Summary")
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)

        with m_c1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="label">Total Tested</div>
                <div class="value">{total_eval:,}</div>
                <div class="detail">Identical messages</div>
            </div>
            """, unsafe_allow_html=True)

        with m_c2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="label">Agreement Rate</div>
                <div class="value">{agreement * 100:.1f}%</div>
                <div class="detail">Both models agreed</div>
            </div>
            """, unsafe_allow_html=True)

        with m_c3:
            cls_acc = cls_metrics["accuracy"] * 100
            bot_acc = bot_metrics["accuracy"] * 100
            delta_acc = cls_acc - bot_acc
            st.markdown(f"""
            <div class="stat-card">
                <div class="label">Classifier Accuracy</div>
                <div class="value" style="color: #a855f7;">{cls_acc:.1f}%</div>
                <div class="detail">{"▲" if delta_acc >= 0 else "▼"} {abs(delta_acc):.1f}% vs Chatbot</div>
            </div>
            """, unsafe_allow_html=True)

        with m_c4:
            st.markdown(f"""
            <div class="stat-card">
                <div class="label">Chatbot Accuracy</div>
                <div class="value" style="color: #38bdf8;">{bot_acc:.1f}%</div>
                <div class="detail">LLM Zero-Shot Accuracy</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Comparison Bar Chart
        chart_col, table_col = st.columns([1.3, 1.1])

        with chart_col:
            st.markdown("#### 📈 Metrics Comparison Chart")
            metric_names = ["Accuracy", "Spam Precision", "Spam Recall", "Spam F1-Score"]
            c_vals = [
                cls_metrics["accuracy"],
                cls_metrics["precision_spam"],
                cls_metrics["recall_spam"],
                cls_metrics["f1_spam"]
            ]
            b_vals = [
                bot_metrics["accuracy"],
                bot_metrics["precision_spam"],
                bot_metrics["recall_spam"],
                bot_metrics["f1_spam"]
            ]

            chart_data = []
            for name, cv, bv in zip(metric_names, c_vals, b_vals):
                chart_data.append({"Metric": name, "System": "Trained Classifier (Logistic Regression)", "Score": cv})
                chart_data.append({"Metric": name, "System": "AI Chatbot (LLM)", "Score": bv})

            fig_comp = px.bar(
                pd.DataFrame(chart_data),
                x="Metric",
                y="Score",
                color="System",
                barmode="group",
                color_discrete_map={
                    "Trained Classifier (Logistic Regression)": "#a855f7",
                    "AI Chatbot (LLM)": "#38bdf8"
                },
                text_auto=".1%"
            )

            fig_comp.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=30, b=30),
                height=340,
                yaxis=dict(range=[0.75, 1.05], tickformat=".0%"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        with table_col:
            st.markdown("#### 📋 Scorecard Table")
            scorecard_df = pd.DataFrame({
                "Evaluation Metric": metric_names,
                "Trained Classifier": [f"{v * 100:.2f}%" for v in c_vals],
                "AI Chatbot": [f"{v * 100:.2f}%" for v in b_vals],
                "Difference": [f"{(c - b) * 100:+.2f}%" for c, b in zip(c_vals, b_vals)]
            })
            st.dataframe(scorecard_df, use_container_width=True, hide_index=True)

            st.markdown("""
            <div style="background: rgba(15,23,42,0.8); border: 1px solid rgba(139,92,246,0.3); padding: 1rem; border-radius: 12px; margin-top: 1rem;">
                <span style="font-weight: 700; color: #c084fc;">Key Observation:</span><br>
                Both systems demonstrate strong text classification capabilities. The small trained model excels in 
                rapid, predictable keyword-weighted scoring, while LLMs exhibit deep semantic understanding but 
                require orders of magnitude more compute.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Disagreement & Error Inspection
        st.markdown("### 🔍 Head-to-Head Case Inspection")
        st.write("Examine where both models agreed vs where one of them failed:")

        # Add agreement column
        valid_df["Status"] = "Unknown"
        for i, row in valid_df.iterrows():
            c_right = (row["classifier_prediction"] == row["actual_label"])
            b_right = (row["chatbot_prediction"] == row["actual_label"])
            if c_right and b_right:
                valid_df.at[i, "Status"] = "✅ Both Correct"
            elif c_right and not b_right:
                valid_df.at[i, "Status"] = "🟣 Only Classifier Correct"
            elif not c_right and b_right:
                valid_df.at[i, "Status"] = "🔵 Only Chatbot Correct"
            else:
                valid_df.at[i, "Status"] = "❌ Both Wrong"

        status_filter = st.selectbox(
            "Filter messages by agreement status:",
            ["All Messages", "✅ Both Correct", "🟣 Only Classifier Correct", "🔵 Only Chatbot Correct", "❌ Both Wrong"]
        )

        display_df = valid_df if status_filter == "All Messages" else valid_df[valid_df["Status"] == status_filter]

        st.dataframe(
            display_df[["message", "actual_label", "classifier_prediction", "chatbot_prediction", "Status"]],
            use_container_width=True,
            hide_index=True
        )

    except Exception as ex:
        st.error(f"Error computing comparison: {ex}")

# Trade-off Analysis Section
st.divider()
st.markdown("### ⚖️ Architectural Trade-off Analysis")

t_col1, t_col2 = st.columns(2)

with t_col1:
    st.markdown("""
    #### 🟣 Small Classifier (TF-IDF + Logistic Regression)
    - **Inference Latency:** Sub-millisecond (~1ms).
    - **Hardware Requirements:** Minimal CPU & <50MB RAM. Runs on edge / mobile.
    - **API Costs:** $0.00 — completely free, offline inference.
    - **Determinism:** 100% deterministic (same input produces exact same output).
    - **Explainability:** Direct coefficients link words to spam probability.
    - **Limitation:** Can be fooled by deliberate misspellings (`fr33`, `w!n`) or novel unseen vocabulary.
    """)

with t_col2:
    st.markdown("""
    #### 🔵 AI Chatbot (Large Language Model)
    - **Inference Latency:** 300ms to 2000ms+ (network roundtrip).
    - **Hardware Requirements:** Cloud GPU clusters or high-end accelerators.
    - **API Costs:** Recurring tokens expense per API request.
    - **Determinism:** Dependent on temperature, top_p, and prompt formatting.
    - **Generalization:** Understands sarcasm, obfuscation, typos, and multi-lingual nuances.
    - **Limitation:** Overkill and expensive for high-volume SMS firewall pipelines.
    """)
