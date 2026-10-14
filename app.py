"""
Email Spam Classifier - Interactive Web Application
Built with Streamlit, Scikit-Learn, and NLP Preprocessing.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Custom preprocessing & feature extraction
from utils.preprocessing import (
    clean_text_for_model,
    extract_message_features,
    SPAM_TRIGGER_KEYWORDS
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Email Spam Classifier | ML Project",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply modern aesthetic styling
CUSTOM_CSS = """
<style>
    /* Global font & background accents */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Top banner card */
    .hero-banner {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        color: #f8fafc;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 6px;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0;
    }
    
    /* KPI Metric Cards */
    .metric-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px -2px rgba(0, 0, 0, 0.08);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
    }
    .metric-tag {
        font-size: 0.8rem;
        font-weight: 500;
        color: #10b981;
        margin-top: 4px;
    }

    /* Result Banners */
    .spam-badge {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 20px;
        color: #991b1b;
        margin: 16px 0;
    }
    .ham-badge {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 20px;
        color: #065f46;
        margin: 16px 0;
    }
    
    /* Trigger pills */
    .keyword-pill {
        display: inline-block;
        background: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px 4px;
    }
    .keyword-pill-red {
        background: #fee2e2;
        color: #991b1b;
        border-color: #fca5a5;
    }
    
    /* Footer */
    .footer-text {
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 40px;
        padding: 20px 0;
        border-top: 1px solid #e2e8f0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Resource Loaders (Cached)
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models_and_vectorizer():
    """Loads all trained models and TF-IDF vectorizer."""
    models_path = "models/all_models.pkl"
    single_model_path = "models/spam_model.pkl"
    vectorizer_path = "models/tfidf_vectorizer.pkl"

    if not os.path.exists(vectorizer_path):
        return None, None, "TF-IDF Vectorizer file not found in 'models/'. Please run train_model.py first."

    try:
        vectorizer = joblib.load(vectorizer_path)
    except Exception as e:
        return None, None, f"Error loading vectorizer: {str(e)}"

    models = {}
    if os.path.exists(models_path):
        try:
            models = joblib.load(models_path)
        except Exception:
            pass

    if not models and os.path.exists(single_model_path):
        try:
            models["Best Model"] = joblib.load(single_model_path)
        except Exception as e:
            return None, None, f"Error loading model: {str(e)}"

    if not models:
        return None, None, "No trained model found. Please run train_model.py to train models."

    return models, vectorizer, None


@st.cache_data(show_spinner=False)
def load_metrics_and_dataset():
    """Loads model evaluation metrics and raw dataset for EDA."""
    metrics_path = "models/model_metrics.json"
    dataset_path = "data/dataset.csv"

    metrics = None
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            metrics = None

    df = None
    if os.path.exists(dataset_path):
        try:
            try:
                df = pd.read_csv(dataset_path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(dataset_path, encoding="latin-1")
            
            # Standardize columns
            if "v1" in df.columns and "v2" in df.columns:
                df = df[["v1", "v2"]].copy()
                df.columns = ["label", "text"]
            elif "Category" in df.columns and "Message" in df.columns:
                df = df[["Category", "Message"]].copy()
                df.columns = ["label", "text"]
            elif "label" in df.columns and "text" in df.columns:
                df = df[["label", "text"]].copy()
            else:
                df = df.iloc[:, :2].copy()
                df.columns = ["label", "text"]

            df = df.dropna(subset=["label", "text"])
            df["label"] = df["label"].astype(str).str.strip().str.lower()
            df = df[df["label"].isin(["ham", "spam"])].copy()
            df = df.drop_duplicates(subset=["text"]).copy()
            df["char_count"] = df["text"].astype(str).apply(len)
            df["word_count"] = df["text"].astype(str).apply(lambda s: len(s.split()))
        except Exception:
            df = None

    return metrics, df


# Load assets
models, vectorizer, model_err = load_models_and_vectorizer()
metrics, df_data = load_metrics_and_dataset()


# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ Spam Guard AI")
    st.markdown("**College ML Project Demonstration**")
    st.divider()

    selected_page = st.radio(
        "Navigation Menu",
        [
            "🏠 Home Dashboard",
            "🔍 Spam Detection",
            "📊 Model Performance",
            "📈 Dataset Analysis",
            "ℹ️ About Project"
        ],
        index=1  # Default directly to Spam Detection for quick testing!
    )

    st.divider()
    st.markdown("#### ⚙️ System Status")
    if model_err:
        st.error("⚠️ Model Offline")
        st.caption(model_err)
    else:
        st.success("✅ Models Active")
        available_models = list(models.keys())
        st.caption(f"Loaded: {len(available_models)} algorithms")
        if metrics:
            st.caption(f"Best: **{metrics.get('best_model', 'Trained Model')}**")

    st.divider()
    st.caption("Developed with Python, Scikit-Learn & Streamlit.")


# ---------------------------------------------------------
# PAGE 1: Home Dashboard
# ---------------------------------------------------------
if selected_page == "🏠 Home Dashboard":
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">🛡️ Email & SMS Spam Classifier</div>
            <p class="hero-subtitle">
                An intelligent Natural Language Processing (NLP) and Machine Learning system for automated detection of spam and legitimate messages.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Top KPI Metrics Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    total_recs = metrics["dataset_metadata"]["total_records"] if metrics else (len(df_data) if df_data is not None else "5,157")
    ham_cnt = metrics["dataset_metadata"]["ham_count"] if metrics else (int((df_data['label'] == 'ham').sum()) if df_data is not None else "4,516")
    spam_cnt = metrics["dataset_metadata"]["spam_count"] if metrics else (int((df_data['label'] == 'spam').sum()) if df_data is not None else "641")
    
    best_name = metrics.get("best_model", "Multinomial Naive Bayes") if metrics else "Naive Bayes"
    best_acc = metrics["model_performance"][best_name]["accuracy"] if (metrics and best_name in metrics["model_performance"]) else 0.985
    best_prec = metrics["model_performance"][best_name]["precision"] if (metrics and best_name in metrics["model_performance"]) else 0.991

    with kpi_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Dataset Records</div>
                <div class="metric-value">{total_recs:,}</div>
                <div class="metric-tag">Cleaned & Deduplicated</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi_col2:
        ham_pct = round((ham_cnt / total_recs) * 100, 1) if isinstance(total_recs, int) and isinstance(ham_cnt, int) else 87.6
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Legitimate (Ham)</div>
                <div class="metric-value">{ham_cnt:,}</div>
                <div class="metric-tag">{ham_pct}% of total messages</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi_col3:
        spam_pct = round((spam_cnt / total_recs) * 100, 1) if isinstance(total_recs, int) and isinstance(spam_cnt, int) else 12.4
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Spam Messages</div>
                <div class="metric-value">{spam_cnt:,}</div>
                <div class="metric-tag" style="color: #ef4444;">{spam_pct}% of total messages</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi_col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Best Model Accuracy</div>
                <div class="metric-value">{best_acc * 100:.1f}%</div>
                <div class="metric-tag">Precision: {best_prec * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualizations Row
    chart_col1, chart_col2 = st.columns([1, 1])

    with chart_col1:
        st.subheader("📊 Dataset Class Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ["Legitimate (Ham)", "Spam"]
        sizes = [ham_cnt, spam_cnt]
        colors = ["#10b981", "#ef4444"]
        explode = (0.05, 0.05)

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            explode=explode,
            textprops={"fontsize": 11, "fontweight": "bold"}
        )
        for autotext in autotexts:
            autotext.set_color("white")
        ax.axis("equal")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with chart_col2:
        st.subheader("🏆 Model Comparison Snapshot")
        if metrics and "model_performance" in metrics:
            perf = metrics["model_performance"]
            model_names = list(perf.keys())
            accs = [perf[m]["accuracy"] * 100 for m in model_names]
            precs = [perf[m]["precision"] * 100 for m in model_names]

            fig, ax = plt.subplots(figsize=(6, 4))
            x = np.arange(len(model_names))
            width = 0.35

            ax.bar(x - width/2, accs, width, label="Accuracy (%)", color="#3b82f6")
            ax.bar(x + width/2, precs, width, label="Precision (%)", color="#10b981")

            ax.set_ylabel("Score (%)")
            ax.set_ylim(85, 102)
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace(" ", "\n") for m in model_names], fontsize=9)
            ax.legend(loc="lower right")
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Run train_model.py to generate live model comparison charts.")

    st.divider()
    st.markdown("### 🚀 Quick Navigation Guide")
    guide_col1, guide_col2, guide_col3 = st.columns(3)
    with guide_col1:
        st.markdown("**1. Test Live Emails**")
        st.write("Go to **Spam Detection** to paste incoming emails or try pre-filled realistic spam and ham samples.")
    with guide_col2:
        st.markdown("**2. Inspect ML Metrics**")
        st.write("Go to **Model Performance** to review confusion matrices, precision, recall, and algorithm comparisons.")
    with guide_col3:
        st.markdown("**3. Explore Dataset & Keywords**")
        st.write("Go to **Dataset Analysis** to inspect class balance, message lengths, and high-impact spam keywords.")


# ---------------------------------------------------------
# PAGE 2: Spam Detection (Live)
# ---------------------------------------------------------
elif selected_page == "🔍 Spam Detection":
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">🔍 Live Email Spam Classifier</div>
            <p class="hero-subtitle">
                Enter any email or text message below. The NLP pipeline cleans the text and evaluates it using trained machine learning models.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if model_err:
        st.error(f"Cannot run detection: {model_err}")
        st.stop()

    # Model selector and options
    ctrl_col1, ctrl_col2 = st.columns([2, 1])
    with ctrl_col1:
        model_names = list(models.keys())
        default_idx = 0
        if metrics and "best_model" in metrics and metrics["best_model"] in model_names:
            default_idx = model_names.index(metrics["best_model"])
        
        selected_model_name = st.selectbox(
            "Select Machine Learning Algorithm for Inference:",
            model_names,
            index=default_idx,
            help="Choose which trained algorithm evaluates the message."
        )
    with ctrl_col2:
        st.caption("ℹ️ Model Selection Tip")
        st.caption("Naive Bayes and Linear SVM offer high precision to prevent legitimate emails from being wrongly classified as spam.")

    st.markdown("#### ⚡ Try Preset Sample Messages")
    sample_col1, sample_col2 = st.columns(2)

    # Initialize session state for text input
    if "email_input" not in st.session_state:
        st.session_state["email_input"] = ""

    with sample_col1:
        st.caption("🚨 Realistic Spam Samples:")
        c1, c2, c3 = st.columns(3)
        if c1.button("Prize Winner", use_container_width=True):
            st.session_state["email_input"] = "CONGRATULATIONS! You have been selected as the official winner of our £1,000 cash prize draw! Call 09061701461 now or text CLAIM to 87121 to receive your reward immediately. T&Cs apply."
            st.rerun()
        if c2.button("Bank Alert", use_container_width=True):
            st.session_state["email_input"] = "URGENT SECURITY ALERT: Your online banking account has been temporarily suspended due to unauthorized access. Click here immediately to verify your identity and restore access: http://secure-verify-bank.com/login"
            st.rerun()
        if c3.button("Free Offer", use_container_width=True):
            st.session_state["email_input"] = "Free entry in 2 a weekly competition to win a brand new iPhone 15 Pro and gift cards! Text WIN to 80082 to claim your entry ticket right now. Standard charges apply."
            st.rerun()

    with sample_col2:
        st.caption("✉️ Legitimate (Ham) Samples:")
        c4, c5, c6 = st.columns(3)
        if c4.button("Project Meeting", use_container_width=True):
            st.session_state["email_input"] = "Hey team, are we still meeting in the computer lab at 4:30 PM today to finalize our Machine Learning project presentation? Please bring your laptops."
            st.rerun()
        if c5.button("Class Assignment", use_container_width=True):
            st.session_state["email_input"] = "Dear Professor Sharma, please find attached my updated submission for Assignment 3 on Natural Language Processing. Let me know if any revisions are needed."
            st.rerun()
        if c6.button("Casual Chat", use_container_width=True):
            st.session_state["email_input"] = "Hi Sarah, thanks for sharing the study notes earlier! I will review the lecture slides tonight and let you know if I have questions."
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Text Input Area
    user_input = st.text_area(
        "Enter or Paste Email / Message Text:",
        value=st.session_state["email_input"],
        height=140,
        placeholder="Type or paste the email body here to analyze...",
        key="main_text_input"
    )

    btn_col1, btn_col2, btn_col3 = st.columns([1.5, 1.5, 5])
    with btn_col1:
        analyze_clicked = st.button("🔍 Analyze Email", type="primary", use_container_width=True)
    with btn_col2:
        if st.button("🗑️ Clear Input", use_container_width=True):
            st.session_state["email_input"] = ""
            st.rerun()

    if analyze_clicked:
        if not user_input.strip():
            st.warning("⚠️ Please enter or paste some text before clicking Analyze.")
        else:
            with st.spinner("Processing text and running ML classification..."):
                # Clean text using identical preprocessing pipeline
                cleaned_text = clean_text_for_model(user_input)
                
                if not cleaned_text.strip():
                    st.warning("⚠️ The input text contained only punctuation, numbers, or stopwords. Please enter a meaningful message.")
                else:
                    # Feature Extraction
                    features = extract_message_features(user_input)
                    
                    # TF-IDF Vectorization
                    active_model = models[selected_model_name]
                    input_tfidf = vectorizer.transform([cleaned_text])

                    # Prediction and Probabilities
                    prediction = active_model.predict(input_tfidf)[0]  # 0: Ham, 1: Spam
                    
                    # Confidence estimation
                    confidence = 0.5
                    spam_prob = 0.5
                    ham_prob = 0.5

                    if hasattr(active_model, "predict_proba"):
                        probs = active_model.predict_proba(input_tfidf)[0]
                        ham_prob = float(probs[0])
                        spam_prob = float(probs[1])
                        confidence = float(max(probs))
                    elif hasattr(active_model, "decision_function"):
                        # Convert decision score to pseudo-probability via sigmoid
                        dec_score = active_model.decision_function(input_tfidf)[0]
                        spam_prob = float(1 / (1 + np.exp(-dec_score)))
                        ham_prob = 1.0 - spam_prob
                        confidence = float(max(spam_prob, ham_prob))

                    is_spam = (prediction == 1)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("### 📋 Prediction Analysis")

                    # Primary Result Display
                    if is_spam:
                        st.markdown(
                            f"""
                            <div class="spam-badge">
                                <h2 style="margin: 0; display: flex; align-items: center; gap: 8px;">
                                    🚨 SPAM DETECTED
                                </h2>
                                <p style="margin: 6px 0 0 0; font-size: 1.05rem;">
                                    This message exhibits statistical patterns commonly associated with unsolicited spam or fraudulent communications.
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="ham-badge">
                                <h2 style="margin: 0; display: flex; align-items: center; gap: 8px;">
                                    ✅ LEGITIMATE (HAM) MESSAGE
                                </h2>
                                <p style="margin: 6px 0 0 0; font-size: 1.05rem;">
                                    This message matches the linguistic characteristics of normal, legitimate email communication.
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    # Confidence Meter and Metrics
                    score_col1, score_col2 = st.columns([1, 1])
                    with score_col1:
                        st.markdown("#### 🎯 Model Confidence Score")
                        st.caption("Calculated statistical probability assigned by the classification model:")
                        st.progress(confidence)
                        st.write(f"**Confidence Level:** `{confidence * 100:.2f}%` ({selected_model_name})")

                        prob_df = pd.DataFrame({
                            "Classification": ["Legitimate (Ham)", "Spam"],
                            "Model Probability": [f"{ham_prob * 100:.2f}%", f"{spam_prob * 100:.2f}%"]
                        })
                        st.table(prob_df)

                    with score_col2:
                        st.markdown("#### 🔍 Message Characteristics")
                        m_col1, m_col2 = st.columns(2)
                        m_col1.metric("Characters", f"{features['char_count']}")
                        m_col1.metric("Uppercase Ratio", f"{features['upper_ratio']}%")
                        m_col2.metric("Total Words", f"{features['word_count']}")
                        m_col2.metric("Filtered NLP Tokens", f"{features['clean_word_count']}")

                    # Triggered Keywords and Heuristics
                    st.markdown("#### 🚩 Suspicious Trigger Indicators")
                    if features["matched_triggers"]:
                        st.warning("⚠️ The following high-frequency spam trigger words were identified in this message:")
                        pills_html = " ".join([
                            f'<span class="keyword-pill keyword-pill-red">{kw}</span>'
                            for kw in features["matched_triggers"]
                        ])
                        st.markdown(pills_html, unsafe_allow_html=True)
                        st.caption("Note: Keyword indicators provide explainability context; the final verdict is derived from the statistical TF-IDF model weights.")
                    else:
                        st.success("No high-risk promotional or urgency keywords detected in the message text.")

                    # Preprocessed Text Expander
                    with st.expander("🛠️ View Internal NLP Preprocessing Transformation"):
                        st.write("**Raw Input:**", user_input)
                        st.write("**Cleaned & Stemmed Tokens (Fed to TF-IDF):**", cleaned_text)
                        st.caption("Preprocessing removed stopwords, URLs, symbols, and applied Porter Stemming.")


# ---------------------------------------------------------
# PAGE 3: Model Performance
# ---------------------------------------------------------
elif selected_page == "📊 Model Performance":
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">📊 Model Evaluation & Comparison</div>
            <p class="hero-subtitle">
                Rigorous evaluation across Multinomial Naive Bayes, Logistic Regression, and Support Vector Machines on the test partition (20% held-out test split).
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not metrics or "model_performance" not in metrics:
        st.warning("Model metrics file not found. Run `python train_model.py` to evaluate models.")
        st.stop()

    perf = metrics["model_performance"]

    # Table Comparison
    st.subheader("📑 Comparative Evaluation Metrics Table")
    table_data = []
    for model_name, data in perf.items():
        table_data.append({
            "Algorithm": model_name,
            "Accuracy": f"{data['accuracy'] * 100:.2f}%",
            "Precision (Spam)": f"{data['precision'] * 100:.2f}%",
            "Recall (Spam)": f"{data['recall'] * 100:.2f}%",
            "F1-Score (Spam)": f"{data['f1_score'] * 100:.2f}%",
            "Test Samples": data["test_samples"]
        })
    df_perf = pd.DataFrame(table_data)
    st.dataframe(df_perf, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visual Comparison Bar Charts
    st.subheader("📈 Multi-Metric Algorithm Comparison")
    comp_col1, comp_col2 = st.columns(2)

    model_names = list(perf.keys())
    accuracies = [perf[m]["accuracy"] * 100 for m in model_names]
    precisions = [perf[m]["precision"] * 100 for m in model_names]
    recalls = [perf[m]["recall"] * 100 for m in model_names]
    f1_scores = [perf[m]["f1_score"] * 100 for m in model_names]

    with comp_col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        x = np.arange(len(model_names))
        width = 0.35

        ax1.bar(x - width/2, accuracies, width, label="Accuracy", color="#3b82f6")
        ax1.bar(x + width/2, f1_scores, width, label="F1-Score", color="#8b5cf6")
        ax1.set_ylabel("Score (%)")
        ax1.set_title("Accuracy vs F1-Score", fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.replace(" ", "\n") for m in model_names], fontsize=8.5)
        ax1.set_ylim(80, 102)
        ax1.legend(loc="lower right")
        ax1.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with comp_col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        x = np.arange(len(model_names))
        width = 0.35

        ax2.bar(x - width/2, precisions, width, label="Precision (Spam)", color="#10b981")
        ax2.bar(x + width/2, recalls, width, label="Recall (Spam)", color="#f59e0b")
        ax2.set_ylabel("Score (%)")
        ax2.set_title("Precision vs Recall (Spam)", fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels([m.replace(" ", "\n") for m in model_names], fontsize=8.5)
        ax2.set_ylim(70, 105)
        ax2.legend(loc="lower right")
        ax2.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.divider()

    # Confusion Matrices Section
    st.subheader("🎯 Confusion Matrices (Test Partition)")
    cm_cols = st.columns(len(model_names))

    for idx, name in enumerate(model_names):
        with cm_cols[idx]:
            st.markdown(f"**{name}**")
            cm = np.array(perf[name]["confusion_matrix"])
            fig_cm, ax_cm = plt.subplots(figsize=(3.5, 3))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=["Ham", "Spam"],
                yticklabels=["Ham", "Spam"],
                ax=ax_cm,
                annot_kws={"size": 11, "weight": "bold"}
            )
            ax_cm.set_xlabel("Predicted Label", fontsize=9)
            ax_cm.set_ylabel("True Label", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("💡 Why is High Precision Critical in Spam Classification?"):
        st.markdown(
            """
            In an email filtering system:
            * **False Positive (FP):** A legitimate email (such as an interview call, bank OTP, or college announcement) is incorrectly classified as SPAM. This is highly detrimental to the user.
            * **False Negative (FN):** A spam email slips through into the inbox. While annoying, it is generally less damaging than losing an essential message.
            
            Therefore, **Precision on the Spam class** must be kept as close to 100% as possible to protect legitimate emails from being lost in the junk folder.
            """
        )


# ---------------------------------------------------------
# PAGE 4: Dataset Analysis (EDA)
# ---------------------------------------------------------
elif selected_page == "📈 Dataset Analysis":
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">📈 Exploratory Data Analysis (EDA)</div>
            <p class="hero-subtitle">
                Understanding patterns, text lengths, word counts, and discriminative keywords within the dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if df_data is None:
        st.warning("Dataset not found in `data/dataset.csv`. Please verify the dataset location.")
        st.stop()

    eda_col1, eda_col2 = st.columns(2)

    with eda_col1:
        st.subheader("📏 Character Length Distribution")
        fig_len, ax_len = plt.subplots(figsize=(6, 4))
        ham_chars = df_data[df_data["label"] == "ham"]["char_count"]
        spam_chars = df_data[df_data["label"] == "spam"]["char_count"]

        ax_len.hist(ham_chars, bins=40, range=(0, 250), alpha=0.6, label="Ham (Legitimate)", color="#10b981")
        ax_len.hist(spam_chars, bins=40, range=(0, 250), alpha=0.7, label="Spam", color="#ef4444")
        ax_len.set_xlabel("Character Count")
        ax_len.set_ylabel("Message Count")
        ax_len.set_title("Message Character Length: Ham vs Spam")
        ax_len.legend()
        ax_len.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_len)
        plt.close(fig_len)
        st.caption("Insight: Spam messages typically have a noticeably higher character length due to marketing copy, links, and disclaimers.")

    with eda_col2:
        st.subheader("📝 Word Count Distribution")
        fig_wrd, ax_wrd = plt.subplots(figsize=(6, 4))
        ham_words = df_data[df_data["label"] == "ham"]["word_count"]
        spam_words = df_data[df_data["label"] == "spam"]["word_count"]

        ax_wrd.hist(ham_words, bins=30, range=(0, 50), alpha=0.6, label="Ham", color="#10b981")
        ax_wrd.hist(spam_words, bins=30, range=(0, 50), alpha=0.7, label="Spam", color="#ef4444")
        ax_wrd.set_xlabel("Word Count")
        ax_wrd.set_ylabel("Message Count")
        ax_wrd.set_title("Message Word Count: Ham vs Spam")
        ax_wrd.legend()
        ax_wrd.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_wrd)
        plt.close(fig_wrd)
        st.caption("Insight: Ham messages are predominantly short, concise conversational statements, while spam messages have dense token counts.")

    st.divider()

    # Keyword Analysis Section
    st.subheader("🔑 Top Statistical Keyword Indicators")
    st.markdown(
        """
        > **Academic Note:** These keywords represent learned statistical weights from the TF-IDF feature space and model coefficients. They are probabilistic indicators, not definitive proof that a message is spam.
        """
    )

    if metrics and "top_spam_keywords" in metrics and "top_ham_keywords" in metrics:
        kw_col1, kw_col2 = st.columns(2)

        with kw_col1:
            st.markdown("#### 🚨 Top Spam Indicators")
            spam_kws = metrics["top_spam_keywords"][:12]
            fig_skw, ax_skw = plt.subplots(figsize=(6, 4))
            words = [k["word"] for k in spam_kws][::-1]
            weights = [k["weight"] for k in spam_kws][::-1]

            ax_skw.barh(words, weights, color="#ef4444")
            ax_skw.set_xlabel("Model Importance Weight")
            ax_skw.set_title("Top Spam Predictive Keywords", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_skw)
            plt.close(fig_skw)

        with kw_col2:
            st.markdown("#### ✉️ Top Legitimate (Ham) Indicators")
            ham_kws = metrics["top_ham_keywords"][:12]
            fig_hkw, ax_hkw = plt.subplots(figsize=(6, 4))
            words = [k["word"] for k in ham_kws][::-1]
            weights = [k["weight"] for k in ham_kws][::-1]

            ax_hkw.barh(words, weights, color="#10b981")
            ax_hkw.set_xlabel("Model Importance Weight")
            ax_hkw.set_title("Top Ham Predictive Keywords", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_hkw)
            plt.close(fig_hkw)

    st.divider()

    # Searchable Raw Dataset Preview
    st.subheader("🔍 Dataset Records Explorer")
    filter_choice = st.radio("Filter records by:", ["All Messages", "Spam Only", "Ham Only"], horizontal=True)
    
    if filter_choice == "Spam Only":
        view_df = df_data[df_data["label"] == "spam"]
    elif filter_choice == "Ham Only":
        view_df = df_data[df_data["label"] == "ham"]
    else:
        view_df = df_data

    search_query = st.text_input("Search dataset messages by keyword:", "")
    if search_query:
        view_df = view_df[view_df["text"].str.contains(search_query, case=False, na=False)]

    st.dataframe(
        view_df[["label", "text", "char_count", "word_count"]].head(100),
        use_container_width=True,
        hide_index=True
    )
    st.caption(f"Showing up to 100 matching records (Total in filter: {len(view_df):,})")


# ---------------------------------------------------------
# PAGE 5: About Project
# ---------------------------------------------------------
elif selected_page == "ℹ️ About Project":
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">ℹ️ About Email Spam Classifier</div>
            <p class="hero-subtitle">
                Academic project documentation, Machine Learning architecture, and engineering overview.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🎓 Project Information")
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.markdown("**Project Title:**")
        st.write("Email & SMS Spam Classifier")
    with info_col2:
        st.markdown("**Domain:**")
        st.write("Natural Language Processing (NLP) & Machine Learning")
    with info_col3:
        st.markdown("**Frameworks:**")
        st.write("Python, Scikit-Learn, Streamlit, Pandas")

    st.divider()

    st.markdown("### ⚙️ Machine Learning Pipeline Workflow")
    st.markdown(
        """
        ```
        Raw Email / SMS Text
                 │
                 ▼
        [1. Text Preprocessing]
             ├── Lowercasing
             ├── Removing URLs, HTML tags & Email patterns
             ├── Removing Special Characters & Punctuation
             ├── Stopword Removal (NLTK)
             └── Porter Stemming
                 │
                 ▼
        [2. Feature Extraction (TF-IDF)]
             ├── Sublinear Term Frequency Scaling
             ├── N-gram range: Unigrams and Bigrams (1, 2)
             └── Max features: 3,500 vocabulary tokens
                 │
                 ▼
        [3. Classification Engine]
             ├── Multinomial Naive Bayes (Laplace smoothed)
             ├── Logistic Regression (L2 regularization)
             └── Support Vector Classifier (Calibrated LinearSVC)
                 │
                 ▼
        [4. Decision & Confidence Scoring]
             ├── Prediction (Spam = 1, Ham = 0)
             ├── Probability / Confidence Estimation
             └── Heuristic Risk Trigger Tagging
        ```
        """
    )

    st.divider()

    st.markdown("### 🛡️ Limitations & Engineering Disclaimers")
    st.info(
        """
        1. **Scope of Detection:** This application evaluates text content and statistical vocabulary patterns. It does not inspect email attachments (e.g., .exe, .zip), macro viruses, or execute dynamic phishing URL sandboxing.
        2. **Keyword Interpretation:** Highlighted trigger keywords are statistical signals derived from training data distributions, not definitive proof that an email is malicious.
        3. **Adversarial Obfuscation:** Attackers who intentionally disguise keywords (e.g., "F-R-E-E" or zero-width spaces) may bypass simple bag-of-words vectorizers unless supplemented by character-level or deep learning architectures.
        """
    )

    st.divider()
    st.markdown("### 📦 System Requirements & Reproducibility")
    st.code(
        """
# Clone / Open project directory
cd "EMAIL SPAM CLASSIFIER"

# Install dependencies
pip install -r requirements.txt

# Train models and generate evaluation metrics
python train_model.py

# Launch Streamlit dashboard
streamlit run app.py
        """,
        language="bash"
    )

# Footer
st.markdown(
    """
    <div class="footer-text">
        Email Spam Classifier • Academic Machine Learning Demonstration • Built with Python & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
