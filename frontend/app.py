import streamlit as st
import pandas as pd
import sys, os
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.core_agent import run_agent

st.set_page_config(
    page_title="DataAgent AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 16px 0 24px;
}
.kpi-card {
    background: var(--background-color, #f8f9fa);
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 12px;
    padding: 16px 18px;
    text-align: center;
}
.kpi-val {
    font-size: 28px;
    font-weight: 700;
    color: #6366f1;
    line-height: 1.2;
}
.kpi-label {
    font-size: 12px;
    color: #888;
    margin-top: 4px;
}
.kpi-sub {
    font-size: 11px;
    color: #aaa;
    margin-top: 2px;
}
.tip-box {
    background: rgba(99, 102, 241, 0.08);
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: 13px;
    color: var(--text-color, inherit);
    margin: 10px 0;
    line-height: 1.6;
}
.warn-box {
    background: rgba(251, 191, 36, 0.1);
    border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: 13px;
    color: var(--text-color, inherit);
    margin: 10px 0;
}
.ok-box {
    background: rgba(34, 197, 94, 0.08);
    border-left: 3px solid #22c55e;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: 13px;
    color: var(--text-color, inherit);
    margin: 10px 0;
}
.err-box {
    background: rgba(239, 68, 68, 0.08);
    border-left: 3px solid #ef4444;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: 13px;
    color: var(--text-color, inherit);
    margin: 10px 0;
}
.log-container {
    background: rgba(0,0,0,0.04);
    border: 1px solid rgba(128,128,128,0.15);
    border-radius: 10px;
    padding: 12px 14px;
    max-height: 520px;
    overflow-y: auto;
    font-family: monospace;
    font-size: 12px;
    line-height: 1.8;
}
.log-stage { color: #6366f1; font-weight: 600; margin-top: 6px; }
.log-ok    { color: #22c55e; }
.log-spin  { color: #a78bfa; }
.log-warn  { color: #f59e0b; }
.log-done  { color: #38bdf8; font-weight: 600; }
.log-err   { color: #ef4444; }
section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}
.step-badge {
    display: inline-block;
    background: #6366f1;
    color: white;
    font-size: 11px;
    font-weight: 700;
    width: 22px;
    height: 22px;
    line-height: 22px;
    text-align: center;
    border-radius: 50%;
    margin-right: 8px;
}
</style>
""", unsafe_allow_html=True)


def get_groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        from dotenv import load_dotenv
        from pathlib import Path
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        return os.getenv("GROQ_API_KEY")


# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🤖 DataAgent AI")
    st.caption("Autonomous Data Analysis")
    st.divider()

    st.markdown('<span class="step-badge">1</span> **Upload your CSV file**', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        df_preview = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)
        st.success(f"✅ Loaded: {df_preview.shape[0]:,} rows × {df_preview.shape[1]} columns")

        st.markdown("")
        st.markdown('<span class="step-badge">2</span> **What do you want to predict?**', unsafe_allow_html=True)
        st.caption("This is the column your ML model will learn to predict. For the Telco dataset, pick **Churn**.")

        target_col = st.selectbox(
            "Target column",
            options=df_preview.columns.tolist(),
            label_visibility="collapsed"
        )
        st.caption(f"Selected: `{target_col}` — the agent will treat this as the output variable.")

        st.markdown("")
        st.markdown('<span class="step-badge">3</span> **Run the full analysis**', unsafe_allow_html=True)
        st.caption("The agent will run 6 stages automatically. This takes about 30–60 seconds.")
        run_button = st.button("🚀  Run Full Analysis", type="primary", use_container_width=True)

        st.divider()
        st.markdown("**Dataset preview (first 5 rows)**")
        st.dataframe(df_preview.head(5), use_container_width=True, height=160)

    else:
        run_button = False
        target_col = None
        # Clear saved results when file is removed
        for k in ["agent_results", "agent_log", "agent_df", "agent_target"]:
            st.session_state.pop(k, None)
        st.info("👈 Upload a CSV file above to get started.")
        st.markdown("""
**Recommended datasets (free on Kaggle):**
- Telco Customer Churn ← start here
- House Prices (regression)
- HR Analytics / Attrition
- Titanic Survival
        """)


# ═══════════════════════════════════════════════════════
# WELCOME SCREEN
# ═══════════════════════════════════════════════════════
if not uploaded_file and "agent_results" not in st.session_state:
    st.markdown("# 🤖 Autonomous Data Analysis Agent")
    st.markdown("Upload any CSV dataset and this AI agent will automatically clean it, analyse it, build the best ML model, explain its decisions, and write a business report — **with zero manual steps.**")
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Stage 1–2: Clean + EDA")
        st.markdown("""
Removes duplicates, fills missing values,
fixes data types, caps outliers.

Then generates correlation heatmaps,
histograms, box plots and bar charts.
        """)
    with col2:
        st.markdown("### Stage 3–4: Anomalies + Models")
        st.markdown("""
Uses Isolation Forest to find unusual
records in your data before modelling.

Then trains 4 ML models, compares
them and picks the best one.
        """)
    with col3:
        st.markdown("### Stage 5–6: Explain + Report")
        st.markdown("""
SHAP analysis explains *why* the model
made each prediction.

Groq LLM writes a full business
report with insights and recommendations.
        """)

    st.divider()
    st.markdown("**👈 Upload a CSV file from the sidebar to begin.**")


# ═══════════════════════════════════════════════════════
# RUN ANALYSIS — store results in session_state
# ═══════════════════════════════════════════════════════
if uploaded_file and run_button:
    df = pd.read_csv(uploaded_file)

    col_main, col_log = st.columns([2.2, 1])

    with col_log:
        st.markdown("### 🧾 Activity Log")
        st.caption("Live record of every step the agent takes.")
        log_placeholder = st.empty()

    with col_main:
        progress_bar = st.progress(0)
        status_msg   = st.empty()

    def render_log(entries):
        html = '<div class="log-container">'
        for s, m in entries:
            if "═══" in m:
                html += f'<div class="log-stage">▶ {m}</div>'
            elif s == "success":
                html += f'<div class="log-ok">✓ {m}</div>'
            elif s == "loading":
                html += f'<div class="log-spin">⟳ {m}</div>'
            elif s == "warning":
                html += f'<div class="log-warn">⚠ {m}</div>'
            elif s == "error":
                html += f'<div class="log-err">✗ {m}</div>'
            elif s == "done":
                html += f'<div class="log-done">★ {m}</div>'
        html += '</div>'
        log_placeholder.markdown(html, unsafe_allow_html=True)

    with st.spinner("Agent is running all 6 stages..."):
        status_msg.info("⟳ Starting agent pipeline — please wait...")
        progress_bar.progress(10)

        try:
            results, full_log = run_agent(df, target_col)
            progress_bar.progress(100)
            status_msg.success("✅ All 6 stages complete! Scroll down to see results.")
            render_log(full_log)

            # ── SAVE TO SESSION STATE ──
            st.session_state["agent_results"] = results
            st.session_state["agent_log"]     = full_log
            st.session_state["agent_df"]      = df
            st.session_state["agent_target"]  = target_col

        except Exception as e:
            progress_bar.progress(0)
            status_msg.error(f"❌ Error: {str(e)}")
            st.markdown(f'<div class="err-box"><strong>Error details:</strong> {str(e)}<br><br>Common fixes:<br>• Make sure the target column exists in your CSV<br>• Check your Groq API key<br>• Make sure your venv is active</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# DISPLAY RESULTS — from session_state (survives reruns)
# ═══════════════════════════════════════════════════════
if "agent_results" in st.session_state:
    results    = st.session_state["agent_results"]
    full_log   = st.session_state["agent_log"]
    df         = st.session_state["agent_df"]
    target_col = st.session_state["agent_target"]

    model_results = results.get("model_results", {})
    all_scores    = model_results.get("all_scores", {})
    task          = model_results.get("task", "classification")
    best_name     = model_results.get("best_model_name", "N/A")
    charts        = results.get("charts", [])
    anomaly       = results.get("anomaly_results", {})
    shap_charts   = results.get("shap_charts", {})

    if task == "classification":
        best_score_val = str(all_scores.get(best_name, {}).get("accuracy", "N/A")) + "%"
    else:
        best_score_val = "R²=" + str(all_scores.get(best_name, {}).get("r2", "N/A"))

    anom_total = anomaly.get("total_anomalies", 0)
    anom_pct   = anomaly.get("anomaly_percent", 0)

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-val">{df.shape[0]:,}</div>
        <div class="kpi-label">Rows analysed</div>
        <div class="kpi-sub">{df.shape[1]} columns</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val">{best_score_val}</div>
        <div class="kpi-label">Best model score</div>
        <div class="kpi-sub">{best_name.split()[0] if best_name != "N/A" else "—"}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val">{anom_total:,}</div>
        <div class="kpi-label">Anomalies flagged</div>
        <div class="kpi-sub">{anom_pct}% of data</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val">{len(charts)}</div>
        <div class="kpi-label">Charts generated</div>
        <div class="kpi-sub">Auto-created</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 EDA Charts",
        "🔍 Anomaly Detection",
        "🤖 ML Models",
        "🧠 SHAP Explainability",
        "📝 Business Report",
        "💬 Chat with Data"
    ])

    # ── TAB 1: EDA ──────────────────────────────
    with tab1:
        st.markdown("### Exploratory Data Analysis")
        st.markdown("The agent automatically generated these charts to help you understand the shape, distribution and relationships inside your dataset.")

        hints = {
            "correlation": "**What this shows:** How strongly each pair of columns relates. Values close to +1 = strong positive link. Values close to -1 = strong negative link. Values near 0 = no relationship.",
            "histogram":   "**What this shows:** The distribution of values in each column. A tall bar in the middle = most values are average. A skewed shape = values cluster at one end.",
            "box":         "**What this shows:** The spread and outliers of each column. The box = the middle 50% of data. The line inside = the median. Dots beyond the whiskers = outliers.",
            "bar":         "**What this shows:** How often each category appears. The taller the bar, the more records have that value."
        }

        useful_charts = [(n, p) for n, p in charts
                         if "customerid" not in n.lower()
                         and os.path.exists(p)]

        if not useful_charts:
            st.warning("No charts were generated. Check that your dataset has numeric columns.")
        else:
            for name, path in useful_charts:
                st.markdown(f"#### {name}")
                for k, hint in hints.items():
                    if k in name.lower():
                        st.markdown(f'<div class="tip-box">{hint}</div>', unsafe_allow_html=True)
                st.image(Image.open(path), use_column_width=True)
                st.divider()

    # ── TAB 2: ANOMALIES ────────────────────────
    with tab2:
        st.markdown("### Anomaly Detection")
        st.markdown("The agent used **Isolation Forest** to scan your dataset for unusual records — rows that look very different from the rest.")

        a1, a2, a3 = st.columns(3)
        a1.metric("Anomalies Found", anom_total)
        a2.metric("% of Dataset", f"{anom_pct}%")
        a3.metric("Normal Records", df.shape[0] - anom_total)

        st.markdown(f'<div class="warn-box">⚠️ <strong>{anom_total} records</strong> ({anom_pct}% of your data) were flagged as anomalies.</div>', unsafe_allow_html=True)

        if "chart" in anomaly and os.path.exists(anomaly["chart"]):
            st.markdown("#### Anomaly Chart")
            st.markdown('<div class="tip-box"><strong>Left chart:</strong> Score distribution — red bars are anomalies, green are normal. <strong>Right chart:</strong> Scatter plot showing where anomalies sit in your data.</div>', unsafe_allow_html=True)
            st.image(Image.open(anomaly["chart"]), use_column_width=True)

        if "anomaly_df" in anomaly and not anomaly["anomaly_df"].empty:
            st.markdown("#### Sample Anomalous Records")
            st.markdown('<div class="warn-box">These are the actual rows flagged. Look for values that seem extreme or unusual.</div>', unsafe_allow_html=True)
            display_df = anomaly["anomaly_df"].drop(columns=["anomaly", "anomaly_score"], errors="ignore").head(10)
            st.dataframe(display_df, use_container_width=True)

    # ── TAB 3: MODELS ───────────────────────────
    with tab3:
        st.markdown("### Machine Learning Model Results")
        st.markdown(f"""
The agent trained **4 different ML models** on your data and compared their performance.

- **Task type detected:** {task.upper()}
- **Target column:** `{target_col}` — this is what the model is predicting
- **Best model:** {best_name}
        """)

        st.markdown(f'<div class="ok-box">✅ Best model: <strong>{best_name}</strong> with score <strong>{best_score_val}</strong>.</div>', unsafe_allow_html=True)

        st.markdown("#### All Model Scores")
        for name, info in all_scores.items():
            is_best = (name == best_name)
            label = f"⭐ {name}  ← BEST" if is_best else name
            with st.expander(label, expanded=is_best):
                if task == "classification":
                    c1, c2 = st.columns(2)
                    c1.metric("Accuracy", f"{info['accuracy']}%")
                    c2.metric("F1 Score", f"{info['f1_score']}%")
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("RMSE", info["rmse"])
                    c2.metric("R²", info["r2"])
                    c3.metric("MAE", info["mae"])

        if "confusion_matrix" in model_results and os.path.exists(model_results["confusion_matrix"]):
            st.markdown("#### Confusion Matrix")
            st.markdown('<div class="tip-box"><strong>How to read this:</strong> Numbers on the diagonal = correct predictions. Off-diagonal = mistakes.</div>', unsafe_allow_html=True)
            st.image(Image.open(model_results["confusion_matrix"]), use_column_width=True)

        if "feature_importance" in model_results and os.path.exists(model_results["feature_importance"]):
            st.markdown("#### Feature Importance Chart")
            st.image(Image.open(model_results["feature_importance"]), use_column_width=True)

    # ── TAB 4: SHAP ─────────────────────────────
    with tab4:
        st.markdown("### SHAP Explainability")
        st.markdown("SHAP explains **why** your model made each prediction — not just which features matter, but in which direction they push the result.")

        if "shap_summary" in shap_charts and os.path.exists(shap_charts["shap_summary"]):
            st.markdown("#### SHAP Summary — Average Feature Impact")
            st.image(Image.open(shap_charts["shap_summary"]), use_column_width=True)

        if "shap_beeswarm" in shap_charts and os.path.exists(shap_charts["shap_beeswarm"]):
            st.markdown("#### SHAP Beeswarm — Direction of Impact")
            st.markdown('<div class="tip-box">Each dot = one record. Red = high feature value. Blue = low value. Right = increased prediction. Left = decreased it.</div>', unsafe_allow_html=True)
            st.image(Image.open(shap_charts["shap_beeswarm"]), use_column_width=True)

        top = shap_charts.get("top_shap_features", [])
        if top:
            st.markdown("#### Top SHAP Factors Ranked")
            for i, (feat, val) in enumerate(top, 1):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"**{i}. {feat}**")
                    st.progress(min(float(val) * 3, 1.0))
                with col_b:
                    st.markdown(f"`{val}`")

    # ── TAB 5: REPORT ───────────────────────────
    with tab5:
        st.markdown("### AI-Generated Business Report")
        st.markdown("Written by **Groq LLM (LLaMA 3)** based on all findings from the 6-stage pipeline.")

        report = results.get("report_text", "")
        if report:
            st.markdown(report)
            st.divider()
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button(
                    label="📥  Download Report (.txt)",
                    data=report,
                    file_name="business_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col_d2:
                log_txt = "\n".join([f"[{s.upper()}] {m}" for s, m in full_log])
                st.download_button(
                    label="📥  Download Activity Log (.txt)",
                    data=log_txt,
                    file_name="activity_log.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        else:
            st.warning("Report could not be generated. Check your Groq API key.")

    # ── TAB 6: CHAT ─────────────────────────────
    with tab6:
        st.markdown("### 💬 Chat with Your Data")
        st.markdown("""
Ask plain English questions about your dataset. The AI answers using results from all 6 analysis stages.

**Example questions:**
- *Which model performed best and why?*
- *What were the most important features for predicting churn?*
- *How many anomalies were found and what could cause them?*
- *What business actions would you recommend?*
        """)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask anything about your data..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    from groq import Groq

                    groq_key = get_groq_key()

                    if not groq_key:
                        st.error("Groq API key not found. Check your Streamlit secrets.")
                    else:
                        chat_client = Groq(api_key=groq_key)

                        top_feats = ""
                        top_shap = shap_charts.get("top_shap_features", [])
                        if top_shap:
                            top_feats = ", ".join([f[0] for f in top_shap[:3]])

                        scores_summary = ", ".join([
                            f"{n}: {v.get('accuracy', v.get('r2', '?'))}"
                            for n, v in all_scores.items()
                        ])

                        context = f"""You are a helpful data analyst assistant.
Here is what the analysis found about this dataset:

Dataset: {df.shape[0]:,} rows x {df.shape[1]} columns
Target column: {target_col}
Task type: {task}
All model scores: {scores_summary}
Best model: {best_name} with score {best_score_val}
Anomalies found: {anom_total} records ({anom_pct}% of data)
Top features driving predictions: {top_feats if top_feats else "not available"}

Answer the user's question clearly and helpfully in plain English.
If asked something not in the analysis results, say so honestly."""

                        response = chat_client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[
                                {"role": "system", "content": context},
                                *[{"role": m["role"], "content": m["content"]}
                                  for m in st.session_state.messages]
                            ]
                        )
                        reply = response.choices[0].message.content
                        st.markdown(reply)
                        st.session_state.messages.append({"role": "assistant", "content": reply})

                except Exception as chat_err:
                    st.error(f"Chat error: {str(chat_err)}")