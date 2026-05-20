import pandas as pd
from agent.tools.data_cleaner import clean_data
from agent.tools.eda_tool import run_eda
from agent.tools.model_builder import build_model
from agent.tools.report_writer import generate_report
from agent.tools.shap_explainer import run_shap
from agent.tools.anomaly_detector import detect_anomalies

def run_agent(df, target_col):
    """
    Master agent that runs all tools in sequence
    """
    full_log = []
    all_results = {}

    # ── Stage 1: Data Cleaning ───────────────────
    full_log.append(("loading", "═══ STAGE 1: DATA CLEANING ═══"))
    cleaned_df, clean_log = clean_data(df)
    full_log.extend(clean_log)
    all_results["cleaned_df"] = cleaned_df

    # ── Stage 2: EDA ─────────────────────────────
    full_log.append(("loading", "═══ STAGE 2: EXPLORATORY DATA ANALYSIS ═══"))
    stats, charts, eda_log = run_eda(cleaned_df)
    full_log.extend(eda_log)
    all_results["stats"] = stats
    all_results["charts"] = charts

    # ── Stage 3: Anomaly Detection ───────────────
    full_log.append(("loading", "═══ STAGE 3: ANOMALY DETECTION ═══"))
    anomaly_results, anomaly_log = detect_anomalies(cleaned_df)
    full_log.extend(anomaly_log)
    all_results["anomaly_results"] = anomaly_results

    # ── Stage 4: Model Building ──────────────────
    full_log.append(("loading", "═══ STAGE 4: MODEL BUILDING ═══"))
    best_model, model_results, model_log = build_model(cleaned_df, target_col)
    full_log.extend(model_log)
    all_results["model_results"] = model_results

    # ── Stage 5: SHAP Explainability ─────────────
    full_log.append(("loading", "═══ STAGE 5: SHAP EXPLAINABILITY ═══"))
    from sklearn.preprocessing import LabelEncoder
    import numpy as np

    df_encoded = cleaned_df.copy()
    for col in df_encoded.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    from sklearn.model_selection import train_test_split
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    shap_charts, shap_log = run_shap(best_model, X_train, X_test, X.columns.tolist())
    full_log.extend(shap_log)
    all_results["shap_charts"] = shap_charts

    # ── Stage 6: Report Writing ──────────────────
    full_log.append(("loading", "═══ STAGE 6: GENERATING AI REPORT ═══"))
    report_text, report_path, report_log = generate_report(
        stats, model_results, full_log
    )
    full_log.extend(report_log)
    all_results["report_text"] = report_text
    all_results["report_path"] = report_path

    return all_results, full_log