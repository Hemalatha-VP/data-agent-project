import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import IsolationForest

def detect_anomalies(df, output_dir="outputs"):
    """
    Detects anomalies using Isolation Forest
    Returns results + charts + log
    """
    log = []
    results = {}
    os.makedirs(output_dir, exist_ok=True)

    log.append(("loading", "Running anomaly detection..."))

    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        log.append(("warning", "Not enough numeric columns for anomaly detection"))
        return results, log

    # Run Isolation Forest
    model = IsolationForest(contamination=0.05, random_state=42)
    preds = model.fit_predict(numeric_df)
    scores = model.decision_function(numeric_df)

    df_result = df.copy()
    df_result["anomaly"] = preds
    df_result["anomaly_score"] = scores

    anomalies = df_result[df_result["anomaly"] == -1]
    normal = df_result[df_result["anomaly"] == 1]

    pct = round(len(anomalies) / len(df) * 100, 2)
    log.append(("success", f"Anomaly detection complete: {len(anomalies)} anomalies found ({pct}% of data)"))

    results["total_anomalies"] = len(anomalies)
    results["anomaly_percent"] = pct
    results["anomaly_df"] = anomalies

    # ── Chart: Anomaly Score Distribution ───────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Score distribution
    axes[0].hist(scores[preds == 1], bins=30, alpha=0.7,
                 color="#4ade80", label="Normal", edgecolor="white")
    axes[0].hist(scores[preds == -1], bins=30, alpha=0.7,
                 color="#f87171", label="Anomaly", edgecolor="white")
    axes[0].set_title("Anomaly Score Distribution")
    axes[0].set_xlabel("Anomaly Score (lower = more anomalous)")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].axvline(x=0, color="white", linestyle="--", alpha=0.5)

    # Scatter of first two numeric columns
    cols = numeric_df.columns[:2].tolist()
    axes[1].scatter(normal[cols[0]], normal[cols[1]],
                    c="#4ade80", alpha=0.4, s=10, label="Normal")
    axes[1].scatter(anomalies[cols[0]], anomalies[cols[1]],
                    c="#f87171", alpha=0.8, s=30, label="Anomaly", marker="x")
    axes[1].set_title(f"Anomalies: {cols[0]} vs {cols[1]}")
    axes[1].set_xlabel(cols[0])
    axes[1].set_ylabel(cols[1])
    axes[1].legend()

    plt.suptitle("Anomaly Detection — Isolation Forest", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "anomaly_detection.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()

    results["chart"] = path
    log.append(("success", "Anomaly detection chart saved"))
    log.append(("done", f"Found {len(anomalies)} unusual records worth investigating"))

    return results, log