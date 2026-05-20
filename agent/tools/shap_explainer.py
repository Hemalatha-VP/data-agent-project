import shap
import numpy as np
import matplotlib.pyplot as plt
import os

def run_shap(model, X_train, X_test, feature_names, output_dir="outputs"):
    """
    Runs SHAP explainability on the best model
    Returns chart paths + log
    """
    log = []
    charts = {}
    os.makedirs(output_dir, exist_ok=True)

    try:
        log.append(("loading", "Running SHAP explainability analysis..."))

        # Use TreeExplainer for tree models, LinearExplainer for others
        model_type = type(model).__name__

        if model_type in ["RandomForestClassifier", "RandomForestRegressor",
                          "GradientBoostingClassifier", "GradientBoostingRegressor",
                          "DecisionTreeClassifier", "DecisionTreeRegressor"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            log.append(("success", f"SHAP TreeExplainer used for {model_type}"))
        else:
            explainer = shap.LinearExplainer(model, X_train)
            shap_values = explainer.shap_values(X_test)
            log.append(("success", f"SHAP LinearExplainer used for {model_type}"))

        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values

        # ── Chart 1: Summary Bar Plot ────────────
        log.append(("loading", "Generating SHAP summary chart..."))
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_vals,
            X_test,
            feature_names=feature_names,
            plot_type="bar",
            show=False
        )
        plt.title("SHAP Feature Importance — Impact on Model Output")
        plt.tight_layout()
        path1 = os.path.join(output_dir, "shap_summary.png")
        plt.savefig(path1, bbox_inches="tight", dpi=150)
        plt.close()
        charts["shap_summary"] = path1
        log.append(("success", "SHAP summary chart saved"))

        # ── Chart 2: Beeswarm Plot ───────────────
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_vals,
            X_test,
            feature_names=feature_names,
            show=False
        )
        plt.title("SHAP Beeswarm — How Each Feature Affects Predictions")
        plt.tight_layout()
        path2 = os.path.join(output_dir, "shap_beeswarm.png")
        plt.savefig(path2, bbox_inches="tight", dpi=150)
        plt.close()
        charts["shap_beeswarm"] = path2
        log.append(("success", "SHAP beeswarm chart saved"))

        # Top SHAP features
        mean_shap = np.abs(shap_vals).mean(axis=0)
        top_indices = mean_shap.argsort()[::-1][:5]
        top_shap_features = [(feature_names[i], round(float(mean_shap[i]), 4)) for i in top_indices]
        log.append(("success", f"Top SHAP factor: {top_shap_features[0][0]} (impact: {top_shap_features[0][1]})"))

        charts["top_shap_features"] = top_shap_features
        log.append(("done", "SHAP explainability complete"))

    except Exception as e:
        log.append(("warning", f"SHAP skipped: {str(e)}"))

    return charts, log