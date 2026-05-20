import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)

def detect_task_type(df, target_col):
    """
    Automatically detects if task is classification or regression
    """
    unique_vals = df[target_col].nunique()
    dtype = df[target_col].dtype

    if dtype == "object" or unique_vals <= 10:
        return "classification"
    else:
        return "regression"

def encode_features(df):
    """
    Encodes categorical columns to numbers
    """
    df = df.copy()
    encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders

def build_model(df, target_col, output_dir="outputs"):
    """
    Automatically builds and evaluates ML models
    Returns best model + results + log
    """
    log = []
    results = {}
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Detect Task ──────────────────────
    task = detect_task_type(df, target_col)
    log.append(("success", f"Task detected: {task.upper()}"))
    log.append(("success", f"Target column: '{target_col}'"))

    # ── Step 2: Prepare Data ─────────────────────
    log.append(("loading", "Encoding categorical features..."))
    df_encoded, encoders = encode_features(df)

    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    log.append(("success", f"Data split — Train: {len(X_train)} rows, Test: {len(X_test)} rows"))

    # ── Step 3: Train Models ─────────────────────
    if task == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree":       DecisionTreeClassifier(random_state=42),
            "Random Forest":       RandomForestClassifier(random_state=42),
            "Gradient Boosting":   GradientBoostingClassifier(random_state=42)
        }
    else:
        models = {
            "Linear Regression":      LinearRegression(),
            "Decision Tree":          DecisionTreeRegressor(random_state=42),
            "Random Forest":          RandomForestRegressor(random_state=42),
            "Gradient Boosting":      GradientBoostingRegressor(random_state=42)
        }

    model_scores = {}

    for name, model in models.items():
        log.append(("loading", f"Training {name}..."))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if task == "classification":
            score = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            model_scores[name] = {
                "model": model,
                "accuracy": round(score * 100, 2),
                "f1_score": round(f1 * 100, 2),
                "predictions": y_pred
            }
            log.append(("success", f"{name} — Accuracy: {score*100:.1f}% | F1: {f1*100:.1f}%"))
        else:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            model_scores[name] = {
                "model": model,
                "rmse": round(rmse, 2),
                "r2": round(r2, 4),
                "mae": round(mae, 2),
                "predictions": y_pred
            }
            log.append(("success", f"{name} — RMSE: {rmse:.2f} | R²: {r2:.4f}"))

    # ── Step 4: Pick Best Model ──────────────────
    if task == "classification":
        best_name = max(model_scores, key=lambda x: model_scores[x]["accuracy"])
        best_score = model_scores[best_name]["accuracy"]
        log.append(("done", f"Best model: {best_name} (Accuracy: {best_score}%)"))
    else:
        best_name = min(model_scores, key=lambda x: model_scores[x]["rmse"])
        best_score = model_scores[best_name]["rmse"]
        log.append(("done", f"Best model: {best_name} (RMSE: {best_score})"))

    best_model = model_scores[best_name]["model"]
    y_pred_best = model_scores[best_name]["predictions"]

    # ── Step 5: Confusion Matrix ─────────────────
    if task == "classification":
        log.append(("loading", "Generating confusion matrix..."))
        cm = confusion_matrix(y_test, y_pred_best)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(f"Confusion Matrix — {best_name}")
        path = os.path.join(output_dir, "confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        log.append(("success", "Confusion matrix saved"))
        results["confusion_matrix"] = path

    # ── Step 6: Feature Importance ───────────────
    log.append(("loading", "Calculating feature importance..."))

    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(feat_df["Feature"], feat_df["Importance"], color="steelblue")
        ax.set_xlabel("Importance Score")
        ax.set_title(f"Top 10 Feature Importances — {best_name}")
        ax.invert_yaxis()
        plt.tight_layout()
        path = os.path.join(output_dir, "feature_importance.png")
        plt.savefig(path)
        plt.close()
        log.append(("success", f"Top features: {', '.join(feat_df['Feature'].head(3).tolist())}"))
        results["feature_importance"] = path
        results["top_features"] = feat_df.to_dict("records")

    # ── Step 7: Save Best Model ──────────────────
    model_path = os.path.join(output_dir, "best_model.pkl")
    joblib.dump(best_model, model_path)
    log.append(("success", f"Best model saved to {model_path}"))

    # ── Step 8: Compile Results ──────────────────
    results["task"] = task
    results["best_model_name"] = best_name
    results["all_scores"] = model_scores
    results["X_columns"] = X.columns.tolist()
    results["y_test"] = y_test
    results["y_pred"] = y_pred_best

    if task == "classification":
        results["classification_report"] = classification_report(
            y_test, y_pred_best, output_dict=True
        )

    return best_model, results, log