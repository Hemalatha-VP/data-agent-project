import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(df, output_dir="outputs"):
    """
    Automatically runs full EDA on any cleaned dataframe
    Returns stats dictionary + list of chart file paths + log
    """
    log = []
    charts = []
    stats = {}

    # Create outputs folder if not exists
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Basic Statistics ─────────────────
    log.append(("loading", "Calculating descriptive statistics..."))
    stats["shape"] = df.shape
    stats["dtypes"] = df.dtypes.astype(str).to_dict()
    stats["describe"] = df.describe().round(2).to_dict()
    stats["null_counts"] = df.isnull().sum().to_dict()
    log.append(("success", f"Statistics calculated for {df.shape[1]} columns"))

    # ── Step 2: Correlation Matrix ───────────────
    log.append(("loading", "Generating correlation heatmap..."))
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax
        )
        ax.set_title("Correlation Heatmap")
        path = os.path.join(output_dir, "correlation_heatmap.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        charts.append(("Correlation Heatmap", path))
        log.append(("success", "Correlation heatmap saved"))

        # Find top correlations
        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        top_corr = upper.stack().sort_values(ascending=False).head(3)
        for (col1, col2), val in top_corr.items():
            log.append(("success", f"Strong correlation: {col1} ↔ {col2} ({val:.2f})"))
        stats["top_correlations"] = top_corr.to_dict()

    # ── Step 3: Histograms ───────────────────────
    log.append(("loading", "Generating histograms..."))
    numeric_cols = numeric_df.columns.tolist()

    if numeric_cols:
        fig, axes = plt.subplots(
            nrows=(len(numeric_cols) + 2) // 3,
            ncols=min(3, len(numeric_cols)),
            figsize=(15, 4 * ((len(numeric_cols) + 2) // 3))
        )
        axes = np.array(axes).flatten()

        for i, col in enumerate(numeric_cols):
            axes[i].hist(df[col].dropna(), bins=30, color="steelblue", edgecolor="white")
            axes[i].set_title(f"{col}")
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Count")

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Histograms — Numeric Columns", fontsize=14, y=1.02)
        plt.tight_layout()
        path = os.path.join(output_dir, "histograms.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        charts.append(("Histograms", path))
        log.append(("success", f"Histograms generated for {len(numeric_cols)} numeric columns"))

    # ── Step 4: Box Plots ────────────────────────
    log.append(("loading", "Generating box plots for outlier detection..."))

    if numeric_cols:
        fig, axes = plt.subplots(
            nrows=(len(numeric_cols) + 2) // 3,
            ncols=min(3, len(numeric_cols)),
            figsize=(15, 4 * ((len(numeric_cols) + 2) // 3))
        )
        axes = np.array(axes).flatten()

        for i, col in enumerate(numeric_cols):
            axes[i].boxplot(df[col].dropna())
            axes[i].set_title(f"{col}")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Box Plots — Outlier Detection", fontsize=14, y=1.02)
        plt.tight_layout()
        path = os.path.join(output_dir, "boxplots.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        charts.append(("Box Plots", path))
        log.append(("success", "Box plots generated"))

    # ── Step 5: Bar Charts (Categorical) ─────────
    log.append(("loading", "Generating bar charts for categorical columns..."))
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in cat_cols[:4]:  # max 4 categorical charts
        fig, ax = plt.subplots(figsize=(8, 4))
        df[col].value_counts().head(10).plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title(f"{col} — Value Counts")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        plt.tight_layout()
        path = os.path.join(output_dir, f"barplot_{col}.png")
        plt.savefig(path)
        plt.close()
        charts.append((f"Bar Chart: {col}", path))

    if cat_cols:
        log.append(("success", f"Bar charts generated for {min(4, len(cat_cols))} categorical columns"))

    # ── Step 6: Done ─────────────────────────────
    log.append(("done", f"EDA complete! {len(charts)} charts generated"))
    stats["chart_count"] = len(charts)

    return stats, charts, log