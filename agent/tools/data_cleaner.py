import pandas as pd
import numpy as np
import streamlit as st

def clean_data(df):
    """
    Automatically cleans any CSV dataset
    Returns cleaned dataframe + activity log
    """
    log = []

    # ── Step 1: Basic Info ──────────────────────
    original_rows = df.shape[0]
    original_cols = df.shape[1]
    log.append(("success", f"Dataset loaded: {original_rows} rows × {original_cols} columns"))

    # ── Step 2: Remove Duplicates ───────────────
    dupes = df.duplicated().sum()
    if dupes > 0:
        df = df.drop_duplicates()
        log.append(("success", f"{dupes} duplicate rows removed"))
    else:
        log.append(("success", "No duplicate rows found"))

    # ── Step 3: Fix Column Names ────────────────
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    log.append(("success", "Column names cleaned and standardised"))

    # ── Step 4: Handle Missing Values ───────────
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]

    if len(null_cols) > 0:
        for col in null_cols.index:
            if df[col].dtype in ["float64", "int64"]:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                log.append(("success", f"Column '{col}': {null_counts[col]} nulls filled with median ({median_val:.2f})"))
            else:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                log.append(("success", f"Column '{col}': {null_counts[col]} nulls filled with mode ('{mode_val}')"))
    else:
        log.append(("success", "No missing values found"))

    # ── Step 5: Fix Data Types ──────────────────
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col])
                log.append(("success", f"Column '{col}' converted to numeric"))
            except:
                df[col] = df[col].str.strip().str.lower()

    log.append(("success", "Text columns standardised to lowercase"))

    # ── Step 6: Detect Outliers ─────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_cols = []

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)].shape[0]
        if outliers > 0:
            outlier_cols.append(col)
            df[col] = df[col].clip(lower, upper)

    if outlier_cols:
        log.append(("warning", f"Outliers detected and capped in: {', '.join(outlier_cols)}"))
    else:
        log.append(("success", "No outliers detected"))

    # ── Step 7: Final Summary ───────────────────
    final_rows = df.shape[0]
    removed = original_rows - final_rows
    log.append(("success", f"Cleaning complete! {removed} rows removed. Final dataset: {final_rows} rows × {df.shape[1]} columns"))

    return df, log