# 🤖 Autonomous Data Analysis Agent

An AI-powered system that automatically analyses any CSV dataset — cleaning data, performing EDA, detecting anomalies, building ML models, explaining predictions with SHAP, and writing a business report.

Built as a final year MSc Data Science research project.

## 🔗 Live Demo
👉 [Click here to try the app](https://data-agent-project-27bxrp9ttnejpjchh46x5p.streamlit.app/)

## 🚀 What It Does
- **Stage 1 — Data Cleaning:** Removes duplicates, fills missing values, caps outliers
- **Stage 2 — EDA:** Generates correlation heatmaps, histograms, box plots automatically
- **Stage 3 — Anomaly Detection:** Flags unusual records using Isolation Forest
- **Stage 4 — ML Model Builder:** Trains 4 models, picks the best one automatically
- **Stage 5 — SHAP Explainability:** Explains why the model made each prediction
- **Stage 6 — AI Business Report:** Groq LLM writes a plain English report with recommendations

## 🛠️ Tech Stack
- Python, Pandas, Scikit-learn, SHAP
- LangChain, Groq API (LLaMA 3)
