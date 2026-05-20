import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_report(stats, model_results, log_entries, output_dir="outputs"):
    """
    Uses LLM to generate a business report from analysis results
    """
    log = []
    os.makedirs(output_dir, exist_ok=True)

    log.append(("loading", "Compiling analysis findings..."))

    # Build summary to send to LLM
    shape = stats.get("shape", ("?", "?"))
    top_corr = stats.get("top_correlations", {})
    best_model = model_results.get("best_model_name", "Unknown")
    task = model_results.get("task", "classification")
    all_scores = model_results.get("all_scores", {})
    top_features = model_results.get("top_features", [])

    # Build model scores summary
    scores_text = ""
    for name, info in all_scores.items():
        if task == "classification":
            scores_text += f"\n- {name}: Accuracy {info['accuracy']}%, F1 {info['f1_score']}%"
        else:
            scores_text += f"\n- {name}: RMSE {info['rmse']}, R² {info['r2']}"

    # Build top features summary
    features_text = ""
    for f in top_features[:5]:
        features_text += f"\n- {f['Feature']}: {f['Importance']:.4f}"

    # Build correlations summary
    corr_text = ""
    for (col1, col2), val in list(top_corr.items())[:3]:
        corr_text += f"\n- {col1} ↔ {col2}: {val:.2f}"

    prompt = f"""
You are a senior data analyst writing a professional business report.
Based on the analysis below, write a clear structured report.

DATASET OVERVIEW:
- Rows: {shape[0]}, Columns: {shape[1]}

TOP CORRELATIONS:
{corr_text if corr_text else "None found"}

MODEL RESULTS:
Task Type: {task}
{scores_text}
Best Model: {best_model}

TOP FEATURES DRIVING PREDICTIONS:
{features_text if features_text else "Not available"}

Write the report with these exact sections:
1. EXECUTIVE SUMMARY (3-4 sentences overview)
2. DATASET OVERVIEW (describe the data)
3. KEY FINDINGS FROM ANALYSIS (bullet points of insights)
4. MODEL PERFORMANCE (explain results in plain English)
5. TOP FACTORS (explain what drives the predictions)
6. BUSINESS RECOMMENDATIONS (3-5 actionable recommendations)
7. LIMITATIONS & NEXT STEPS (2-3 points)

Write in clear professional English. No technical jargon.
Keep it under 600 words.
"""

    log.append(("loading", "Sending findings to AI for report generation..."))

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    report_text = response.choices[0].message.content
    log.append(("success", "Business report generated successfully"))

    # Save report as text file
    report_path = os.path.join(output_dir, "business_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    log.append(("success", "Report saved to outputs/business_report.txt"))
    log.append(("done", "All analysis complete! 🎉"))

    return report_text, report_path, log