import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from src.config import PROCESSED_FINAL, MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR, TARGET_COL
from src.logger import get_logger

logger = get_logger(__name__)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def load_best_model():
    best_name_path = os.path.join(MODELS_DIR, "best_model.txt")
    with open(best_name_path) as f:
        best_name = f.read().strip()

    model_path = os.path.join(MODELS_DIR, f"{best_name}.joblib")
    saved      = joblib.load(model_path)

    # Handle both old format (pipeline directly) and new format (dict)
    if isinstance(saved, dict):
        pipeline  = saved["pipeline"]
        threshold = saved.get("threshold", 0.5)
    else:
        pipeline  = saved
        threshold = 0.5

    logger.info(f"Loaded best model: {best_name} | threshold={threshold:.4f}")
    return best_name, pipeline

def run_shap_analysis(X_sample: pd.DataFrame = None, sample_size: int = 2000) -> None:
    best_name, pipeline = load_best_model()

    if X_sample is None:
        df      = pd.read_csv(PROCESSED_FINAL)
        drop    = [TARGET_COL, "SK_ID_CURR"] if "SK_ID_CURR" in df.columns else [TARGET_COL]
        X_all   = df.drop(columns=drop).select_dtypes(include=[np.number])
        X_sample = X_all.sample(n=min(sample_size, len(X_all)), random_state=42)

    logger.info(f"Running SHAP on {len(X_sample)} samples with {best_name}...")

    # Get the actual model from pipeline if wrapped
    model = pipeline
    if hasattr(pipeline, "named_steps"):
        model = pipeline.named_steps.get("model", pipeline)

    # Use TreeExplainer for tree-based models
    if best_name in ["RandomForest", "XGBoost", "LightGBM", "DecisionTree"]:
        explainer  = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # For binary classification some models return list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        explainer   = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)

    # ── 1. Summary Plot (Beeswarm) ────────────────────────────────────────────
    plt.figure(figsize=(12, 9))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.title(f"SHAP Summary — {best_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "11_shap_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {path}")

    # ── 2. Bar Plot (Mean |SHAP|) ─────────────────────────────────────────────
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
    plt.title(f"SHAP Feature Importance — {best_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "12_shap_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {path}")

    # ── 3. Save SHAP values ───────────────────────────────────────────────────
    shap_df = pd.DataFrame(shap_values, columns=X_sample.columns)
    shap_df.to_csv(os.path.join(OUTPUTS_DIR, "shap_values.csv"), index=False)
    logger.info("SHAP values saved → outputs/shap_values.csv")

    # ── 4. Mean absolute SHAP ranking ────────────────────────────────────────
    mean_shap = pd.DataFrame({
        "feature"   : X_sample.columns,
        "mean_shap" : np.abs(shap_values).mean(axis=0)
    }).sort_values("mean_shap", ascending=False)

    mean_shap.to_csv(os.path.join(OUTPUTS_DIR, "shap_importance.csv"), index=False)
    logger.info("Top 10 SHAP features:\n" + mean_shap.head(10).to_string(index=False))


if __name__ == "__main__":
    run_shap_analysis()