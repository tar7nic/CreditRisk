import os
import json
import joblib
import pandas as pd
import numpy as np
from src.config import (
    PROCESSED_FINAL, MODELS_DIR, OUTPUTS_DIR,
    FIGURES_DIR, TARGET_COL, DATA_PROCESSED
)


@st.cache_data if False else lambda f: f
def _dummy(): pass


def load_cleaned_data() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_FINAL)


def load_model_metrics() -> list:
    path = os.path.join(OUTPUTS_DIR, "model_metrics.json")
    with open(path) as f:
        return json.load(f)


def load_best_model():
    with open(os.path.join(MODELS_DIR, "best_model.txt")) as f:
        name = f.read().strip()
    pipeline = joblib.load(os.path.join(MODELS_DIR, f"{name}.joblib"))
    return name, pipeline


def load_shap_importance() -> pd.DataFrame:
    return pd.read_csv(os.path.join(OUTPUTS_DIR, "shap_importance.csv"))


def load_sql_cohorts() -> dict:
    cohorts = {}
    files = {
        "overall"       : "sql_overall_default_rate.csv",
        "gender"        : "sql_default_by_gender.csv",
        "education"     : "sql_default_by_education.csv",
        "income"        : "sql_income_cohorts.csv",
        "age"           : "sql_age_cohorts.csv",
        "credit_stress" : "sql_credit_stress_ratio.csv",
    }
    for key, fname in files.items():
        path = os.path.join(DATA_PROCESSED, fname)
        if os.path.exists(path):
            cohorts[key] = pd.read_csv(path)
    return cohorts


def get_feature_columns() -> list:
    df = pd.read_csv(PROCESSED_FINAL)
    drop = [TARGET_COL, "SK_ID_CURR"] if "SK_ID_CURR" in df.columns else [TARGET_COL]
    return df.drop(columns=drop).select_dtypes(include=[np.number]).columns.tolist()