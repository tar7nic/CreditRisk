import os
import pandas as pd
import numpy as np
from src.config import (
    RAW_TRAIN, PROCESSED_TRAIN, DATA_PROCESSED, TARGET_COL
)
from src.logger import get_logger

logger = get_logger(__name__)

ANOMALY_SENTINEL   = -999          # replacement for known anomalous values
DAYS_COLS          = [             # DAYS_* columns encoded as negatives
    "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE",
]
HIGH_MISSING_THRESH = 0.45         # drop columns missing >45 % of values
LOW_VAR_THRESH      = 0.01         # drop near-zero variance numeric cols


def load_raw(path: str = RAW_TRAIN) -> pd.DataFrame:
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Raw shape: {df.shape}  |  Target distribution:\n{df[TARGET_COL].value_counts(normalize=True).round(4)}")
    return df

def drop_high_missing(df: pd.DataFrame, thresh: float = HIGH_MISSING_THRESH) -> pd.DataFrame:
    missing_rate = df.isnull().mean()
    to_drop      = missing_rate[missing_rate > thresh].index.tolist()
    logger.info(f"Dropping {len(to_drop)} columns with >{thresh*100:.0f}% missing values")
    return df.drop(columns=to_drop)

def fix_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    # DAYS_EMPLOYED == 365243 is a data-entry sentinel for "unemployed / N/A"
    if "DAYS_EMPLOYED" in df.columns:
        anomaly_count = (df["DAYS_EMPLOYED"] == 365243).sum()
        logger.info(f"DAYS_EMPLOYED anomaly (365243): {anomaly_count} rows → replaced with NaN")
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # Convert negative DAYS_* to positive (they represent days before application)
    for col in DAYS_COLS:
        if col in df.columns:
            df[col] = df[col].abs()

    return df

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Numeric → median (robust to skew / outliers common in financial data)
    num_missing = [c for c in num_cols if df[c].isnull().any()]
    logger.info(f"Imputing {len(num_missing)} numeric columns with median")
    df[num_missing] = df[num_missing].fillna(df[num_missing].median())

    # Categorical → mode
    cat_missing = [c for c in cat_cols if df[c].isnull().any()]
    logger.info(f"Imputing {len(cat_missing)} categorical columns with mode")
    for col in cat_missing:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df     = df.drop_duplicates()
    logger.info(f"Duplicates removed: {before - len(df)}")
    return df

def drop_low_variance(df: pd.DataFrame, thresh: float = LOW_VAR_THRESH) -> pd.DataFrame:
    num_cols    = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COL in num_cols:
        num_cols.remove(TARGET_COL)

    normed_std  = df[num_cols].std() / (df[num_cols].mean().abs() + 1e-9)
    to_drop     = normed_std[normed_std < thresh].index.tolist()
    logger.info(f"Dropping {len(to_drop)} near-zero variance columns")
    return df.drop(columns=to_drop)

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    logger.info(f"Label-encoding {len(cat_cols)} categorical columns")
    for col in cat_cols:
        df[col] = pd.Categorical(df[col]).codes
    return df

def clip_outliers(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COL in num_cols:
        num_cols.remove(TARGET_COL)

    logger.info(f"Winsorizing {len(num_cols)} numeric columns at [{lower},{upper}] quantiles")
    for col in num_cols:
        lo = df[col].quantile(lower)
        hi = df[col].quantile(upper)
        df[col] = df[col].clip(lo, hi)
    return df

def run_cleaning_pipeline(save: bool = True) -> pd.DataFrame:
    os.makedirs(DATA_PROCESSED, exist_ok=True)

    df = load_raw()
    df = drop_high_missing(df)
    df = fix_anomalies(df)
    df = remove_duplicates(df)
    df = impute_missing(df)
    df = drop_low_variance(df)
    df = clip_outliers(df)
    df = encode_categoricals(df)

    logger.info(f"Cleaned shape: {df.shape}")

    if save:
        df.to_csv(PROCESSED_TRAIN, index=False)
        logger.info(f"Cleaned data saved → {PROCESSED_TRAIN}")

    return df


if __name__ == "__main__":
    run_cleaning_pipeline()