import os
import pandas as pd
import numpy as np
from src.config import PROCESSED_TRAIN, PROCESSED_FINAL, DATA_PROCESSED, TARGET_COL
from src.logger import get_logger

logger = get_logger(__name__)
os.makedirs(DATA_PROCESSED, exist_ok=True)


# ── 1. Financial Ratio Features ───────────────────────────────────────────────
def add_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding financial ratio features...")

    # Debt-to-Income ratio
    df["DEBT_TO_INCOME"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)

    # Annuity-to-Income ratio (monthly payment burden)
    df["ANNUITY_TO_INCOME"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)

    # Credit-to-Annuity ratio (loan term proxy)
    df["CREDIT_TO_ANNUITY"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1)

    # Goods price coverage (how much of goods price is covered by credit)
    if "AMT_GOODS_PRICE" in df.columns:
        df["CREDIT_TO_GOODS"] = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1)
        df["GOODS_PRICE_DIFF"] = df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"]

    # Income per family member
    if "CNT_FAM_MEMBERS" in df.columns:
        df["INCOME_PER_MEMBER"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1)

    # Income per child
    if "CNT_CHILDREN" in df.columns:
        df["INCOME_PER_CHILD"] = df["AMT_INCOME_TOTAL"] / (df["CNT_CHILDREN"] + 1)

    return df


# ── 2. Age & Employment Features ─────────────────────────────────────────────
def add_age_employment_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding age and employment features...")

    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"]    = df["DAYS_BIRTH"] / 365
        df["AGE_GROUP"]    = pd.cut(
            df["AGE_YEARS"],
            bins  =[0, 25, 35, 45, 55, 100],
            labels=[0,  1,  2,  3,   4],
        ).astype(float)

    if "DAYS_EMPLOYED" in df.columns:
        df["EMPLOYED_YEARS"] = df["DAYS_EMPLOYED"] / 365

        # Employment stability: years employed relative to age
        if "AGE_YEARS" in df.columns:
            df["EMPLOYMENT_RATIO"] = df["EMPLOYED_YEARS"] / (df["AGE_YEARS"] + 1)

    if "DAYS_REGISTRATION" in df.columns:
        df["REGISTRATION_YEARS"] = df["DAYS_REGISTRATION"] / 365

    if "DAYS_ID_PUBLISH" in df.columns:
        df["ID_PUBLISH_YEARS"] = df["DAYS_ID_PUBLISH"] / 365

    return df


# ── 3. External Score Features ────────────────────────────────────────────────
def add_external_score_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding external score features...")

    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]

    if len(ext_cols) >= 2:
        df["EXT_SOURCE_MEAN"]    = df[ext_cols].mean(axis=1)
        df["EXT_SOURCE_STD"]     = df[ext_cols].std(axis=1).fillna(0)
        df["EXT_SOURCE_MIN"]     = df[ext_cols].min(axis=1)
        df["EXT_SOURCE_MAX"]     = df[ext_cols].max(axis=1)
        df["EXT_SOURCE_RANGE"]   = df["EXT_SOURCE_MAX"] - df["EXT_SOURCE_MIN"]

    if "EXT_SOURCE_2" in df.columns and "EXT_SOURCE_3" in df.columns:
        df["EXT_23_PRODUCT"]     = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
        df["EXT_23_RATIO"]       = df["EXT_SOURCE_2"] / (df["EXT_SOURCE_3"] + 1e-9)

    return df


# ── 4. Document & Flag Features ───────────────────────────────────────────────
def add_flag_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding document and flag aggregate features...")

    # Count of submitted documents
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        df["DOCUMENT_COUNT"] = df[doc_cols].sum(axis=1)

    # Count of social circle defaults
    social_cols = [c for c in df.columns if "SOCIAL_CIRCLE" in c]
    if social_cols:
        df["SOCIAL_CIRCLE_DEFAULTS"] = df[
            [c for c in social_cols if "DEF" in c]
        ].sum(axis=1)

    # Contact flags sum
    contact_cols = [c for c in df.columns if c.startswith("FLAG_") and "CONTACT" in c]
    if contact_cols:
        df["CONTACT_FLAG_SUM"] = df[contact_cols].sum(axis=1)

    return df


# ── 5. Risk Score Feature ─────────────────────────────────────────────────────
def add_composite_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding composite risk score...")

    risk_score = pd.Series(np.zeros(len(df)), index=df.index)

    if "DEBT_TO_INCOME" in df.columns:
        risk_score += df["DEBT_TO_INCOME"].clip(0, 10) / 10 * 0.3

    if "ANNUITY_TO_INCOME" in df.columns:
        risk_score += df["ANNUITY_TO_INCOME"].clip(0, 1) * 0.2

    if "EXT_SOURCE_MEAN" in df.columns:
        risk_score += (1 - df["EXT_SOURCE_MEAN"].clip(0, 1)) * 0.35

    if "EMPLOYMENT_RATIO" in df.columns:
        risk_score += (1 - df["EMPLOYMENT_RATIO"].clip(0, 1)) * 0.15

    df["COMPOSITE_RISK_SCORE"] = risk_score.clip(0, 1)

    return df


# ── Master Pipeline ───────────────────────────────────────────────────────────
def run_feature_engineering(df: pd.DataFrame = None, save: bool = True) -> pd.DataFrame:
    if df is None:
        logger.info(f"Loading cleaned data from {PROCESSED_TRAIN}")
        df = pd.read_csv(PROCESSED_TRAIN)

    original_cols = df.shape[1]

    df = add_financial_ratios(df)
    df = add_age_employment_features(df)
    df = add_external_score_features(df)
    df = add_flag_features(df)
    df = add_composite_risk_score(df)

    # Final cleanup — drop any inf/nan introduced by divisions
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    new_cols = df.shape[1] - original_cols
    logger.info(f"Feature engineering complete. Added {new_cols} new features. Final shape: {df.shape}")

    if save:
        df.to_csv(PROCESSED_FINAL, index=False)
        logger.info(f"Final features saved → {PROCESSED_FINAL}")

    return df


if __name__ == "__main__":
    run_feature_engineering()