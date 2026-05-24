import os
import pandas as pd
from sqlalchemy import create_engine, text
from src.config import DB_URL, PROCESSED_TRAIN, TARGET_COL
from src.logger import get_logger

logger = get_logger(__name__)


# ── Engine ────────────────────────────────────────────────────────────────────
def get_engine():
    engine = create_engine(DB_URL, echo=False)
    logger.info(f"Database engine created → {DB_URL}")
    return engine


# ── Store Cleaned Data ────────────────────────────────────────────────────────
def store_cleaned_data(df: pd.DataFrame = None) -> None:
    if df is None:
        logger.info("Loading cleaned data for DB storage...")
        df = pd.read_csv(PROCESSED_TRAIN)

    engine = get_engine()
    df.to_sql("loan_applications", con=engine, if_exists="replace", index=False, chunksize=5000)
    logger.info(f"Stored {len(df):,} rows into table: loan_applications")


# ── SQL Cohort Queries ────────────────────────────────────────────────────────
QUERIES = {
    "overall_default_rate": """
        SELECT
            COUNT(*)                                         AS total_applicants,
            SUM(TARGET)                                      AS total_defaults,
            ROUND(AVG(TARGET) * 100, 2)                      AS default_rate_pct,
            ROUND(AVG(AMT_INCOME_TOTAL), 2)                  AS avg_income,
            ROUND(AVG(AMT_CREDIT), 2)                        AS avg_credit
        FROM loan_applications
    """,

    "default_by_gender": """
        SELECT
            CODE_GENDER,
            COUNT(*)                            AS applicants,
            SUM(TARGET)                         AS defaults,
            ROUND(AVG(TARGET) * 100, 2)         AS default_rate_pct,
            ROUND(AVG(AMT_INCOME_TOTAL), 2)     AS avg_income
        FROM loan_applications
        GROUP BY CODE_GENDER
        ORDER BY default_rate_pct DESC
    """,

    "default_by_education": """
        SELECT
            NAME_EDUCATION_TYPE,
            COUNT(*)                            AS applicants,
            SUM(TARGET)                         AS defaults,
            ROUND(AVG(TARGET) * 100, 2)         AS default_rate_pct,
            ROUND(AVG(AMT_CREDIT), 2)           AS avg_credit
        FROM loan_applications
        GROUP BY NAME_EDUCATION_TYPE
        ORDER BY default_rate_pct DESC
    """,

    "income_cohorts": """
        SELECT
            CASE
                WHEN AMT_INCOME_TOTAL < 90000  THEN '1_Low (<90K)'
                WHEN AMT_INCOME_TOTAL < 180000 THEN '2_Mid (90K-180K)'
                WHEN AMT_INCOME_TOTAL < 360000 THEN '3_High (180K-360K)'
                ELSE '4_Very High (>360K)'
            END                                AS income_bracket,
            COUNT(*)                           AS applicants,
            SUM(TARGET)                        AS defaults,
            ROUND(AVG(TARGET) * 100, 2)        AS default_rate_pct,
            ROUND(AVG(AMT_CREDIT), 2)          AS avg_credit_amount
        FROM loan_applications
        GROUP BY income_bracket
        ORDER BY income_bracket
    """,

    "age_cohorts": """
        SELECT
            CASE
                WHEN DAYS_BIRTH / 365 < 30 THEN '1_Young (<30)'
                WHEN DAYS_BIRTH / 365 < 40 THEN '2_Adult (30-40)'
                WHEN DAYS_BIRTH / 365 < 55 THEN '3_Mid-Age (40-55)'
                ELSE '4_Senior (55+)'
            END                                AS age_group,
            COUNT(*)                           AS applicants,
            SUM(TARGET)                        AS defaults,
            ROUND(AVG(TARGET) * 100, 2)        AS default_rate_pct,
            ROUND(AVG(AMT_INCOME_TOTAL), 2)    AS avg_income
        FROM loan_applications
        GROUP BY age_group
        ORDER BY age_group
    """,

    "credit_stress_ratio": """
        SELECT
            CASE
                WHEN AMT_ANNUITY / NULLIF(AMT_INCOME_TOTAL, 0) < 0.1  THEN '1_Low Stress (<10%)'
                WHEN AMT_ANNUITY / NULLIF(AMT_INCOME_TOTAL, 0) < 0.2  THEN '2_Moderate (10-20%)'
                WHEN AMT_ANNUITY / NULLIF(AMT_INCOME_TOTAL, 0) < 0.35 THEN '3_High (20-35%)'
                ELSE '4_Very High (>35%)'
            END                                AS annuity_income_ratio,
            COUNT(*)                           AS applicants,
            ROUND(AVG(TARGET) * 100, 2)        AS default_rate_pct
        FROM loan_applications
        GROUP BY annuity_income_ratio
        ORDER BY annuity_income_ratio
    """,

    "high_risk_segment": """
        SELECT
            SK_ID_CURR,
            AMT_INCOME_TOTAL,
            AMT_CREDIT,
            AMT_ANNUITY,
            DAYS_BIRTH / 365          AS age_years,
            EXT_SOURCE_2,
            TARGET
        FROM loan_applications
        WHERE TARGET = 1
          AND EXT_SOURCE_2 < 0.3
          AND AMT_ANNUITY / NULLIF(AMT_INCOME_TOTAL, 0) > 0.25
        ORDER BY EXT_SOURCE_2 ASC
        LIMIT 50
    """,
}


def run_sql_queries(save_results: bool = True) -> dict:
    engine  = get_engine()
    outputs = {}

    out_dir = os.path.join(os.path.dirname(DB_URL.replace("sqlite:///", "")), "processed")
    os.makedirs(out_dir, exist_ok=True)

    for name, query in QUERIES.items():
        try:
            result = pd.read_sql(text(query), con=engine.connect())
            outputs[name] = result
            logger.info(f"\n{'='*50}\nQuery: {name}\n{result.to_string(index=False)}")

            if save_results:
                path = os.path.join(out_dir, f"sql_{name}.csv")
                result.to_csv(path, index=False)
                logger.info(f"Saved → {path}")

        except Exception as e:
            logger.error(f"Query '{name}' failed: {e}")

    return outputs


# ── Master DB Runner ──────────────────────────────────────────────────────────
def run_database_pipeline(df: pd.DataFrame = None) -> dict:
    store_cleaned_data(df)
    return run_sql_queries()


if __name__ == "__main__":
    run_database_pipeline()