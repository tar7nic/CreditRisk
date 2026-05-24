from src.data_cleaning       import run_cleaning_pipeline
from src.eda                 import run_eda
from src.database            import run_database_pipeline
from src.feature_eng         import run_feature_engineering
from src.train               import run_model_training
from src.shapEx              import run_shap_analysis
from src.logger              import get_logger

logger = get_logger("runner")

if __name__ == "__main__":
    logger.info("Starting CreditRisk Insights Pipeline...")

    logger.info("Step 1/6 — Data Cleaning")
    df_clean = run_cleaning_pipeline(save=True)

    logger.info("Step 2/6 — EDA")
    run_eda(df_clean)

    logger.info("Step 3/6 — Database + SQL")
    run_database_pipeline(df_clean)

    logger.info("Step 4/6 — Feature Engineering")
    df_final = run_feature_engineering(df_clean, save=True)

    logger.info("Step 5/6 — Model Training")
    training_output = run_model_training(df_final)

    logger.info("Step 6/6 — SHAP Explainability")
    run_shap_analysis(training_output["X_test"].sample(n=2000, random_state=42))

    logger.info("Pipeline Complete. All outputs saved.")