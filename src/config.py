import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW        = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED  = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR     = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR     = os.path.join(BASE_DIR, "reports", "figures")

# ── Raw Data Files ────────────────────────────────────────────────────────────
RAW_TRAIN       = os.path.join(DATA_RAW, "application_train.csv")

# ── Processed Data Files ──────────────────────────────────────────────────────
PROCESSED_TRAIN = os.path.join(DATA_PROCESSED, "cleaned_train.csv")
PROCESSED_FINAL = os.path.join(DATA_PROCESSED, "final_features.csv")

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH         = os.path.join(BASE_DIR, "data", "creditrisk.db")
DB_URL          = f"sqlite:///{DB_PATH}"

# ── Model Config ──────────────────────────────────────────────────────────────
TARGET_COL      = "TARGET"
RANDOM_STATE    = 42
TEST_SIZE       = 0.2
CV_FOLDS        = 5

# ── Class Imbalance ───────────────────────────────────────────────────────────
SMOTE_RANDOM_STATE = 42

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"