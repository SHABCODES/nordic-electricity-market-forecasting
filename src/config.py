"""
Centralized configuration for the forecasting pipeline.

All paths, hyperparameters, feature definitions, and model configs live here
so that nothing is hard-coded across modules.
"""

from pathlib import Path
from typing import Dict, Any, List

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
IMAGES_DIR = PROJECT_ROOT / "images"

RAW_DATA_PATH = DATA_DIR / "cleaned_finland_energy_prices.csv"

# Ensure output directories exist
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================
# DATA CONFIG
# ============================================================

TIMESTAMP_COL = "timestamp"
TARGET_COL = "price"
TEST_FRACTION = 0.20  # Chronological 80/20 split
RANDOM_STATE = 42

# ============================================================
# FEATURE ENGINEERING CONFIG
# ============================================================

# Lag features (in number of rows; 1 row = 15 min)
LAG_PERIODS: List[int] = [
    4,     # 1 hour
    96,    # 24 hours (1 day)
    672,   # 168 hours (1 week)
]

# Rolling window sizes (in number of rows)
ROLLING_WINDOWS: List[int] = [
    96,    # 24-hour rolling
    672,   # 1-week rolling
]

# Fourier pairs for cyclical encoding
FOURIER_FEATURES: Dict[str, int] = {
    "hour": 24,        # Period for hour-of-day
    "day_of_week": 7,  # Period for day-of-week
    "month": 12,       # Period for month-of-year
}

# Final feature list (auto-built by feature_engineering module)
CALENDAR_FEATURES = [
    "hour", "day_of_month", "month", "day_of_week",
    "is_weekend", "is_business_hour",
]

# ============================================================
# MODEL HYPERPARAMETERS
# ============================================================

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ridge": {
        "class": "sklearn.linear_model.Ridge",
        "params": {"alpha": 1.0},
        "scale_features": True,
    },
    "random_forest": {
        "class": "sklearn.ensemble.RandomForestRegressor",
        "params": {
            "n_estimators": 300,
            "max_depth": 20,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
        },
        "scale_features": False,
    },
    "xgboost": {
        "class": "xgboost.XGBRegressor",
        "params": {
            "n_estimators": 500,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
            "verbosity": 0,
        },
        "scale_features": False,
    },
    "lightgbm": {
        "class": "lightgbm.LGBMRegressor",
        "params": {
            "n_estimators": 500,
            "max_depth": 10,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
            "verbose": -1,
        },
        "scale_features": False,
    },
}

# Time-series cross-validation
TSCV_N_SPLITS = 5

# ============================================================
# EVALUATION
# ============================================================

METRICS_LIST = ["mae", "rmse", "mape", "r2", "directional_accuracy"]
