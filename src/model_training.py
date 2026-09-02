"""
Multi-model training pipeline with temporal cross-validation.

Trains Ridge, Random Forest, XGBoost, and LightGBM with proper
time-series validation. Demonstrates algorithm comparison,
hyperparameter tuning, and model persistence.
"""

import importlib
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

from src.config import (
    TARGET_COL, MODEL_CONFIGS, MODELS_DIR,
    TSCV_N_SPLITS, RANDOM_STATE,
)
from src.utils import setup_logger, timer

logger = setup_logger("training")


# ================================================================
# HYPERPARAMETER SEARCH SPACES (for RandomizedSearchCV)
# ================================================================

SEARCH_SPACES: Dict[str, Dict[str, Any]] = {
    "random_forest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [10, 15, 20, 25, None],
        "min_samples_split": [5, 10, 20],
        "min_samples_leaf": [3, 5, 10],
    },
    "xgboost": {
        "n_estimators": [200, 300, 500],
        "max_depth": [5, 8, 10, 12],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
    },
    "lightgbm": {
        "n_estimators": [200, 300, 500],
        "max_depth": [6, 8, 10, 12],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "num_leaves": [31, 50, 70, 100],
    },
}


def _import_model_class(class_path: str) -> Any:
    """Dynamically import a model class from its dotted path."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


@timer
def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: List[str],
    tune_hyperparams: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Train all configured models and return results.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    feature_names : List[str]
        Feature column names (for importance extraction).
    tune_hyperparams : bool
        If True, run RandomizedSearchCV with TimeSeriesSplit on
        tree-based models. Slower but may improve accuracy.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Keys are model names. Values contain:
          - 'model': fitted estimator
          - 'scaler': fitted StandardScaler (or None)
          - 'train_time': wall-clock seconds
          - 'cv_score': mean CV R² (if tuning was done)
          - 'best_params': best hyperparameters (if tuning was done)
    """
    results: Dict[str, Dict[str, Any]] = {}

    for name, config in MODEL_CONFIGS.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Training: {name.upper()}")
        logger.info(f"{'='*50}")

        try:
            result = _train_single_model(
                name=name,
                config=config,
                X_train=X_train,
                y_train=y_train,
                tune=tune_hyperparams and name in SEARCH_SPACES,
            )
            results[name] = result

            # Save model artifact
            _save_model(name, result)

            logger.info(
                f"  {name}: trained in {result['train_time']:.2f}s"
                + (f" | CV R²={result.get('cv_score', 'N/A')}" if result.get('cv_score') else "")
            )

        except Exception as e:
            logger.error(f"  ✗ Failed to train {name}: {e}")
            continue

    logger.info(f"\nSuccessfully trained {len(results)}/{len(MODEL_CONFIGS)} models")
    return results


def _train_single_model(
    name: str,
    config: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune: bool = False,
) -> Dict[str, Any]:
    """Train a single model with optional hyperparameter tuning."""
    ModelClass = _import_model_class(config["class"])
    scaler = None

    # Scale features if required (Ridge regression)
    X_fit = X_train.copy()
    if config.get("scale_features", False):
        scaler = StandardScaler()
        X_fit = pd.DataFrame(
            scaler.fit_transform(X_fit),
            columns=X_fit.columns,
            index=X_fit.index,
        )

    start = time.perf_counter()

    if tune and name in SEARCH_SPACES:
        # ── Hyperparameter tuning with TimeSeriesSplit ────────
        base_model = ModelClass(**{
            k: v for k, v in config["params"].items()
            if k not in SEARCH_SPACES[name]
        })

        tscv = TimeSeriesSplit(n_splits=TSCV_N_SPLITS)
        search = RandomizedSearchCV(
            base_model,
            param_distributions=SEARCH_SPACES[name],
            n_iter=20,
            cv=tscv,
            scoring="r2",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=0,
        )
        search.fit(X_fit, y_train)
        model = search.best_estimator_
        cv_score = search.best_score_
        best_params = search.best_params_
        logger.info(f"  Best CV R²: {cv_score:.4f}")
        logger.info(f"  Best params: {best_params}")
    else:
        # ── Direct training with configured params ────────────
        model = ModelClass(**config["params"])
        model.fit(X_fit, y_train)
        cv_score = None
        best_params = config["params"]

    train_time = time.perf_counter() - start

    return {
        "model": model,
        "scaler": scaler,
        "train_time": train_time,
        "cv_score": cv_score,
        "best_params": best_params,
    }


def _save_model(name: str, result: Dict[str, Any]) -> None:
    """Persist model and scaler to disk."""
    model_path = MODELS_DIR / f"{name}_model.joblib"
    joblib.dump(result["model"], model_path)
    logger.info(f"  Saved model → {model_path}")

    if result["scaler"] is not None:
        scaler_path = MODELS_DIR / f"{name}_scaler.joblib"
        joblib.dump(result["scaler"], scaler_path)
        logger.info(f"  Saved scaler → {scaler_path}")


def load_model(name: str) -> Dict[str, Any]:
    """Load a saved model and its scaler (if any) from disk."""
    model_path = MODELS_DIR / f"{name}_model.joblib"
    model = joblib.load(model_path)

    scaler_path = MODELS_DIR / f"{name}_scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    return {"model": model, "scaler": scaler}


def predict(
    model_result: Dict[str, Any],
    X: pd.DataFrame,
) -> np.ndarray:
    """Generate predictions, applying scaler if the model requires it."""
    X_pred = X.copy()
    if model_result.get("scaler") is not None:
        X_pred = pd.DataFrame(
            model_result["scaler"].transform(X_pred),
            columns=X_pred.columns,
            index=X_pred.index,
        )
    return model_result["model"].predict(X_pred)


def get_feature_importance(
    model_result: Dict[str, Any],
    feature_names: List[str],
    model_name: str,
) -> Optional[pd.DataFrame]:
    """
    Extract feature importance for tree-based models.
    Returns a sorted DataFrame or None for models without importances.
    """
    model = model_result["model"]
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return importance_df
    return None
