"""
Model evaluation: metrics, comparison tables, residual analysis, and
error breakdowns by time slice.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import TARGET_COL, TIMESTAMP_COL, RESULTS_DIR
from src.utils import setup_logger, timer

logger = setup_logger("evaluation")


# ================================================================
# METRICS
# ================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute a comprehensive set of regression metrics.

    Returns
    -------
    Dict with keys: mae, rmse, mape, r2, directional_accuracy
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE — skip zeros in denominator
    mask = y_true != 0
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = float("nan")

    # Directional accuracy — did we predict the direction of change?
    if len(y_true) > 1:
        actual_dir = np.diff(y_true) > 0
        pred_dir = np.diff(y_pred) > 0
        dir_acc = float(np.mean(actual_dir == pred_dir) * 100)
    else:
        dir_acc = float("nan")

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 2),
        "r2": round(r2, 4),
        "directional_accuracy": round(dir_acc, 2),
    }


# ================================================================
# MODEL COMPARISON
# ================================================================

@timer
def evaluate_all_models(
    model_results: Dict[str, Dict[str, Any]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predict_fn,
) -> pd.DataFrame:
    """
    Evaluate all trained models on the test set and produce a comparison table.

    Parameters
    ----------
    model_results : dict
        Output from train_all_models().
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.
    predict_fn : callable
        Function(model_result, X) → predictions array.

    Returns
    -------
    pd.DataFrame
        Comparison table with one row per model.
    """
    rows = []

    for name, result in model_results.items():
        logger.info(f"Evaluating {name}...")
        preds = predict_fn(result, X_test)
        metrics = compute_metrics(y_test.values, preds)
        metrics["model"] = name
        metrics["train_time_s"] = round(result["train_time"], 2)
        rows.append(metrics)

    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df[
        ["model", "mae", "rmse", "mape", "r2", "directional_accuracy", "train_time_s"]
    ].sort_values("rmse")

    # Log the table
    logger.info("\n" + "=" * 70)
    logger.info("MODEL COMPARISON (sorted by RMSE)")
    logger.info("=" * 70)
    logger.info("\n" + comparison_df.to_string(index=False))

    # Save to disk
    comparison_path = RESULTS_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"\nSaved comparison → {comparison_path}")

    return comparison_df


# ================================================================
# RESIDUAL ANALYSIS
# ================================================================

def compute_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Build a residual analysis DataFrame.

    Returns columns: actual, predicted, residual, abs_residual, pct_error
    (plus timestamp and hour if timestamps provided).
    """
    residuals = y_true - y_pred
    # Compute pct_error safely — np.divide with where= avoids
    # evaluating the division for zero-price rows entirely
    pct_err = np.full_like(y_true, np.nan, dtype=np.float64)
    nonzero = y_true != 0
    np.divide(np.abs(residuals), np.abs(y_true), out=pct_err, where=nonzero)
    pct_err *= 100

    df = pd.DataFrame({
        "actual": y_true,
        "predicted": y_pred,
        "residual": residuals,
        "abs_residual": np.abs(residuals),
        "pct_error": pct_err,
    })
    if timestamps is not None:
        df["timestamp"] = timestamps.values
        df["hour"] = pd.to_datetime(timestamps.values).hour
        df["month"] = pd.to_datetime(timestamps.values).month
        df["day_of_week"] = pd.to_datetime(timestamps.values).weekday
    return df


def error_by_time_slice(residual_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Break down MAE by hour-of-day, month, and day-of-week.

    This reveals when the model struggles most — critical insight
    for model improvement and operational deployment.
    """
    breakdowns = {}

    if "hour" in residual_df.columns:
        by_hour = residual_df.groupby("hour").agg(
            mae=("abs_residual", "mean"),
            rmse=("residual", lambda x: np.sqrt(np.mean(x**2))),
            count=("residual", "count"),
        ).round(3)
        breakdowns["by_hour"] = by_hour

    if "month" in residual_df.columns:
        by_month = residual_df.groupby("month").agg(
            mae=("abs_residual", "mean"),
            rmse=("residual", lambda x: np.sqrt(np.mean(x**2))),
            count=("residual", "count"),
        ).round(3)
        breakdowns["by_month"] = by_month

    if "day_of_week" in residual_df.columns:
        by_dow = residual_df.groupby("day_of_week").agg(
            mae=("abs_residual", "mean"),
            rmse=("residual", lambda x: np.sqrt(np.mean(x**2))),
            count=("residual", "count"),
        ).round(3)
        breakdowns["by_day_of_week"] = by_dow

    return breakdowns


# ================================================================
# BEST MODEL SELECTION
# ================================================================

def select_best_model(
    comparison_df: pd.DataFrame,
    metric: str = "rmse",
) -> str:
    """Return the name of the best-performing model by the given metric."""
    if metric == "r2":
        best_idx = comparison_df[metric].idxmax()
    else:
        best_idx = comparison_df[metric].idxmin()
    best_name = comparison_df.loc[best_idx, "model"]
    logger.info(f"Best model by {metric}: {best_name}")
    return best_name


# ================================================================
# SAVE FULL RESULTS
# ================================================================

def save_evaluation_results(
    comparison_df: pd.DataFrame,
    residual_df: pd.DataFrame,
    breakdowns: Dict[str, pd.DataFrame],
    best_model_name: str,
) -> None:
    """Persist all evaluation artifacts to the results/ directory."""
    # Comparison table
    comparison_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    # Residuals
    residual_df.to_csv(RESULTS_DIR / "residuals.csv", index=False)

    # Error breakdowns
    for key, breakdown_df in breakdowns.items():
        breakdown_df.to_csv(RESULTS_DIR / f"error_{key}.csv")

    # Summary JSON
    summary = {
        "best_model": best_model_name,
        "models_evaluated": comparison_df["model"].tolist(),
        "best_metrics": comparison_df[
            comparison_df["model"] == best_model_name
        ].iloc[0].to_dict(),
    }
    with open(RESULTS_DIR / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"All evaluation artifacts saved → {RESULTS_DIR}/")
