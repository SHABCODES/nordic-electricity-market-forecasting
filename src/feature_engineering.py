"""
Feature engineering for electricity price time-series.

All transforms are fully vectorized (no Python loops) using
Pandas .shift(), .rolling(), and NumPy broadcasting. This module
demonstrates production-grade time-series feature construction.
"""

import pandas as pd
import numpy as np
from typing import List

from src.config import (
    TIMESTAMP_COL, TARGET_COL,
    LAG_PERIODS, ROLLING_WINDOWS, FOURIER_FEATURES,
    CALENDAR_FEATURES,
)
from src.utils import setup_logger, timer

logger = setup_logger("features")


@timer
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transforms to the dataframe.

    Transforms applied (in order):
      1. Calendar features (hour, day, month, weekday, is_weekend, is_business_hour)
      2. Fourier cyclical encoding (sin/cos for periodic features)
      3. Lag features (price at t-1h, t-24h, t-1w)
      4. Rolling statistics (mean, std over 24h and 1-week windows)
      5. Price derivatives (diff, pct_change)
      6. Interaction features (hour × weekend)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'timestamp' and 'price' columns.

    Returns
    -------
    pd.DataFrame
        Original columns + all engineered features. Rows with NaN
        from lag/rolling operations are dropped.
    """
    df = df.copy()

    # ── 1. Calendar features ──────────────────────────────────
    df = _add_calendar_features(df)

    # ── 2. Fourier cyclical encoding ──────────────────────────
    df = _add_fourier_features(df)

    # ── 3. Lag features ───────────────────────────────────────
    df = _add_lag_features(df)

    # ── 4. Rolling statistics ─────────────────────────────────
    df = _add_rolling_features(df)

    # ── 5. Price derivatives ──────────────────────────────────
    df = _add_derivative_features(df)

    # ── 6. Interaction features ───────────────────────────────
    df = _add_interaction_features(df)

    # ── Drop rows with NaN from lag/rolling (first N rows) ────
    n_before = len(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(
        f"Feature engineering complete | "
        f"Dropped {n_before - len(df)} rows (lag/rolling warm-up) | "
        f"Final: {len(df):,} rows × {len(df.columns)} columns"
    )
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return list of feature column names (excludes timestamp, price, price_change)."""
    exclude = {TIMESTAMP_COL, TARGET_COL, "price_change", "year"}
    return [c for c in df.columns if c not in exclude]


# ================================================================
# PRIVATE TRANSFORM FUNCTIONS
# ================================================================

def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar components from the timestamp column."""
    ts = df[TIMESTAMP_COL]
    df["hour"] = ts.dt.hour
    df["day_of_month"] = ts.dt.day
    df["month"] = ts.dt.month
    df["day_of_week"] = ts.dt.weekday
    df["is_weekend"] = (ts.dt.weekday >= 5).astype(np.int8)
    df["is_business_hour"] = (
        (ts.dt.hour >= 8) & (ts.dt.hour <= 18) & (ts.dt.weekday < 5)
    ).astype(np.int8)
    logger.info("  + Calendar features: hour, day_of_month, month, day_of_week, is_weekend, is_business_hour")
    return df


def _add_fourier_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode periodic features as sin/cos pairs.

    This is far superior to raw integer encoding for tree models and
    essential for linear models — it captures the cyclical nature of
    hour-of-day and day-of-week without artificial discontinuities
    (e.g., hour 23 → 0 is a small step, not a jump of 23).
    """
    for col, period in FOURIER_FEATURES.items():
        if col == "hour":
            values = df["hour"].values
        elif col == "day_of_week":
            values = df["day_of_week"].values
        elif col == "month":
            values = df["month"].values
        else:
            continue

        # Vectorized sin/cos transform (NumPy broadcasting)
        angle = 2 * np.pi * values / period
        df[f"sin_{col}"] = np.sin(angle)
        df[f"cos_{col}"] = np.cos(angle)

    fourier_cols = [c for c in df.columns if c.startswith(("sin_", "cos_"))]
    logger.info(f"  + Fourier features: {fourier_cols}")
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lagged price features using pd.Series.shift().

    Lag features provide autoregressive signal — the single strongest
    predictor family for electricity prices because prices are highly
    autocorrelated (today's 3pm price ≈ yesterday's 3pm price).
    """
    for lag in LAG_PERIODS:
        hours = lag / 4  # Convert 15-min rows to hours
        col_name = f"price_lag_{int(hours)}h"
        df[col_name] = df[TARGET_COL].shift(lag)

    lag_cols = [c for c in df.columns if c.startswith("price_lag_")]
    logger.info(f"  + Lag features: {lag_cols}")
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling mean and std over specified windows.

    Rolling statistics capture the current price regime (trending up,
    high volatility, etc.) which is critical context for the model.
    """
    for window in ROLLING_WINDOWS:
        hours = window / 4
        # Rolling mean — captures current price level / trend
        df[f"rolling_mean_{int(hours)}h"] = (
            df[TARGET_COL]
            .shift(1)  # Shift to avoid look-ahead
            .rolling(window=window, min_periods=1)
            .mean()
        )
        # Rolling std — captures current volatility regime
        df[f"rolling_std_{int(hours)}h"] = (
            df[TARGET_COL]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .std()
        )

    rolling_cols = [c for c in df.columns if c.startswith("rolling_")]
    logger.info(f"  + Rolling features: {rolling_cols}")
    return df


def _add_derivative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute price rate-of-change features.

    These capture momentum — whether prices are accelerating up or down.
    """
    # 1-step diff (15 min change)
    df["price_diff_1"] = df[TARGET_COL].shift(1).diff()
    # 4-step diff (1 hour change)
    df["price_diff_4"] = df[TARGET_COL].shift(1) - df[TARGET_COL].shift(5)
    # Percentage change (1 step)
    df["price_pct_change_1"] = df[TARGET_COL].shift(1).pct_change()
    # Cap extreme pct changes to avoid inf
    df["price_pct_change_1"] = df["price_pct_change_1"].clip(-5, 5)

    logger.info("  + Derivative features: price_diff_1, price_diff_4, price_pct_change_1")
    return df


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features that capture non-linear relationships.

    Peak pricing patterns differ significantly between weekdays and weekends.
    """
    df["hour_x_weekend"] = df["hour"] * df["is_weekend"]
    df["hour_x_business"] = df["hour"] * df["is_business_hour"]
    logger.info("  + Interaction features: hour_x_weekend, hour_x_business")
    return df
