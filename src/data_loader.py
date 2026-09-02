"""
Data loading, validation, and temporal train/test splitting.

Handles reading the ENTSO-E CSV, type enforcement, and chronological
splitting that preserves time-series integrity (no data leakage).
"""

import pandas as pd
import numpy as np
from typing import Tuple

from src.config import RAW_DATA_PATH, TIMESTAMP_COL, TARGET_COL, TEST_FRACTION
from src.utils import setup_logger, timer

logger = setup_logger("data_loader")


@timer
def load_data(filepath: str | None = None) -> pd.DataFrame:
    """
    Load and validate the Finland electricity price dataset.

    Parameters
    ----------
    filepath : str or None
        Path to CSV file. Defaults to config.RAW_DATA_PATH.

    Returns
    -------
    pd.DataFrame
        Sorted by timestamp, with proper dtypes enforced.
    """
    path = filepath or str(RAW_DATA_PATH)
    logger.info(f"Loading data from {path}")

    df = pd.read_csv(path)

    # ── Type enforcement ──────────────────────────────────────
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # ── Drop rows with missing timestamps or prices ───────────
    n_before = len(df)
    df.dropna(subset=[TIMESTAMP_COL, TARGET_COL], inplace=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning(f"Dropped {n_dropped} rows with missing timestamp/price")

    # ── Sort chronologically (critical for time-series) ───────
    df.sort_values(TIMESTAMP_COL, inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(
        f"Loaded {len(df):,} rows | "
        f"{df[TIMESTAMP_COL].min()} → {df[TIMESTAMP_COL].max()}"
    )
    return df


@timer
def temporal_train_test_split(
    df: pd.DataFrame,
    test_fraction: float = TEST_FRACTION,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically — the last `test_fraction` of rows become the
    test set. This prevents look-ahead data leakage that inflates metrics
    when using random splits on autocorrelated time-series data.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset, already sorted by timestamp.
    test_fraction : float
        Proportion of data to use as test set (default 0.20).

    Returns
    -------
    (train_df, test_df) : Tuple[pd.DataFrame, pd.DataFrame]
    """
    split_idx = int(len(df) * (1 - test_fraction))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    logger.info(
        f"Temporal split → Train: {len(train_df):,} rows "
        f"({train_df[TIMESTAMP_COL].min()} to {train_df[TIMESTAMP_COL].max()}) | "
        f"Test: {len(test_df):,} rows "
        f"({test_df[TIMESTAMP_COL].min()} to {test_df[TIMESTAMP_COL].max()})"
    )
    return train_df, test_df


def get_data_summary(df: pd.DataFrame) -> dict:
    """Return a summary dict for dashboard display."""
    return {
        "total_rows": len(df),
        "date_range": f"{df[TIMESTAMP_COL].min()} → {df[TIMESTAMP_COL].max()}",
        "price_mean": float(df[TARGET_COL].mean()),
        "price_std": float(df[TARGET_COL].std()),
        "price_min": float(df[TARGET_COL].min()),
        "price_max": float(df[TARGET_COL].max()),
        "null_count": int(df.isnull().sum().sum()),
        "frequency": "15-min intervals",
    }
