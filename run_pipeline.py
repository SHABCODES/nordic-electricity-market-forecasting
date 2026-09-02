"""
End-to-end CLI pipeline for Nordic Electricity Market Forecasting.

Usage:
    python run_pipeline.py                     # Full pipeline
    python run_pipeline.py --skip-training     # Load saved models, evaluate only
    python run_pipeline.py --tune              # Include hyperparameter tuning
    python run_pipeline.py --model xgboost     # Train only one model

Demonstrates: scalable backend systems, AI/ML pipelines, CLI design.
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from src.config import TARGET_COL, TIMESTAMP_COL, RESULTS_DIR, MODEL_CONFIGS
from src.data_loader import load_data, temporal_train_test_split, get_data_summary
from src.feature_engineering import engineer_features, get_feature_columns
from src.model_training import train_all_models, predict, load_model, get_feature_importance
from src.evaluation import (
    evaluate_all_models, compute_metrics, compute_residuals,
    error_by_time_slice, select_best_model, save_evaluation_results,
)
from src.utils import setup_logger

logger = setup_logger("pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nordic Electricity Market Forecasting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip training; load saved models and evaluate.",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run hyperparameter tuning with RandomizedSearchCV + TimeSeriesSplit.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=list(MODEL_CONFIGS.keys()),
        help="Train only a specific model (default: all).",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to input CSV (default: data/cleaned_finland_energy_prices.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline_start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("NORDIC ELECTRICITY MARKET FORECASTING PIPELINE")
    logger.info("=" * 60)

    # ── STEP 1: Load Data ─────────────────────────────────────
    logger.info("\n📂 STEP 1: Loading data...")
    df = load_data(args.data)
    summary = get_data_summary(df)
    logger.info(f"  Records: {summary['total_rows']:,}")
    logger.info(f"  Date range: {summary['date_range']}")
    logger.info(f"  Price range: {summary['price_min']:.2f} – {summary['price_max']:.2f} EUR/MWh")

    # ── STEP 2: Feature Engineering ───────────────────────────
    logger.info("\n🔧 STEP 2: Feature engineering...")
    df = engineer_features(df)
    feature_cols = get_feature_columns(df)
    logger.info(f"  Total features: {len(feature_cols)}")
    logger.info(f"  Features: {feature_cols}")

    # ── STEP 3: Temporal Train/Test Split ─────────────────────
    logger.info("\n✂️ STEP 3: Temporal train/test split...")
    train_df, test_df = temporal_train_test_split(df)

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]

    logger.info(f"  Train shape: {X_train.shape}")
    logger.info(f"  Test shape: {X_test.shape}")

    # ── STEP 4: Model Training ────────────────────────────────
    if args.skip_training:
        logger.info("\n⏭️ STEP 4: Loading saved models (--skip-training)...")
        model_results = {}
        for name in MODEL_CONFIGS:
            if args.model and name != args.model:
                continue
            try:
                model_results[name] = load_model(name)
                model_results[name]["train_time"] = 0.0
                logger.info(f"  Loaded: {name}")
            except FileNotFoundError:
                logger.warning(f"  ✗ No saved model for {name}")
    else:
        logger.info(f"\n🧠 STEP 4: Training models {'(with tuning)' if args.tune else ''}...")

        # If --model specified, filter configs
        if args.model:
            original_configs = dict(MODEL_CONFIGS)
            # Temporarily restrict to single model
            keys_to_remove = [k for k in MODEL_CONFIGS if k != args.model]
            for k in keys_to_remove:
                del MODEL_CONFIGS[k]

        model_results = train_all_models(
            X_train, y_train, feature_cols,
            tune_hyperparams=args.tune,
        )

        # Restore configs if filtered
        if args.model:
            MODEL_CONFIGS.update(original_configs)

    if not model_results:
        logger.error("No models available. Exiting.")
        sys.exit(1)

    # ── STEP 5: Evaluation ────────────────────────────────────
    logger.info("\n📊 STEP 5: Evaluating models...")
    comparison_df = evaluate_all_models(
        model_results, X_test, y_test, predict,
    )

    # ── STEP 6: Best Model Analysis ──────────────────────────
    logger.info("\n🏆 STEP 6: Best model analysis...")
    best_name = select_best_model(comparison_df, metric="rmse")
    best_result = model_results[best_name]

    # Predictions from best model
    best_preds = predict(best_result, X_test)

    # Residual analysis
    residual_df = compute_residuals(
        y_test.values, best_preds,
        timestamps=test_df[TIMESTAMP_COL],
    )
    breakdowns = error_by_time_slice(residual_df)

    # Log error breakdowns
    for key, bdf in breakdowns.items():
        logger.info(f"\n  Error {key}:")
        logger.info(f"\n{bdf.to_string()}")

    # Feature importance (for tree-based models)
    for name, result in model_results.items():
        fi = get_feature_importance(result, feature_cols, name)
        if fi is not None:
            logger.info(f"\n  Feature importance ({name}) — Top 10:")
            logger.info(f"\n{fi.head(10).to_string(index=False)}")
            fi.to_csv(RESULTS_DIR / f"feature_importance_{name}.csv", index=False)

    # ── STEP 7: Save Results ──────────────────────────────────
    logger.info("\n💾 STEP 7: Saving evaluation results...")
    save_evaluation_results(comparison_df, residual_df, breakdowns, best_name)

    # ── DONE ──────────────────────────────────────────────────
    elapsed = time.perf_counter() - pipeline_start
    logger.info("\n" + "=" * 60)
    logger.info(f"✅ PIPELINE COMPLETE in {elapsed:.1f}s")
    logger.info(f"   Best model: {best_name}")
    best_row = comparison_df[comparison_df["model"] == best_name].iloc[0]
    logger.info(f"   MAE:  {best_row['mae']:.4f}")
    logger.info(f"   RMSE: {best_row['rmse']:.4f}")
    logger.info(f"   R²:   {best_row['r2']:.4f}")
    logger.info(f"   Directional Accuracy: {best_row['directional_accuracy']:.1f}%")
    logger.info(f"   Results saved → {RESULTS_DIR}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
