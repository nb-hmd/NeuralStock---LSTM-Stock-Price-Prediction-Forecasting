"""
main.py — End-to-end Stock Price Prediction pipeline using LSTM.

Stages executed sequentially:
  1. Data Acquisition      — Download real OHLCV data from Yahoo Finance
  2. Feature Engineering   — Add SMA, EMA, RSI, MACD, Bollinger Bands
  3. Preprocessing         — Clean, scale, split, generate sequences
  4. Model Building        — Stacked LSTM architecture
  5. Training              — EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
  6. Evaluation            — RMSE, MAE, MAPE, R², Directional Accuracy
  7. Visualization         — 7 publication-quality plots
  8. Future Forecast       — Recursive 30-day prediction

Usage:
  python main.py

To change the stock ticker or any parameter, edit src/config.py.
"""

import logging
import sys
import os
import warnings
import io
warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup (must come before any project imports)
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "outputs", "reports", "pipeline.log"
            ),
            mode="w",
            encoding="utf-8"
        )
    ]
)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from src import config as cfg
from src.data_acquisition    import download_stock_data
from src.feature_engineering import add_technical_indicators, get_feature_columns
from src.preprocessing       import (
    handle_missing_values,
    chronological_split,
    fit_scaler,
    scale_data,
    create_sequences,
    validation_split,
    inverse_transform_predictions,
)
from src.model               import build_lstm_model
from src.train               import train_model, set_random_seeds
from src.evaluate            import evaluate_model
from src.visualize           import (
    plot_price_history,
    plot_train_test_split,
    plot_feature_correlation,
    plot_training_history,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_future_forecast,
    print_plot_summary,
)


def generate_future_dates(last_date: pd.Timestamp, n_days: int) -> pd.DatetimeIndex:
    """Generate n_days of business (trading) dates after last_date."""
    future = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_days)
    return future


def recursive_forecast(
    model,
    last_window: np.ndarray,
    scaler,
    n_features: int,
    target_col_idx: int,
    n_days: int = cfg.FORECAST_DAYS
) -> np.ndarray:
    """
    Generate a multi-step recursive forecast.

    At each step:
      1. Feed the current 60-day window → model → scaled prediction
      2. Append the scaled prediction to the window (drop oldest day)
      3. Repeat for n_days steps

    Note: error compounds over time; short forecasts (≤30 days) are most reliable.

    Parameters
    ----------
    last_window : np.ndarray  shape (1, seq_len, n_features)  — last known window
    scaler      : MinMaxScaler fitted on training data
    n_features  : int
    target_col_idx : int
    n_days      : int

    Returns
    -------
    np.ndarray  shape (n_days,)  — predicted prices in real $ scale
    """
    window = last_window.copy()          # (1, seq_len, n_features)
    predictions_scaled = []

    for _ in range(n_days):
        pred_scaled = model.predict(window, verbose=0)[0, 0]  # scalar
        predictions_scaled.append(pred_scaled)

        # Build new timestep row (copy last row, overwrite target column)
        new_step = window[0, -1, :].copy()
        new_step[target_col_idx] = pred_scaled

        # Slide window: drop oldest, append new step
        window = np.concatenate(
            [window[:, 1:, :], new_step.reshape(1, 1, n_features)],
            axis=1
        )

    # Inverse-transform
    future_prices = inverse_transform_predictions(
        np.array(predictions_scaled),
        scaler,
        n_features,
        target_col_idx
    )
    return future_prices


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    set_random_seeds(cfg.RANDOM_SEED)

    print("\n" + "#" * 60)
    print("  STOCK PRICE PREDICTION -- LSTM PIPELINE")
    print(f"  Ticker: {cfg.TICKER}  |  {cfg.START_DATE} -> {cfg.END_DATE}")
    print("#" * 60 + "\n")

    # -- STAGE 1: Data Acquisition --
    print("-" * 60)
    print("  STAGE 1/7 -- Data Acquisition")
    print("-" * 60)
    raw_df = download_stock_data(
        ticker=cfg.TICKER,
        start=cfg.START_DATE,
        end=cfg.END_DATE
    )

    # -- STAGE 2: Feature Engineering --
    print("-" * 60)
    print("  STAGE 2/7 -- Feature Engineering")
    print("-" * 60)
    # Clean missing values first (on raw data before indicator computation)
    raw_df = handle_missing_values(raw_df)
    feat_df = add_technical_indicators(raw_df)
    feature_cols = get_feature_columns(feat_df)
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    # -- STAGE 3: Preprocessing --
    print("-" * 60)
    print("  STAGE 3/7 -- Preprocessing")
    print("-" * 60)

    # Plot 1: full price history (before split)
    plot_price_history(raw_df, ticker=cfg.TICKER)

    # Chronological train/test split
    train_df, test_df = chronological_split(feat_df[feature_cols],
                                            test_split=cfg.TEST_SPLIT)

    # Plot 2: train/test split visualization
    plot_train_test_split(train_df, test_df, ticker=cfg.TICKER)

    # Plot 3: feature correlation heatmap
    plot_feature_correlation(feat_df[feature_cols])

    # Fit scaler on training data ONLY
    scaler = fit_scaler(train_df)

    # Scale both splits
    train_scaled, test_scaled = scale_data(train_df, test_df, scaler)

    # Determine target column index in feature list
    target_col_idx = feature_cols.index(cfg.TARGET_COL)
    n_features     = len(feature_cols)

    # Create sequences
    X_train_full, y_train_full = create_sequences(
        train_scaled, cfg.SEQUENCE_LENGTH,
        target_col_idx=target_col_idx
    )
    X_test, y_test = create_sequences(
        test_scaled, cfg.SEQUENCE_LENGTH,
        target_col_idx=target_col_idx
    )

    # Validation split (chronological, from training sequences)
    X_tr, X_val, y_tr, y_val = validation_split(
        X_train_full, y_train_full,
        val_fraction=cfg.VALIDATION_SPLIT
    )

    print(f"\n  Shapes — X_train: {X_tr.shape}, X_val: {X_val.shape}, "
          f"X_test: {X_test.shape}")

    # -- STAGE 4: Model Building --
    print("\n" + "-" * 60)
    print("  STAGE 4/7 -- Model Building")
    print("-" * 60)
    model = build_lstm_model(
        sequence_length=cfg.SEQUENCE_LENGTH,
        n_features=n_features
    )

    # -- STAGE 5: Training --
    print("-" * 60)
    print("  STAGE 5/7 -- Training")
    print("-" * 60)
    history = train_model(
        model, X_tr, y_tr, X_val, y_val,
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE
    )

    # Plot 4: training/validation loss curves
    hist_df = pd.read_csv(cfg.HISTORY_PATH, index_col=0)
    plot_training_history(hist_df)

    # -- STAGE 6: Evaluation --
    print("-" * 60)
    print("  STAGE 6/7 -- Evaluation")
    print("-" * 60)

    # Generate predictions on test set
    y_pred_scaled = model.predict(X_test, verbose=0)

    # Inverse-transform to real $ scale
    y_pred_real = inverse_transform_predictions(
        y_pred_scaled, scaler, n_features, target_col_idx
    )
    y_true_real = inverse_transform_predictions(
        y_test, scaler, n_features, target_col_idx
    )

    # Compute test-set dates (aligned with sequences)
    # Sequences start at index SEQUENCE_LENGTH inside test_df
    test_dates = test_df.index[cfg.SEQUENCE_LENGTH:]

    # Evaluate
    metrics, residual_stats = evaluate_model(y_true_real, y_pred_real, ticker=cfg.TICKER)

    # Plot 5: actual vs. predicted
    plot_actual_vs_predicted(test_dates, y_true_real, y_pred_real,
                             metrics, ticker=cfg.TICKER)

    # Plot 6: residuals
    plot_residuals(test_dates, y_true_real, y_pred_real, ticker=cfg.TICKER)

    # -- STAGE 7: Future Forecast --
    print("-" * 60)
    print(f"  STAGE 7/7 -- Recursive {cfg.FORECAST_DAYS}-Day Forecast")
    print("-" * 60)

    # Use the last SEQUENCE_LENGTH rows of the full (train+test) scaled data
    full_scaled = np.vstack([train_scaled, test_scaled])
    last_window = full_scaled[-cfg.SEQUENCE_LENGTH:].reshape(
        1, cfg.SEQUENCE_LENGTH, n_features
    )

    future_prices = recursive_forecast(
        model, last_window, scaler, n_features,
        target_col_idx, n_days=cfg.FORECAST_DAYS
    )

    last_real_date = feat_df.index[-1]
    future_dates   = generate_future_dates(last_real_date, cfg.FORECAST_DAYS)

    # Print future forecast table
    print("  Forecasted Prices:")
    print("  " + "-" * 30)
    for d, p in zip(future_dates, future_prices):
        print(f"  {d.date()}  ->  ${p:.2f}")
    print("  " + "-" * 30 + "\n")

    # Plot 7: future forecast
    all_close = feat_df["Close"].values
    plot_future_forecast(
        historical_dates=feat_df.index,
        historical_prices=all_close,
        forecast_dates=future_dates,
        forecast_prices=future_prices,
        residual_stats=residual_stats,
        ticker=cfg.TICKER
    )

    # ── Final Summary ─────────────────────────────────────────────────────────
    print_plot_summary()

    print("\n" + "#" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Ticker       : {cfg.TICKER}")
    print(f"  RMSE         : ${metrics['RMSE']:.4f}")
    print(f"  MAE          : ${metrics['MAE']:.4f}")
    print(f"  MAPE         : {metrics['MAPE']:.4f}%")
    print(f"  R2           : {metrics['R2']:.4f}")
    print(f"  Dir. Accuracy: {metrics['DA']:.2f}%")
    print(f"  Model saved  : {cfg.MODEL_PATH}")
    print(f"  Plots saved  : {cfg.PLOTS_DIR}")
    print(f"  Report saved : {cfg.METRICS_JSON}")
    print("#" * 60 + "\n")

    return metrics


if __name__ == "__main__":
    main()
