"""
pipeline_runner.py — Ticker-aware pipeline orchestrator for the web UI.

Runs the full LSTM prediction pipeline for any stock ticker.
Saves all artifacts (model, scaler, plots, metrics) to ticker-specific
subdirectories so multiple tickers can coexist without conflicts.
"""

import io
import json
import logging
import os
import pickle
import sys
import threading
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONUTF8"] = "1"

# Re-wrap stdout/stderr to UTF-8 on Windows
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer") and sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import (
    Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

from src import config as cfg
from src.data_acquisition import download_stock_data
from src.evaluate import evaluate_model
from src.feature_engineering import add_technical_indicators, get_feature_columns
from src.model import build_lstm_model
from src.preprocessing import (
    chronological_split, create_sequences, fit_scaler,
    handle_missing_values, inverse_transform_predictions,
    scale_data, validation_split,
)
from src.visualize import (
    plot_actual_vs_predicted,
    plot_feature_correlation,
    plot_future_forecast,
    plot_price_history,
    plot_residuals,
    plot_train_test_split,
    plot_training_history,
)

from src.shariah import resolve_symbol
from src.train import set_random_seeds

logger = logging.getLogger(__name__)

# Per-ticker locks to prevent parallel runs for the same ticker
_ticker_locks: dict[str, threading.Lock] = {}
_locks_mutex = threading.Lock()


def _get_lock(ticker: str) -> threading.Lock:
    sym = resolve_symbol(ticker)["canonical_symbol"]
    with _locks_mutex:
        if sym not in _ticker_locks:
            _ticker_locks[sym] = threading.Lock()
        return _ticker_locks[sym]


def get_ticker_paths(ticker: str) -> dict:
    """Return all file-system paths for a given ticker."""
    sym_info = resolve_symbol(ticker)
    t = sym_info["storage_dir"]
    data_dir   = os.path.join(PROJECT_ROOT, "data", "raw")
    model_dir  = os.path.join(PROJECT_ROOT, "models",  t)
    plots_dir  = os.path.join(PROJECT_ROOT, "outputs", "plots",   t)
    report_dir = os.path.join(PROJECT_ROOT, "outputs", "reports", t)
    for d in (data_dir, model_dir, plots_dir, report_dir):
        os.makedirs(d, exist_ok=True)
    return {
        "raw_data":   os.path.join(data_dir,   f"{t}_raw.csv"),
        "model":      os.path.join(model_dir,  "lstm_model.keras"),
        "scaler":     os.path.join(model_dir,  "scaler.pkl"),
        "history":    os.path.join(report_dir, "training_history.csv"),
        "metrics_j":  os.path.join(report_dir, "evaluation_metrics.json"),
        "metrics_c":  os.path.join(report_dir, "evaluation_metrics.csv"),
        "forecast":   os.path.join(report_dir, "forecast.json"),
        "meta":       os.path.join(report_dir, "meta.json"),
        "plots_dir":  plots_dir,
    }


def is_cached(ticker: str) -> bool:
    """Return True if all 7 plots + metrics already exist for this ticker."""
    p = get_ticker_paths(ticker)
    if not os.path.exists(p["plots_dir"]):
        return False
    plots = [f for f in os.listdir(p["plots_dir"]) if f.endswith(".png")]
    return (
        os.path.exists(p["metrics_j"])
        and len(plots) >= 7
    )


def load_cached_results(ticker: str) -> dict:
    """Load previously computed metrics and plot filenames from disk."""
    sym_info = resolve_symbol(ticker)
    p = get_ticker_paths(ticker)
    with open(p["metrics_j"]) as f:
        metrics = json.load(f)
    plots = sorted(f for f in os.listdir(p["plots_dir"]) if f.endswith(".png"))
    # Try to load forecast if saved
    forecast = []
    for fc_path in [
        p.get("forecast"),
        os.path.join(p["plots_dir"], "forecast.json"),
        os.path.join(p["plots_dir"], "..", "forecast.json"),
    ]:
        if fc_path and os.path.exists(fc_path):
            with open(fc_path) as f:
                forecast = json.load(f)
            break

    # Load metadata if saved, or use fresh resolved info
    meta = sym_info
    if os.path.exists(p.get("meta", "")):
        try:
            with open(p["meta"]) as f:
                saved_meta = json.load(f)
                meta.update(saved_meta)
        except Exception:
            pass

    return {
        "metrics": metrics,
        "plots": plots,
        "forecast": forecast,
        "sym_info": meta,
        "ticker": sym_info["canonical_symbol"],
        "query_ticker": sym_info["query_ticker"],
    }


# ---------------------------------------------------------------------------
# Keras training progress callback
# ---------------------------------------------------------------------------

class _ProgressCB(Callback):
    def __init__(self, store: dict, job_id: str, total_epochs: int):
        super().__init__()
        self._store      = store
        self._job_id     = job_id
        self._total      = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        if self._job_id not in self._store:
            return
        pct = 35 + int(50 * (epoch + 1) / self._total)
        loss = logs.get("loss", 0)
        val  = logs.get("val_loss", 0)
        self._store[self._job_id].update({
            "progress": min(pct, 84),
            "stage": (
                f"Training LSTM — Epoch {epoch + 1}/{self._total} "
                f"| loss: {loss:.5f} | val_loss: {val:.5f}"
            ),
        })


# ---------------------------------------------------------------------------
# Main pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(ticker: str, store: dict, job_id: str) -> None:
    """
    Execute the complete prediction pipeline for `ticker`.
    Progress updates are written to store[job_id] at each stage.
    On success: store[job_id]["status"] = "complete".
    On failure:  store[job_id]["status"] = "error".
    """
    import traceback

    sym_info = resolve_symbol(ticker)
    canonical = sym_info["canonical_symbol"]
    query_ticker = sym_info["query_ticker"]
    cs = sym_info["currency_symbol"]
    cn = sym_info["currency"]
    paths = get_ticker_paths(ticker)

    def _update(stage: str, pct: int) -> None:
        if job_id in store:
            store[job_id].update({"stage": stage, "progress": pct})

    lock = _get_lock(ticker)
    if not lock.acquire(blocking=False):
        store[job_id].update({
            "status": "error",
            "error":  f"Another prediction for {canonical} is already running.",
        })
        return

    try:
        # ── Stage 1: Data ────────────────────────────────────────────────────
        _update(f"Downloading authentic market data for {canonical}...", 5)
        raw_df = download_stock_data(
            ticker=ticker,
            start=cfg.START_DATE,
            end=None,
            force_refresh=True,
            save_path=paths["raw_data"],
        )

        # ── Stage 2: Feature Engineering ─────────────────────────────────────
        _update("Computing technical indicators (SMA, EMA, RSI, MACD, BB)...", 15)
        raw_df    = handle_missing_values(raw_df)
        feat_df   = add_technical_indicators(raw_df)
        feat_cols = get_feature_columns(feat_df)
        tgt_idx   = feat_cols.index(cfg.TARGET_COL)
        n_feat    = len(feat_cols)

        # ── Stage 3: Preprocessing ────────────────────────────────────────────
        _update("Preprocessing: scaling & generating sequences...", 25)
        train_df, test_df = chronological_split(feat_df[feat_cols])

        # Viz 1, 2, 3 (generated before training)
        plot_price_history(raw_df, ticker=canonical, save_dir=paths["plots_dir"],
                           currency_symbol=cs, currency_name=cn)
        plot_train_test_split(train_df, test_df, ticker=canonical, save_dir=paths["plots_dir"],
                             currency_symbol=cs, currency_name=cn)
        plot_feature_correlation(feat_df[feat_cols], save_dir=paths["plots_dir"])

        scaler = fit_scaler(train_df, scaler_path=paths["scaler"])
        tr_scaled, te_scaled = scale_data(train_df, test_df, scaler)

        X_full, y_full = create_sequences(tr_scaled, cfg.SEQUENCE_LENGTH, target_col_idx=tgt_idx)
        X_test, y_test = create_sequences(te_scaled,  cfg.SEQUENCE_LENGTH, target_col_idx=tgt_idx)
        X_tr, X_val, y_tr, y_val = validation_split(X_full, y_full)

        # ── Stage 4: Build Model ──────────────────────────────────────────────
        _update("Building LSTM architecture...", 35)
        set_random_seeds(cfg.RANDOM_SEED)
        model = build_lstm_model(cfg.SEQUENCE_LENGTH, n_feat)

        # ── Stage 5: Train ────────────────────────────────────────────────────
        _update(f"Training LSTM model (up to {cfg.EPOCHS} epochs)...", 37)
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=cfg.PATIENCE,
                          restore_best_weights=True, verbose=0),
            ModelCheckpoint(filepath=paths["model"], monitor="val_loss",
                            save_best_only=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=cfg.LR_FACTOR,
                              patience=cfg.LR_PATIENCE, min_lr=1e-7, verbose=0),
            _ProgressCB(store, job_id, cfg.EPOCHS),
        ]
        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=cfg.EPOCHS,
            batch_size=cfg.BATCH_SIZE,
            callbacks=callbacks,
            shuffle=False,
            verbose=0,
        )
        hist_df = pd.DataFrame(history.history)
        hist_df.to_csv(paths["history"])

        best_epoch = int(hist_df["val_loss"].idxmin()) + 1

        # Viz 4: loss curves
        _update("Generating training history plot...", 86)
        plot_training_history(hist_df, save_dir=paths["plots_dir"])

        # ── Stage 6: Evaluate ─────────────────────────────────────────────────
        _update("Evaluating model on test set...", 88)
        y_pred_sc   = model.predict(X_test, verbose=0)
        y_pred_real = inverse_transform_predictions(y_pred_sc, scaler, n_feat, tgt_idx)
        y_true_real = inverse_transform_predictions(y_test,    scaler, n_feat, tgt_idx)
        test_dates  = test_df.index[cfg.SEQUENCE_LENGTH:]

        metrics, residual_stats = evaluate_model(
            y_true_real, y_pred_real,
            ticker=canonical,
            output_json=paths["metrics_j"],
            output_csv=paths["metrics_c"],
        )

        # Viz 5, 6
        _update("Generating prediction and residual plots...", 91)
        plot_actual_vs_predicted(
            test_dates, y_true_real, y_pred_real, metrics,
            ticker=canonical, save_dir=paths["plots_dir"],
            currency_symbol=cs, currency_name=cn
        )
        plot_residuals(
            test_dates, y_true_real, y_pred_real,
            ticker=canonical, save_dir=paths["plots_dir"],
            currency_symbol=cs
        )

        # ── Stage 7: Future Forecast ──────────────────────────────────────────
        _update(f"Generating {cfg.FORECAST_DAYS}-day future forecast...", 94)
        full_scaled  = np.vstack([tr_scaled, te_scaled])
        last_window  = full_scaled[-cfg.SEQUENCE_LENGTH:].reshape(1, cfg.SEQUENCE_LENGTH, n_feat)
        future_prices = []
        window = last_window.copy()
        for _ in range(cfg.FORECAST_DAYS):
            pred = model.predict(window, verbose=0)[0, 0]
            future_prices.append(pred)
            new_row = window[0, -1, :].copy()
            new_row[tgt_idx] = pred
            window = np.concatenate([window[:, 1:, :],
                                     new_row.reshape(1, 1, n_feat)], axis=1)

        # Inverse-transform forecast
        dummy = np.zeros((len(future_prices), n_feat))
        dummy[:, tgt_idx] = future_prices
        future_prices_real = scaler.inverse_transform(dummy)[:, tgt_idx]

        last_real_date = feat_df.index[-1]
        future_dates   = pd.bdate_range(
            start=last_real_date + pd.Timedelta(days=1),
            periods=cfg.FORECAST_DAYS
        )

        # Viz 7
        plot_future_forecast(
            feat_df.index, feat_df["Close"].values,
            future_dates, future_prices_real,
            residual_stats,
            ticker=canonical, save_dir=paths["plots_dir"],
            currency_symbol=cs, currency_name=cn
        )

        # Save forecast JSON
        forecast = [
            {"date": str(d.date()), "price": round(float(p), 2)}
            for d, p in zip(future_dates, future_prices_real)
        ]
        forecast_path = paths["forecast"]
        with open(forecast_path, "w") as f:
            json.dump(forecast, f, indent=2)

        # Save metadata JSON with live market summary
        sym_info["date_range"] = f"{raw_df.index[0].date()} to {raw_df.index[-1].date()}"
        sym_info["latest_price"] = round(float(raw_df["Close"].iloc[-1]), 2)
        sym_info["trading_days"] = len(raw_df)
        with open(paths["meta"], "w") as f:
            json.dump(sym_info, f, indent=2)

        # ── Done ──────────────────────────────────────────────────────────────
        plots = sorted(fn for fn in os.listdir(paths["plots_dir"]) if fn.endswith(".png"))
        store[job_id].update({
            "status":   "complete",
            "progress": 100,
            "stage":    "Complete!",
            "metrics":  metrics,
            "plots":    plots,
            "forecast": forecast,
            "best_epoch":     best_epoch,
            "training_epochs": len(hist_df),
            "sym_info": sym_info,
            "ticker": canonical,
            "query_ticker": query_ticker,
        })

    except Exception as exc:
        store[job_id].update({
            "status": "error",
            "error":  str(exc),
            "detail": traceback.format_exc(),
        })
    finally:
        lock.release()
