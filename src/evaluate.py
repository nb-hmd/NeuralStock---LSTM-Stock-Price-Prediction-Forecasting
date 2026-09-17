"""
evaluate.py — Model evaluation on real-scale (inverse-transformed) predictions.

Metrics computed:
  - RMSE  : Root Mean Squared Error       — primary metric, in $ units
  - MAE   : Mean Absolute Error           — average absolute $ error
  - MAPE  : Mean Absolute Percentage Error — % error relative to actual price
  - R²    : Coefficient of Determination  — variance explained (0–1)
  - DA    : Directional Accuracy          — % of days with correct trend direction

All metrics are computed on INVERSE-TRANSFORMED predictions so they are
expressed in real dollar values, not normalized [0, 1] space.
"""

import io
import json
import logging
import sys

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer") and sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.config import METRICS_JSON, METRICS_CSV, CONFIDENCE_LEVEL

_ZSCORE_TABLE = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
_confidence_z = _ZSCORE_TABLE.get(round(CONFIDENCE_LEVEL, 2), 1.96)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Individual Metric Functions
# ─────────────────────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error (same unit as price, e.g. $)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (same unit as price)."""
    return float(mean_absolute_error(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error (%).

    Avoids division by zero by skipping samples where y_true == 0.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero = y_true != 0
    if not nonzero.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination R² (closer to 1 is better)."""
    return float(r2_score(y_true, y_pred))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Directional Accuracy (DA) — percentage of days where the predicted
    price movement direction matches the actual direction.

    DA = 100 * mean( sign(Δy_true) == sign(Δy_pred) )

    Parameters
    ----------
    y_true, y_pred : 1D arrays of real prices (chronological order).

    Returns
    -------
    float — percentage (0–100).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    delta_true = np.diff(y_true)
    delta_pred = np.diff(y_pred)
    correct = np.sign(delta_true) == np.sign(delta_pred)
    return float(np.mean(correct) * 100)


# ─────────────────────────────────────────────────────────────────────────────
# Full Evaluation Suite
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ticker: str = "AAPL",
    split: str = "test",
    output_json: str = None,
    output_csv: str = None,
) -> tuple[dict, dict]:
    """
    Compute and persist all evaluation metrics plus residual-derived uncertainty statistics.

    Parameters
    ----------
    y_true : np.ndarray  Real closing prices (inverse-transformed, $).
    y_pred : np.ndarray  Predicted closing prices (inverse-transformed, $).
    ticker : str         Stock ticker (for labelling saved files).
    split  : str         'test' or 'train' (logged in the report).
    output_json : str, optional  Override JSON save path.
    output_csv  : str, optional  Override CSV save path.

    Returns
    -------
    (metrics, residual_stats) : tuple of dicts
      metrics        = keys: ticker, split, n_samples, RMSE, MAE, MAPE, R2, DA
      residual_stats = keys: mean, std, ci_width, confidence_level, z_score
    """
    json_path = output_json or METRICS_JSON
    csv_path  = output_csv  or METRICS_CSV

    y_true_arr = np.array(y_true, dtype=np.float64)
    y_pred_arr = np.array(y_pred, dtype=np.float64)
    residuals  = y_true_arr - y_pred_arr

    metrics = {
        "ticker"  : ticker,
        "split"   : split,
        "n_samples": int(len(y_true_arr)),
        "RMSE"    : round(rmse(y_true_arr, y_pred_arr),   4),
        "MAE"     : round(mae(y_true_arr, y_pred_arr),    4),
        "MAPE"    : round(mape(y_true_arr, y_pred_arr),   4),
        "R2"      : round(r_squared(y_true_arr, y_pred_arr), 4),
        "DA"      : round(directional_accuracy(y_true_arr, y_pred_arr), 2),
    }

    residual_stats = {
        "mean":             round(float(np.mean(residuals)), 6),
        "std":              round(float(np.std(residuals)), 6),
        "ci_width":         round(float(_confidence_z * np.std(residuals)), 6),
        "confidence_level": CONFIDENCE_LEVEL,
        "z_score":          _confidence_z,
        "min_residual":     round(float(np.min(residuals)), 6),
        "max_residual":     round(float(np.max(residuals)), 6),
        "median_residual":  round(float(np.median(residuals)), 6),
    }
    metrics["residual_stats"] = residual_stats

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    flat = {k: v for k, v in metrics.items() if k != "residual_stats"}
    flat.update({f"residual_{k}": v for k, v in residual_stats.items()})
    pd.DataFrame([flat]).to_csv(csv_path, index=False)

    print("\n" + "=" * 60)
    print(f"  Model Evaluation -- {ticker} ({split} set)")
    print("=" * 60)
    print(f"  Samples          : {metrics['n_samples']:,}")
    print(f"  RMSE             : ${metrics['RMSE']:.4f}")
    print(f"  MAE              : ${metrics['MAE']:.4f}")
    print(f"  MAPE             : {metrics['MAPE']:.4f}%")
    print(f"  R2               : {metrics['R2']:.4f}")
    print(f"  Directional Acc. : {metrics['DA']:.2f}%")
    print(f"  --- Residual-Derived Uncertainty ({CONFIDENCE_LEVEL:.0%}) ---")
    print(f"  Residual Mean    : ${residual_stats['mean']:.4f}")
    print(f"  Residual Std (σ) : ${residual_stats['std']:.4f}")
    print(f"  {_confidence_z}σ CI ± Width    : ${residual_stats['ci_width']:.4f}")
    print("-" * 60)
    print(f"  Saved -> {json_path}")
    print(f"  Saved -> {csv_path}")
    print("=" * 60 + "\n")

    logger.info(
        f"[Evaluate] {ticker} | RMSE=${metrics['RMSE']:.2f} | "
        f"MAE=${metrics['MAE']:.2f} | MAPE={metrics['MAPE']:.2f}% | "
        f"R2={metrics['R2']:.4f} | DA={metrics['DA']:.1f}% | "
        f"σ=${residual_stats['std']:.2f}"
    )
    return metrics, residual_stats
