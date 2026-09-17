"""
preprocessing.py — Data cleaning, normalization, and sequence generation.

Key design decisions (all required for a valid time-series ML pipeline):
  1. Forward-fill only (no backward-fill) to avoid look-ahead bias.
  2. MinMaxScaler is FIT on training data only — never on test data.
  3. Train/test split is CHRONOLOGICAL (no shuffling).
  4. Sequences are generated with a sliding window AFTER the split.

All functions return documented numpy arrays or pandas objects.
"""

import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    TARGET_COL, SEQUENCE_LENGTH, TEST_SPLIT,
    SCALER_PATH, VALIDATION_SPLIT
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Missing Value Handling
# ─────────────────────────────────────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill missing values then drop any remaining NaN rows.

    Forward-fill is used (not backward-fill) so each gap is filled with the
    most recent known price — this avoids using future information to fill
    past gaps (look-ahead bias).

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame (time-indexed, ascending).

    Returns
    -------
    pd.DataFrame
        DataFrame with no NaN values.
    """
    n_before = df.isna().sum().sum()
    df = df.ffill()           # Forward-fill first
    n_after_ffill = df.isna().sum().sum()
    df = df.dropna()          # Drop any remaining (e.g. NaN at very start)
    n_dropped = n_after_ffill

    logger.info(
        f"[Preprocessing] Missing values: {n_before} total | "
        f"forward-filled, {n_dropped} rows dropped."
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Chronological Train / Test Split
# ─────────────────────────────────────────────────────────────────────────────

def chronological_split(
    df: pd.DataFrame,
    test_split: float = TEST_SPLIT
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train and test sets by time (no shuffling).

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned DataFrame, ascending time index.
    test_split : float
        Fraction reserved for test (e.g. 0.20 → last 20%).

    Returns
    -------
    (train_df, test_df) : tuple of DataFrames
    """
    split_idx = int(len(df) * (1 - test_split))
    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()

    logger.info(
        f"[Preprocessing] Chronological split | "
        f"Train: {len(train_df):,} rows "
        f"({train_df.index[0].date()} → {train_df.index[-1].date()}) | "
        f"Test:  {len(test_df):,} rows "
        f"({test_df.index[0].date()} → {test_df.index[-1].date()})"
    )
    return train_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature Scaling
# ─────────────────────────────────────────────────────────────────────────────

def fit_scaler(train_df: pd.DataFrame, scaler_path: str = None) -> MinMaxScaler:
    """
    Fit a MinMaxScaler on training data ONLY.

    The scaler is serialized to disk so the same transformation can be applied
    during inference on unseen data and reversed for evaluation.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training subset of the full dataset.
    scaler_path : str, optional
        Override save path. Defaults to SCALER_PATH from config.

    Returns
    -------
    MinMaxScaler
        Fitted scaler (fit on training data only).
    """
    save_path = scaler_path or SCALER_PATH
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df)
    # Save scaler for later inverse-transform
    with open(save_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"[Preprocessing] Scaler fitted on train data and saved to {save_path}")
    return scaler


def load_scaler(scaler_path: str = None) -> MinMaxScaler:
    """Load a previously saved scaler from disk."""
    path = scaler_path or SCALER_PATH
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    logger.info(f"[Preprocessing] Scaler loaded from {path}")
    return scaler


def scale_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaler: MinMaxScaler
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply MinMaxScaler to both train and test sets.

    IMPORTANT: The scaler was fit on training data only; here we apply the
    same transformation to both sets to ensure test data is scaled without
    leaking information from the test period.

    Returns
    -------
    (train_scaled, test_scaled) : np.ndarray with shape (n, num_features)
    """
    train_scaled = scaler.transform(train_df)
    test_scaled  = scaler.transform(test_df)
    logger.info(
        f"[Preprocessing] Data scaled | "
        f"Train shape: {train_scaled.shape} | Test shape: {test_scaled.shape}"
    )
    return train_scaled, test_scaled


# ─────────────────────────────────────────────────────────────────────────────
# 4. Sequence / Sliding Window Generation
# ─────────────────────────────────────────────────────────────────────────────

def create_sequences(
    data: np.ndarray,
    sequence_length: int = SEQUENCE_LENGTH,
    target_col_idx: int = None,
    all_feature_cols: list = None,
    target_col_name: str = TARGET_COL
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a scaled 2D array into (X, y) sequence pairs using a sliding window.

    For each position i:
      X[i] = data[i : i + sequence_length]          → shape (seq_len, features)
      y[i] = data[i + sequence_length, target_idx]  → next-step closing price

    This creates a 3D tensor for X: (samples, timesteps, features).

    Parameters
    ----------
    data : np.ndarray
        Scaled 2D array of shape (n_samples, n_features).
    sequence_length : int
        Number of past timesteps per input sample.
    target_col_idx : int or None
        Column index of the target (Close price). If None, auto-detected.
    all_feature_cols : list or None
        List of feature column names (used for auto-detection).
    target_col_name : str
        Name of the target column (default: 'Close').

    Returns
    -------
    X : np.ndarray  shape (n_sequences, sequence_length, n_features)
    y : np.ndarray  shape (n_sequences,)
    """
    if target_col_idx is None:
        if all_feature_cols is None:
            raise ValueError(
                "Provide either target_col_idx or all_feature_cols."
            )
        target_col_idx = all_feature_cols.index(target_col_name)

    X, y = [], []
    n = len(data)
    for i in range(n - sequence_length):
        X.append(data[i : i + sequence_length])
        y.append(data[i + sequence_length, target_col_idx])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    logger.info(
        f"[Preprocessing] Sequences created | X: {X.shape} | y: {y.shape} | "
        f"target_col_idx={target_col_idx}"
    )
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 5. Chronological Validation Split (from training sequences)
# ─────────────────────────────────────────────────────────────────────────────

def validation_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    val_fraction: float = VALIDATION_SPLIT
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split training sequences into train/validation sets chronologically.

    Keras's built-in validation_split also does this correctly for sequential
    data (it takes the last val_fraction of the data). We replicate it here
    for explicit control and logging.

    Returns
    -------
    X_tr, X_val, y_tr, y_val
    """
    val_idx = int(len(X_train) * (1 - val_fraction))
    X_tr  = X_train[:val_idx]
    X_val = X_train[val_idx:]
    y_tr  = y_train[:val_idx]
    y_val = y_train[val_idx:]
    logger.info(
        f"[Preprocessing] Train/Val split | "
        f"Train samples: {len(X_tr):,} | Val samples: {len(X_val):,}"
    )
    return X_tr, X_val, y_tr, y_val


# ─────────────────────────────────────────────────────────────────────────────
# 6. Inverse Transform (for evaluation in real price scale)
# ─────────────────────────────────────────────────────────────────────────────

def inverse_transform_predictions(
    predictions: np.ndarray,
    scaler: MinMaxScaler,
    n_features: int,
    target_col_idx: int
) -> np.ndarray:
    """
    Reverse the MinMax normalization to recover real price values.

    Since the scaler was fit on all features simultaneously, we reconstruct
    a dummy array of the same width, place predictions in the target column,
    and inverse-transform the whole array — then extract only the target column.

    Parameters
    ----------
    predictions : np.ndarray  shape (n,) or (n, 1)
    scaler : MinMaxScaler      The scaler that was fit on training data.
    n_features : int           Total number of feature columns.
    target_col_idx : int       Index of the Close-price column.

    Returns
    -------
    np.ndarray  shape (n,)  — prices in original dollar scale.
    """
    predictions = predictions.flatten()
    dummy = np.zeros((len(predictions), n_features), dtype=np.float32)
    dummy[:, target_col_idx] = predictions
    inv = scaler.inverse_transform(dummy)
    return inv[:, target_col_idx]
