"""
train.py — LSTM model training with full callback suite.

Callbacks used:
  1. EarlyStopping      — Stops training when val_loss stops improving.
                          Restores the best weights automatically.
  2. ModelCheckpoint    — Saves the best model (by val_loss) to disk.
  3. ReduceLROnPlateau  — Halves learning rate when val_loss plateaus.

Random seeds are set for NumPy, TensorFlow, and Python's random module
to ensure reproducible results across runs.
"""

import io
import logging
import os
import random
import sys

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer") and sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

from src.config import (
    EPOCHS, BATCH_SIZE, PATIENCE, LR_PATIENCE, LR_FACTOR,
    MODEL_PATH, HISTORY_PATH, RANDOM_SEED
)

logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = RANDOM_SEED) -> None:
    """
    Set all random seeds for full reproducibility.

    Covers: Python stdlib random, NumPy, TensorFlow (graph-level + op-level).
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"[Train] Random seeds set to {seed}")


def build_callbacks(model_path: str = MODEL_PATH) -> list:
    """
    Construct the Keras callback list for training.

    Returns
    -------
    list of Keras callbacks
    """
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,   # Revert to best epoch on stop
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,         # Only overwrite if val_loss improves
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_FACTOR,            # New LR = LR * factor
            patience=LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
    ]
    return callbacks


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE
) -> tf.keras.callbacks.History:
    """
    Train the LSTM model and return the history object.

    The model is trained on X_train/y_train and validated on X_val/y_val.
    Best weights are restored by EarlyStopping, and the best model is
    persisted by ModelCheckpoint.

    Parameters
    ----------
    model : tf.keras.Model
        Compiled LSTM model from model.py.
    X_train : np.ndarray  shape (n, seq_len, features)
    y_train : np.ndarray  shape (n,)
    X_val   : np.ndarray  shape (m, seq_len, features)
    y_val   : np.ndarray  shape (m,)
    epochs : int
    batch_size : int

    Returns
    -------
    tf.keras.callbacks.History
    """
    set_random_seeds()
    callbacks = build_callbacks()

    print("\n" + "=" * 60)
    print("  [Train] Training LSTM Model")
    print("-" * 60)
    print(f"  Train samples   : {len(X_train):,}")
    print(f"  Val samples     : {len(X_val):,}")
    print(f"  Max epochs      : {epochs}")
    print(f"  Batch size      : {batch_size}")
    print(f"  EarlyStopping   : patience={PATIENCE}")
    print(f"  ReduceLROnPlateau: patience={LR_PATIENCE}, factor={LR_FACTOR}")
    print("=" * 60 + "\n")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=False,      # Never shuffle sequential data
        verbose=1
    )

    # ── Save training history ─────────────────────────────────────────────────
    hist_df = pd.DataFrame(history.history)
    hist_df.index.name = "epoch"
    hist_df.to_csv(HISTORY_PATH)

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best_val_loss = min(history.history["val_loss"])
    total_epochs  = len(history.history["val_loss"])

    print("\n" + "-" * 60)
    print("  [Train] Training complete!")
    print(f"  Best epoch      : {best_epoch} / {total_epochs}")
    print(f"  Best val_loss   : {best_val_loss:.6f}")
    print(f"  Model saved to  : {MODEL_PATH}")
    print(f"  History saved to: {HISTORY_PATH}")
    print("-" * 60 + "\n")

    logger.info(
        f"[Train] Done | best epoch: {best_epoch} | "
        f"best val_loss: {best_val_loss:.6f}"
    )
    return history


def load_trained_model(model_path: str = MODEL_PATH) -> tf.keras.Model:
    """Load a saved .keras model from disk."""
    model = tf.keras.models.load_model(model_path)
    logger.info(f"[Train] Model loaded from {model_path}")
    return model
