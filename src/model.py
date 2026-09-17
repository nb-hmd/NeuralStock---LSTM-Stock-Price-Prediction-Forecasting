"""
model.py — Stacked LSTM neural network architecture.

Architecture:
  Input(seq_len, n_features)
    → LSTM(128, return_sequences=True)
    → Dropout(0.2)
    → LSTM(64, return_sequences=False)
    → Dropout(0.2)
    → Dense(32, relu)
    → Dense(1, linear)     ← predicts next-day closing price

Compiled with Adam optimizer and MSE loss (standard for regression).
"""

import io
import logging
import sys

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer") and sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input
)
from tensorflow.keras.optimizers import Adam

from src.config import (
    LSTM_UNITS, DROPOUT_RATE, DENSE_UNITS,
    LEARNING_RATE, SEQUENCE_LENGTH
)

logger = logging.getLogger(__name__)


def build_lstm_model(
    sequence_length: int,
    n_features: int,
    lstm_units: list = None,
    dropout_rate: float = None,
    dense_units: int = None,
    learning_rate: float = None
) -> tf.keras.Model:
    """
    Construct and compile a stacked LSTM model for next-step regression.

    Parameters
    ----------
    sequence_length : int
        Number of timesteps per input sample (e.g. 60).
    n_features : int
        Number of input features per timestep (e.g. 15 for OHLCV + indicators).
    lstm_units : list of int
        Number of units in each stacked LSTM layer. Default: [128, 64].
    dropout_rate : float
        Dropout rate after each LSTM layer (regularization). Default: 0.2.
    dense_units : int
        Units in the intermediate Dense layer before output. Default: 32.
    learning_rate : float
        Adam optimizer learning rate. Default: 0.001.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model ready for training.
    """
    # Use config defaults if not overridden
    lstm_units    = lstm_units    or LSTM_UNITS
    dropout_rate  = dropout_rate  if dropout_rate is not None else DROPOUT_RATE
    dense_units   = dense_units   or DENSE_UNITS
    learning_rate = learning_rate or LEARNING_RATE

    model = Sequential(name="LSTM_StockPredictor")

    # ── Input Layer ───────────────────────────────────────────────────────────
    model.add(Input(shape=(sequence_length, n_features)))

    # ── First LSTM Layer ──────────────────────────────────────────────────────
    # return_sequences=True: passes full sequence to the next LSTM layer
    model.add(LSTM(
        units=lstm_units[0],
        return_sequences=True,
        name="LSTM_1"
    ))
    model.add(Dropout(dropout_rate, name="Dropout_1"))

    # ── Second LSTM Layer ─────────────────────────────────────────────────────
    # return_sequences=False: returns only the last timestep output
    model.add(LSTM(
        units=lstm_units[1],
        return_sequences=False,
        name="LSTM_2"
    ))
    model.add(Dropout(dropout_rate, name="Dropout_2"))

    # ── Dense Layers ──────────────────────────────────────────────────────────
    model.add(Dense(dense_units, activation="relu", name="Dense_hidden"))
    model.add(Dense(1, activation="linear", name="Output"))   # regression output

    # ── Compile ───────────────────────────────────────────────────────────────
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="mse",             # Mean Squared Error — standard for regression
        metrics=["mae"]         # Track Mean Absolute Error during training
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  [Model] LSTM Model Architecture")
    print("=" * 60)
    model.summary()
    print("=" * 60 + "\n")

    total_params = model.count_params()
    logger.info(
        f"[Model] Built LSTM model | "
        f"Input: ({sequence_length}, {n_features}) | "
        f"LSTM layers: {lstm_units} | "
        f"Total params: {total_params:,}"
    )

    return model
