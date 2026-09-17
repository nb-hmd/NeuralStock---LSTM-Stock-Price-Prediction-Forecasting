"""
feature_engineering.py — Computes technical indicators from raw OHLCV data.

All indicators are computed using the `ta` library (Technical Analysis Library
in Python). Indicators are appended as extra columns before scaling so the
LSTM receives a richer multivariate input beyond raw OHLCV.

Indicators computed:
  - SMA_20  : 20-day Simple Moving Average of Close
  - EMA_20  : 20-day Exponential Moving Average of Close
  - RSI_14  : 14-period Relative Strength Index
  - MACD    : MACD line (12/26 EMA difference)
  - MACD_Signal : MACD signal line (9-period EMA of MACD)
  - MACD_Hist   : MACD histogram
  - BB_High : Bollinger Band upper band
  - BB_Low  : Bollinger Band lower band
  - BB_Mid  : Bollinger Band middle (SMA)
  - BB_Width: Bollinger Band width (volatility proxy)
"""

import io
import logging
import sys

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer") and sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import ta

from src.config import (
    SMA_WINDOW, EMA_WINDOW, RSI_WINDOW,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_WINDOW, BB_STD
)

logger = logging.getLogger(__name__)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append technical indicator columns to the OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Clean OHLCV DataFrame with columns:
        ['Open', 'High', 'Low', 'Close', 'Volume'].

    Returns
    -------
    pd.DataFrame
        Original DataFrame plus the following new columns:
        SMA_20, EMA_20, RSI_14, MACD, MACD_Signal, MACD_Hist,
        BB_High, BB_Low, BB_Mid, BB_Width.

    Notes
    -----
    The first N rows will have NaN indicators (look-back period). These are
    dropped AFTER computing indicators to avoid using partial-window values.
    """
    df = df.copy()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    # ── Moving Averages ───────────────────────────────────────────────────────
    df[f"SMA_{SMA_WINDOW}"] = ta.trend.sma_indicator(close, window=SMA_WINDOW)
    df[f"EMA_{EMA_WINDOW}"] = ta.trend.ema_indicator(close, window=EMA_WINDOW)

    # ── RSI ───────────────────────────────────────────────────────────────────
    df[f"RSI_{RSI_WINDOW}"] = ta.momentum.rsi(close, window=RSI_WINDOW)

    # ── MACD ──────────────────────────────────────────────────────────────────
    macd_obj = ta.trend.MACD(
        close,
        window_fast=MACD_FAST,
        window_slow=MACD_SLOW,
        window_sign=MACD_SIGNAL
    )
    df["MACD"]        = macd_obj.macd()
    df["MACD_Signal"] = macd_obj.macd_signal()
    df["MACD_Hist"]   = macd_obj.macd_diff()

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_obj = ta.volatility.BollingerBands(
        close, window=BB_WINDOW, window_dev=BB_STD
    )
    df["BB_High"]  = bb_obj.bollinger_hband()
    df["BB_Low"]   = bb_obj.bollinger_lband()
    df["BB_Mid"]   = bb_obj.bollinger_mavg()
    df["BB_Width"] = bb_obj.bollinger_wband()   # (High - Low) / Mid

    # ── Drop rows with NaN (initial look-back period) ─────────────────────────
    n_before = len(df)
    df.dropna(inplace=True)
    n_dropped = n_before - len(df)

    logger.info(
        f"[FeatureEngineering] Added 10 indicators | "
        f"Dropped {n_dropped} NaN rows (look-back period) | "
        f"Remaining rows: {len(df):,}"
    )

    _log_feature_summary(df)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the ordered list of feature column names in the DataFrame.

    This is the ground-truth column order used by the scaler and the
    sequence generator — always derive this from the actual DataFrame
    rather than hardcoding.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    list[str]
    """
    # Ensure these base columns come first, then all computed indicators
    base_cols  = ["Open", "High", "Low", "Close", "Volume"]
    extra_cols = [c for c in df.columns if c not in base_cols]
    ordered    = [c for c in base_cols if c in df.columns] + extra_cols
    return ordered


def _log_feature_summary(df: pd.DataFrame) -> None:
    """Print a compact feature summary."""
    print("\n" + "-" * 60)
    print("  [Features] Feature Engineering Summary")
    print("-" * 60)
    print(f"  Total columns : {len(df.columns)}")
    print(f"  Columns       : {list(df.columns)}")
    print(f"  Data shape    : {df.shape}")
    print("-" * 60 + "\n")
