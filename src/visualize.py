"""
visualize.py — All project visualizations saved as high-resolution PNG files.

Plots generated:
  1. raw_price_history.png       — Full OHLCV close price history
  2. train_test_split.png        — Train/test split with shaded regions
  3. feature_correlation.png     — Seaborn heatmap of feature correlations
  4. training_loss_curves.png    — Training vs. validation loss per epoch
  5. actual_vs_predicted.png     — Overlaid actual & predicted on test set
  6. residuals.png               — Prediction residuals (errors) over time
  7. future_forecast.png         — Recursive 30-day forward prediction

Style: dark background, vibrant color palette, professional typography.
"""

import io
import logging
import os
import sys

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer") and sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch

from src.config import PLOTS_DIR, TICKER, FORECAST_DAYS, CONFIDENCE_LEVEL

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Global plot style
# ─────────────────────────────────────────────────────────────────────────────
STYLE = {
    "bg"       : "#0d1117",
    "surface"  : "#161b22",
    "text"     : "#e6edf3",
    "subtext"  : "#8b949e",
    "grid"     : "#21262d",
    "actual"   : "#58a6ff",   # Blue  — actual prices
    "pred"     : "#f78166",   # Coral — predicted prices
    "train"    : "#3fb950",   # Green — training data
    "val"      : "#d29922",   # Amber — validation data
    "future"   : "#bc8cff",   # Purple — future forecast
    "loss_tr"  : "#58a6ff",
    "loss_val" : "#f78166",
    "residual" : "#ffa657",
}

DPI = 150


def _base_fig(figsize=(14, 6)):
    """Create a pre-styled dark figure."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=STYLE["bg"])
    ax.set_facecolor(STYLE["surface"])
    ax.tick_params(colors=STYLE["text"], labelsize=9)
    ax.xaxis.label.set_color(STYLE["text"])
    ax.yaxis.label.set_color(STYLE["text"])
    ax.title.set_color(STYLE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE["grid"])
    ax.grid(True, color=STYLE["grid"], linewidth=0.6, linestyle="--", alpha=0.7)
    return fig, ax


def _save(fig, filename: str, save_dir: str = None) -> str:
    """Save figure to save_dir (or PLOTS_DIR) and close it."""
    out_dir = save_dir or PLOTS_DIR
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"[Visualize] Saved -> {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Raw Price History
# ─────────────────────────────────────────────────────────────────────────────

def plot_price_history(
    df: pd.DataFrame,
    ticker: str = TICKER,
    save_dir: str = None,
    currency_symbol: str = "$",
    currency_name: str = "USD"
) -> str:
    """Full historical close price with volume bar overlay."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 8), sharex=True,
        facecolor=STYLE["bg"],
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06}
    )
    for ax in (ax1, ax2):
        ax.set_facecolor(STYLE["surface"])
        ax.tick_params(colors=STYLE["text"], labelsize=9)
        ax.xaxis.label.set_color(STYLE["text"])
        ax.yaxis.label.set_color(STYLE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(STYLE["grid"])
        ax.grid(True, color=STYLE["grid"], linewidth=0.5, linestyle="--", alpha=0.7)

    ax1.plot(df.index, df["Close"], color=STYLE["actual"], linewidth=1.4, label="Close Price")
    ax1.fill_between(df.index, df["Close"], alpha=0.08, color=STYLE["actual"])
    ax1.set_ylabel(f"Price ({currency_name})", color=STYLE["text"], fontsize=11)
    ax1.set_title(
        f"{ticker} -- Historical Close Price",
        color=STYLE["text"], fontsize=14, fontweight="bold", pad=14
    )
    ax1.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{currency_symbol}{x:,.0f}")
    )
    ax1.legend(facecolor=STYLE["surface"], edgecolor=STYLE["grid"],
               labelcolor=STYLE["text"], fontsize=9)

    volume_colors = [STYLE["train"] if df["Close"].iloc[i] >= df["Close"].iloc[i - 1]
                     else STYLE["pred"] for i in range(len(df))]
    ax2.bar(df.index, df["Volume"] / 1e6, color=volume_colors, alpha=0.7, width=1.5)
    ax2.set_ylabel("Volume (M)", color=STYLE["text"], fontsize=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=0, color=STYLE["text"])

    return _save(fig, "01_raw_price_history.png", save_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Train / Test Split
# ─────────────────────────────────────────────────────────────────────────────

def plot_train_test_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ticker: str = TICKER,
    save_dir: str = None,
    currency_symbol: str = "$",
    currency_name: str = "USD"
) -> str:
    fig, ax = _base_fig(figsize=(16, 6))
    ax.plot(train_df.index, train_df["Close"], color=STYLE["train"], linewidth=1.3, label="Training Data")
    ax.plot(test_df.index, test_df["Close"],  color=STYLE["pred"],  linewidth=1.3, label="Test Data")
    ax.axvspan(train_df.index[0], train_df.index[-1], alpha=0.07, color=STYLE["train"])
    ax.axvspan(test_df.index[0],  test_df.index[-1],  alpha=0.07, color=STYLE["pred"])
    ax.axvline(test_df.index[0], color=STYLE["subtext"], linewidth=1.2, linestyle="--", label="Split Boundary")
    ax.set_title(f"{ticker} -- Chronological Train / Test Split",
                 color=STYLE["text"], fontsize=14, fontweight="bold", pad=14)
    ax.set_ylabel(f"Close Price ({currency_name})", color=STYLE["text"], fontsize=11)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{currency_symbol}{x:,.0f}"))
    ax.legend(facecolor=STYLE["surface"], edgecolor=STYLE["grid"], labelcolor=STYLE["text"], fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30, color=STYLE["text"])
    return _save(fig, "02_train_test_split.png", save_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Feature Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────

# Premium curated diverging palette: Electric Cyan -> Pearl Ice -> Radiant Coral -> Vivid Crimson
CORR_CMAP = LinearSegmentedColormap.from_list("neural_correlation", [
    (0.00, "#0891b2"),  # -1.0: Deep Cyan
    (0.25, "#22d3ee"),  # -0.5: Electric Cyan
    (0.42, "#cffafe"),  # -0.15: Light Ice
    (0.50, "#f8fafc"),  #  0.0: Pure Pearl Neutral
    (0.58, "#ffe4e6"),  # +0.15: Soft Blush
    (0.75, "#fb7185"),  # +0.5: Radiant Coral
    (0.90, "#f43f5e"),  # +0.8: Vivid Rose
    (1.00, "#be123c"),  # +1.0: Deep Crimson
])


def plot_feature_correlation(df: pd.DataFrame, save_dir: str = None) -> str:
    """
    Publication-grade, 100% legible feature correlation heatmap with dynamic text contrast.
    Every single cell is completely readable: dark text on light cells, white text on dark cells.
    """
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(14, 11), facecolor=STYLE["bg"])
    ax.set_facecolor(STYLE["bg"])

    norm = Normalize(vmin=-1, vmax=1)
    sns.heatmap(
        corr, ax=ax, cmap=CORR_CMAP, center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f",
        annot_kws={"size": 8.5},
        linewidths=0.7, linecolor=STYLE["bg"], square=True,
        cbar_kws={"shrink": 0.75, "ticks": [-1.0, -0.5, 0.0, 0.5, 1.0]}
    )

    # Automatic adaptive text contrast: calculate background luminance for each cell
    # and guarantee crisp dark text on light cells and crisp white text on dark cells
    for text in ax.texts:
        try:
            val = float(text.get_text())
            rgba = CORR_CMAP(norm(val))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            if luminance > 0.48:
                text.set_color("#090d16")   # Jet dark on light cells
                text.set_fontweight("bold")
            else:
                text.set_color("#ffffff")   # Pure white on saturated cells
                text.set_fontweight("bold")
        except ValueError:
            pass

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=STYLE["text"], labelsize=10)
    cbar.outline.set_edgecolor("#334155")
    cbar.set_label("Pearson Correlation Coefficient", color=STYLE["text"], fontsize=11, labelpad=12)

    ax.set_title("Feature Correlation Heatmap",
                 color=STYLE["text"], fontsize=16, fontweight="bold", pad=18)
    ax.tick_params(colors=STYLE["text"], labelsize=9.5)
    plt.xticks(rotation=45, ha="right", color=STYLE["text"])
    plt.yticks(color=STYLE["text"])
    plt.tight_layout()
    return _save(fig, "03_feature_correlation.png", save_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Training & Validation Loss Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(history_df: pd.DataFrame, save_dir: str = None) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor=STYLE["bg"])
    for ax in (ax1, ax2):
        ax.set_facecolor(STYLE["surface"])
        ax.tick_params(colors=STYLE["text"], labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(STYLE["grid"])
        ax.grid(True, color=STYLE["grid"], linewidth=0.5, linestyle="--", alpha=0.7)
    epochs = range(1, len(history_df) + 1)
    ax1.plot(epochs, history_df["loss"],     color=STYLE["loss_tr"],  linewidth=1.8, label="Train Loss")
    ax1.plot(epochs, history_df["val_loss"], color=STYLE["loss_val"], linewidth=1.8, linestyle="--", label="Val Loss")
    best_ep = int(history_df["val_loss"].idxmin()) + 1
    ax1.axvline(best_ep, color=STYLE["future"], linewidth=1, linestyle=":", label=f"Best Epoch ({best_ep})")
    ax1.set_title("Training vs. Validation Loss (MSE)", color=STYLE["text"], fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", color=STYLE["text"], fontsize=10)
    ax1.set_ylabel("Loss (MSE)", color=STYLE["text"], fontsize=10)
    ax1.legend(facecolor=STYLE["surface"], edgecolor=STYLE["grid"], labelcolor=STYLE["text"], fontsize=9)
    ax2.plot(epochs, history_df["mae"],     color=STYLE["loss_tr"],  linewidth=1.8, label="Train MAE")
    ax2.plot(epochs, history_df["val_mae"], color=STYLE["loss_val"], linewidth=1.8, linestyle="--", label="Val MAE")
    ax2.set_title("Training vs. Validation MAE", color=STYLE["text"], fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", color=STYLE["text"], fontsize=10)
    ax2.set_ylabel("MAE (normalized)", color=STYLE["text"], fontsize=10)
    ax2.legend(facecolor=STYLE["surface"], edgecolor=STYLE["grid"], labelcolor=STYLE["text"], fontsize=9)
    fig.suptitle("LSTM Training History", color=STYLE["text"], fontsize=14, fontweight="bold", y=1.01)
    return _save(fig, "04_training_loss_curves.png", save_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — Actual vs. Predicted
# ─────────────────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict,
    ticker: str = TICKER,
    save_dir: str = None,
    currency_symbol: str = "$",
    currency_name: str = "USD"
) -> str:
    fig, ax = _base_fig(figsize=(16, 7))
    ax.plot(dates, y_true, color=STYLE["actual"], linewidth=1.5, label="Actual Close Price", zorder=3)
    ax.plot(dates, y_pred, color=STYLE["pred"], linewidth=1.5, linestyle="--", label="Predicted Close Price", zorder=4)
    error = np.abs(y_true - y_pred)
    ax.fill_between(dates, y_pred - error, y_pred + error, alpha=0.15, color=STYLE["pred"], label="Error Band")
    ax.set_title(
        f"{ticker} -- Actual vs. Predicted Close Price (Test Set)\n"
        f"RMSE: {currency_symbol}{metrics['RMSE']:.2f} | MAE: {currency_symbol}{metrics['MAE']:.2f} | "
        f"MAPE: {metrics['MAPE']:.2f}% | R2: {metrics['R2']:.4f} | DA: {metrics['DA']:.1f}%",
        color=STYLE["text"], fontsize=12, fontweight="bold", pad=12
    )
    ax.set_ylabel(f"Price ({currency_name})", color=STYLE["text"], fontsize=11)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{currency_symbol}{x:,.0f}"))
    ax.legend(facecolor=STYLE["surface"], edgecolor=STYLE["grid"], labelcolor=STYLE["text"], fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30, color=STYLE["text"])
    return _save(fig, "05_actual_vs_predicted.png", save_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6 — Residuals
# ─────────────────────────────────────────────────────────────────────────────

def plot_residuals(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ticker: str = TICKER,
    save_dir: str = None,
    currency_symbol: str = "$"
) -> str:
    residuals = y_true - y_pred
    cs_label = currency_symbol.strip()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor=STYLE["bg"])
    for ax in (ax1, ax2):
        ax.set_facecolor(STYLE["surface"])
        ax.tick_params(colors=STYLE["text"], labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(STYLE["grid"])
        ax.grid(True, color=STYLE["grid"], linewidth=0.5, linestyle="--", alpha=0.7)
    ax1.bar(dates, residuals,
            color=[STYLE["train"] if r >= 0 else STYLE["pred"] for r in residuals],
            alpha=0.7, width=1.5)
    ax1.axhline(0, color=STYLE["text"], linewidth=0.8, linestyle="--")
    ax1.set_title(f"{ticker} -- Residuals Over Time", color=STYLE["text"], fontsize=12, fontweight="bold")
    ax1.set_ylabel(f"Actual - Predicted ({cs_label})", color=STYLE["text"], fontsize=10)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, color=STYLE["text"])
    ax2.hist(residuals, bins=40, color=STYLE["residual"], edgecolor=STYLE["bg"], alpha=0.85)
    ax2.axvline(0, color=STYLE["text"], linewidth=1, linestyle="--")
    ax2.axvline(np.mean(residuals), color=STYLE["future"], linewidth=1.2,
                linestyle="-", label=f"Mean: {currency_symbol}{np.mean(residuals):.2f}")
    ax2.set_title("Residuals Distribution", color=STYLE["text"], fontsize=12, fontweight="bold")
    ax2.set_xlabel(f"Residual ({cs_label})", color=STYLE["text"], fontsize=10)
    ax2.set_ylabel("Frequency", color=STYLE["text"], fontsize=10)
    ax2.legend(facecolor=STYLE["surface"], edgecolor=STYLE["grid"], labelcolor=STYLE["text"], fontsize=9)
    fig.suptitle("Prediction Residual Analysis", color=STYLE["text"], fontsize=14, fontweight="bold", y=1.01)
    return _save(fig, "06_residuals.png", save_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 7 — Future Forecast
# ─────────────────────────────────────────────────────────────────────────────

def plot_future_forecast(
    historical_dates: pd.DatetimeIndex,
    historical_prices: np.ndarray,
    forecast_dates: pd.DatetimeIndex,
    forecast_prices: np.ndarray,
    residual_stats: dict,
    ticker: str = TICKER,
    n_context: int = 120,
    save_dir: str = None,
    currency_symbol: str = "$",
    currency_name: str = "USD"
) -> str:
    """Plot the last n_context days of history + the recursive future forecast.

    Uncertainty bands are STATISTICALLY DERIVED from test-set residuals (not
    synthetic or hardcoded percentages). Raises ValueError if residual_stats
    is None to guarantee no fake uncertainty values ever appear on a plot.
    """
    if residual_stats is None or "ci_width" not in residual_stats:
        raise ValueError(
            "residual_stats with 'ci_width' key is REQUIRED for plot_future_forecast. "
            "Uncertainty bands must be derived from actual model evaluation residuals — "
            "synthetic/hardcoded percentage bands (e.g. ±2%) are strictly forbidden."
        )

    ci_width = float(residual_stats["ci_width"])
    conf_level = float(residual_stats.get("confidence_level", CONFIDENCE_LEVEL))
    cs_label = currency_symbol.strip() if currency_symbol.strip() else "$"

    fig, ax = _base_fig(figsize=(16, 7))
    ctx_dates  = historical_dates[-n_context:]
    ctx_prices = historical_prices[-n_context:]
    ax.plot(ctx_dates, ctx_prices, color=STYLE["actual"], linewidth=1.5, label="Historical Close")
    ax.fill_between(ctx_dates, ctx_prices, alpha=0.07, color=STYLE["actual"])
    ax.plot(forecast_dates, forecast_prices, color=STYLE["future"], linewidth=2.0,
            linestyle="--", marker="o", markersize=3, label=f"{FORECAST_DAYS}-Day Forecast")
    lower = forecast_prices - ci_width
    upper = forecast_prices + ci_width
    band_label = f"±{cs_label}{ci_width:.2f}  {conf_level:.0%} CI (test residuals, {residual_stats.get('z_score', '?')}σ)"
    ax.fill_between(forecast_dates, lower, upper, alpha=0.20, color=STYLE["future"], label=band_label)
    ax.axvline(forecast_dates[0], color=STYLE["subtext"], linewidth=1.0, linestyle=":", label="Forecast Start")
    ax.set_title(f"{ticker} -- {FORECAST_DAYS}-Day Future Price Forecast",
                 color=STYLE["text"], fontsize=14, fontweight="bold", pad=14)
    ax.set_ylabel(f"Price ({currency_name})", color=STYLE["text"], fontsize=11)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{currency_symbol}{x:,.0f}"))
    ax.legend(facecolor=STYLE["surface"], edgecolor=STYLE["grid"], labelcolor=STYLE["text"], fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.xticks(rotation=30, color=STYLE["text"])
    return _save(fig, "07_future_forecast.png", save_dir)


# --  Convenience: print saved paths  ----------------------------------------

def print_plot_summary(plots_dir: str = None) -> None:
    out = plots_dir or PLOTS_DIR
    print("\n" + "=" * 60)
    print("  All Visualizations Saved")
    print("=" * 60)
    for f in sorted(os.listdir(out)):
        if f.endswith(".png"):
            print(f"  * {f}")
    print(f"\n  Directory: {out}")
    print("=" * 60 + "\n")
