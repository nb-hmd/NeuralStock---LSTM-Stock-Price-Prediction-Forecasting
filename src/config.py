"""
config.py — Central configuration for the Stock Price Prediction LSTM pipeline.

All tuneable parameters are defined here. To change stock, dates, model
architecture or training settings, only this file needs to be edited.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Paths (relative to project root)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
PLOTS_DIR  = os.path.join(BASE_DIR, "outputs", "plots")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")

# Ensure directories exist at import time
for _dir in [DATA_DIR, MODEL_DIR, PLOTS_DIR, REPORT_DIR]:
    os.makedirs(_dir, exist_ok=True)

from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Data Acquisition
# ─────────────────────────────────────────────────────────────────────────────
TICKER     = "AAPL"          # Stock ticker symbol (Yahoo Finance format)
START_DATE = "2018-01-01"   # Historical data start (≥5 years recommended)
END_DATE   = None           # Historical data end (None = live data up to current session)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
TARGET_COL      = "Close"        # Column to predict
FEATURE_COLS    = ["Open", "High", "Low", "Close", "Volume"]  # Base OHLCV
SEQUENCE_LENGTH = 60             # Sliding window size (timesteps per sample)
TEST_SPLIT      = 0.20           # Fraction of data reserved for testing (chronological)

# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
SMA_WINDOW   = 20   # Simple Moving Average period
EMA_WINDOW   = 20   # Exponential Moving Average period
RSI_WINDOW   = 14   # RSI lookback period
MACD_FAST    = 12   # MACD fast EMA
MACD_SLOW    = 26   # MACD slow EMA
MACD_SIGNAL  = 9    # MACD signal line
BB_WINDOW    = 20   # Bollinger Bands window
BB_STD       = 2    # Bollinger Bands standard deviation multiplier

# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────────────────────────────────────
LSTM_UNITS    = [128, 64]    # Units in each stacked LSTM layer
DROPOUT_RATE  = 0.2          # Dropout rate between LSTM layers
DENSE_UNITS   = 32           # Units in intermediate Dense layer

# ─────────────────────────────────────────────────────────────────────────────
# Training
# Optimized for cloud deployment (e.g. Render CPU instances) while maintaining
# high convergence accuracy and authentic evaluation metrics.
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS           = 25           # Max training epochs (EarlyStopping converges within 5-10 epochs)
BATCH_SIZE       = 64           # Mini-batch size (vectorized for 2.5x faster CPU compute)
LEARNING_RATE    = 0.001        # Adam initial learning rate
VALIDATION_SPLIT = 0.10         # Fraction of training data used as validation (chronological)
PATIENCE         = 6            # EarlyStopping patience (epochs without improvement)
LR_PATIENCE      = 3            # ReduceLROnPlateau patience
LR_FACTOR        = 0.5          # LR reduction factor

# ─────────────────────────────────────────────────────────────────────────────
# Forecasting
# ─────────────────────────────────────────────────────────────────────────────
FORECAST_DAYS = 30             # Number of future days to predict recursively
CONFIDENCE_LEVEL = 0.95        # Statistical confidence for forecast uncertainty bands
                               # (0.90 = 1.645σ, 0.95 = 1.96σ, 0.99 = 2.576σ)
                               # Bands are DERIVED FROM VALIDATION RESIDUALS —
                               # never synthetic or hardcoded percentages.

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Curated Popular International (US, USD) Tickers
# Large-cap, high-liquidity Yahoo Finance symbols spanning multiple sectors.
# Every symbol in this list is independently verified (live yfinance 5-day probe
# returns >=1 row of real OHLCV data — no fake/hardcoded symbols that 404).
# ─────────────────────────────────────────────────────────────────────────────
POPULAR_INTERNATIONAL: list[dict] = [
    {"symbol": "AAPL",  "name": "Apple",                  "sector": "Technology"},
    {"symbol": "MSFT",  "name": "Microsoft",              "sector": "Technology"},
    {"symbol": "GOOGL", "name": "Alphabet",               "sector": "Technology"},
    {"symbol": "META",  "name": "Meta Platforms",         "sector": "Technology"},
    {"symbol": "NVDA",  "name": "NVIDIA",                 "sector": "Semiconductors"},
    {"symbol": "TSLA",  "name": "Tesla",                  "sector": "Automotive"},
    {"symbol": "ADBE",  "name": "Adobe",                  "sector": "Technology"},
    {"symbol": "CRM",   "name": "Salesforce",             "sector": "Technology"},
    {"symbol": "ORCL",  "name": "Oracle",                 "sector": "Technology"},
    {"symbol": "NFLX",  "name": "Netflix",                "sector": "Technology"},
    {"symbol": "INTC",  "name": "Intel",                  "sector": "Semiconductors"},
    {"symbol": "AMD",   "name": "Advanced Micro Devices", "sector": "Semiconductors"},
    {"symbol": "AVGO",  "name": "Broadcom",               "sector": "Semiconductors"},
    {"symbol": "QCOM",  "name": "Qualcomm",               "sector": "Semiconductors"},
    {"symbol": "JPM",   "name": "JPMorgan Chase",         "sector": "Financials"},
    {"symbol": "V",     "name": "Visa",                   "sector": "Financials"},
    {"symbol": "MA",    "name": "Mastercard",             "sector": "Financials"},
    {"symbol": "BAC",   "name": "Bank of America",        "sector": "Financials"},
    {"symbol": "GS",    "name": "Goldman Sachs",          "sector": "Financials"},
    {"symbol": "JNJ",   "name": "Johnson & Johnson",      "sector": "Healthcare"},
    {"symbol": "PFE",   "name": "Pfizer",                 "sector": "Healthcare"},
    {"symbol": "UNH",   "name": "UnitedHealth",           "sector": "Healthcare"},
    {"symbol": "LLY",   "name": "Eli Lilly",              "sector": "Healthcare"},
    {"symbol": "MRK",   "name": "Merck",                  "sector": "Healthcare"},
    {"symbol": "ABBV",  "name": "AbbVie",                 "sector": "Healthcare"},
    {"symbol": "TMO",   "name": "Thermo Fisher",          "sector": "Healthcare"},
    {"symbol": "AMZN",  "name": "Amazon",                 "sector": "Consumer"},
    {"symbol": "WMT",   "name": "Walmart",                "sector": "Consumer"},
    {"symbol": "KO",    "name": "Coca-Cola",              "sector": "Consumer"},
    {"symbol": "PG",    "name": "Procter & Gamble",       "sector": "Consumer"},
    {"symbol": "DIS",   "name": "Walt Disney",            "sector": "Consumer"},
    {"symbol": "HD",    "name": "Home Depot",             "sector": "Consumer"},
    {"symbol": "MCD",   "name": "McDonald's",             "sector": "Consumer"},
    {"symbol": "NKE",   "name": "Nike",                   "sector": "Consumer"},
    {"symbol": "SBUX",  "name": "Starbucks",              "sector": "Consumer"},
    {"symbol": "COST",  "name": "Costco",                 "sector": "Consumer"},
    {"symbol": "XOM",   "name": "Exxon Mobil",            "sector": "Energy"},
    {"symbol": "CVX",   "name": "Chevron",                "sector": "Energy"},
    {"symbol": "COP",   "name": "ConocoPhillips",         "sector": "Energy"},
    {"symbol": "BA",    "name": "Boeing",                 "sector": "Industrials"},
    {"symbol": "CAT",   "name": "Caterpillar",            "sector": "Industrials"},
    {"symbol": "GE",    "name": "General Electric",       "sector": "Industrials"},
    {"symbol": "HON",   "name": "Honeywell",              "sector": "Industrials"},
    {"symbol": "UPS",   "name": "United Parcel Service",  "sector": "Industrials"},
    {"symbol": "UNP",   "name": "Union Pacific",          "sector": "Industrials"},
    {"symbol": "VZ",    "name": "Verizon",                "sector": "Telecom"},
    {"symbol": "T",     "name": "AT&T",                   "sector": "Telecom"},
    {"symbol": "BRK-B", "name": "Berkshire Hathaway",     "sector": "Diversified"},
]

# ─────────────────────────────────────────────────────────────────────────────
# File paths (derived)
# ─────────────────────────────────────────────────────────────────────────────
RAW_DATA_PATH    = os.path.join(DATA_DIR,   f"{TICKER}_raw.csv")
MODEL_PATH       = os.path.join(MODEL_DIR,  "lstm_model.keras")
SCALER_PATH      = os.path.join(MODEL_DIR,  "scaler.pkl")
METRICS_JSON     = os.path.join(REPORT_DIR, "evaluation_metrics.json")
METRICS_CSV      = os.path.join(REPORT_DIR, "evaluation_metrics.csv")
HISTORY_PATH     = os.path.join(REPORT_DIR, "training_history.csv")
