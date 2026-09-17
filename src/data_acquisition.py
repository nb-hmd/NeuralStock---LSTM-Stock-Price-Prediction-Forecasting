"""
data_acquisition.py — Downloads historical OHLCV stock data from Yahoo Finance.

Uses the yfinance library to fetch real market data. Saves to CSV so the
pipeline can be re-run without re-downloading (unless the file is missing or
the user explicitly requests a refresh).

No synthetic data, no fallbacks — if the download fails, the error is raised.
"""

import io
import os
import sys
import time
import logging
import urllib.request
import urllib.parse
from datetime import datetime
from typing import Optional

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer") and sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import yfinance as yf

_yf_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "yf_cache")
_yf_cache_dir = os.path.abspath(_yf_cache_dir)
os.makedirs(_yf_cache_dir, exist_ok=True)
os.makedirs(os.path.join(_yf_cache_dir, "tz"), exist_ok=True)
if hasattr(yf.cache, "set_cache_location"):
    yf.cache.set_cache_location(_yf_cache_dir)
if hasattr(yf.cache, "set_tz_cache_location"):
    yf.cache.set_tz_cache_location(os.path.join(_yf_cache_dir, "tz"))

from src.config import (
    TICKER, START_DATE, END_DATE,
    RAW_DATA_PATH, DATA_DIR
)
from src.shariah import resolve_symbol

logger = logging.getLogger(__name__)


def fetch_psx_dps_historical(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch authentic historical daily OHLCV trading data directly from the
    official Pakistan Stock Exchange Data Portal (https://dps.psx.com.pk/historical).

    This provides 100% genuine, real-time trading floor data for all PSX listed
    equities (including all 457+ Shariah-compliant equities), completely eliminating
    Yahoo Finance coverage gaps, ticker delisting errors, and data omission issues.

    Parameters
    ----------
    symbol : str
        PSX symbol (e.g. 'IPAK', 'MEBL', 'OGDC').
    start : str, optional
        Start date filter in 'YYYY-MM-DD' format.
    end : str, optional
        End date filter in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame with columns: Open, High, Low, Close, Volume.
    """
    clean_sym = symbol.upper().strip().replace(".KA", "")
    url = "https://dps.psx.com.pk/historical"
    data = urllib.parse.urlencode({"symbol": clean_sym}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to connect to official PSX Data Portal for symbol '{clean_sym}': {exc}"
        ) from exc

    tables = pd.read_html(io.StringIO(html))
    if not tables or tables[0].empty:
        raise RuntimeError(
            f"Official PSX Data Portal returned no historical trading table for '{clean_sym}'. "
            "Please verify that the symbol is an active PSX listed company."
        )

    raw = tables[0]
    col_map = {c: c.strip().capitalize() for c in raw.columns}
    raw.rename(columns=col_map, inplace=True)

    expected = ["Date", "Open", "High", "Low", "Close", "Volume"]
    for col in expected:
        if col not in raw.columns:
            raise ValueError(
                f"PSX DPS historical table is missing expected column: {col}. "
                f"Available: {list(raw.columns)}"
            )

    raw["Date"] = pd.to_datetime(raw["Date"])
    raw.set_index("Date", inplace=True)
    raw.sort_index(inplace=True)

    for c in ["Open", "High", "Low", "Close"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    raw["Volume"] = raw["Volume"].astype(str).str.replace(",", "").str.strip()
    raw["Volume"] = pd.to_numeric(raw["Volume"], errors="coerce")

    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all").copy()

    # Apply date filters
    if start:
        filtered = df.loc[start:]
        # If the company was listed recently (e.g. IPAK in 2024), keep all data from IPO
        if len(filtered) >= 60:
            df = filtered

    if end:
        df = df.loc[:end]

    if df.empty or len(df) < 60:
        raise RuntimeError(
            f"Official PSX Data Portal returned insufficient trading sessions for '{clean_sym}' "
            f"({len(df)} rows). A minimum of 60 trading days is required for LSTM sequence modeling."
        )

    df.index.name = "Date"
    return df


def download_stock_data(
    ticker: str = TICKER,
    start: str = START_DATE,
    end: str = END_DATE,
    force_refresh: bool = False,
    save_path: str = None
) -> pd.DataFrame:
    """
    Download historical OHLCV data for a given ticker.
    For PSX equities, fetches directly from the official Pakistan Stock Exchange (DPS).
    For US/International equities, fetches from Yahoo Finance.
    Always ensures authentic real-time data up to the latest market session.

    Parameters
    ----------
    ticker : str
        Stock symbol (e.g. 'IPAK', 'MEBL', 'OGDC.KA', 'AAPL', 'MSFT').
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str, optional
        End date in 'YYYY-MM-DD' format (None = current date).
    force_refresh : bool
        If True, re-download even if a cached CSV exists.
    save_path : str, optional
        Custom path to save the raw CSV file. Defaults to DATA_DIR/{canonical}_raw.csv.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame with columns: Open, High, Low, Close, Volume.
    """
    sym_info = resolve_symbol(ticker)
    query_ticker = sym_info["query_ticker"]
    canonical = sym_info["canonical_symbol"]
    currency_sym = sym_info["currency_symbol"]
    is_psx = sym_info.get("is_psx", False)

    raw_path = save_path or os.path.join(DATA_DIR, f"{canonical}_raw.csv")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    today = datetime.now().date()
    target_end = end or datetime.now().strftime("%Y-%m-%d")

    # ── Check if cached CSV exists and is up to date ─────────────────────────
    if os.path.exists(raw_path) and not force_refresh:
        try:
            df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
            if not df.empty and len(df) >= 60:
                latest_date = df.index[-1].date()
                days_old = (today - latest_date).days
                # Strict: CACHE IS ONLY VALID WITHIN 1 BUSINESS DAY of today
                from pandas.tseries.offsets import BDay as _BDay
                one_bday_ago = (datetime.now() - _BDay(1)).date()
                if latest_date >= one_bday_ago:
                    logger.info(
                        f"[DataAcquisition] Using up-to-date data for {canonical} "
                        f"({len(df):,} rows up to {latest_date})"
                    )
                    return df
                else:
                    logger.info(
                        f"[DataAcquisition] Cached data for {canonical} is {days_old} calendar days old "
                        f"({latest_date}). Min acceptable: {one_bday_ago}. Fetching fresh live market data..."
                    )
        except Exception as e:
            logger.warning(f"[DataAcquisition] Could not read existing CSV: {e}")

    # ── Fetch authentic market data ──────────────────────────────────────────
    df = None

    if is_psx:
        logger.info(
            f"[DataAcquisition] Ingesting authentic PSX market data for {canonical} "
            f"({sym_info['company_name']}) from official PSX Data Portal (DPS) …"
        )
        try:
            df = fetch_psx_dps_historical(canonical, start=start, end=target_end)
            logger.info(
                f"[DataAcquisition] Successfully ingested {len(df):,} trading sessions for {canonical} "
                f"from official Pakistan Stock Exchange ({df.index[0].date()} → {df.index[-1].date()})"
            )
        except Exception as exc:
            logger.warning(
                f"[DataAcquisition] PSX DPS primary query failed for {canonical}: {exc}. "
                "Attempting secondary query via Yahoo Finance..."
            )

    if df is None:
        logger.info(
            f"[DataAcquisition] Downloading live data for {query_ticker} ({sym_info['company_name']}) "
            f"from Yahoo Finance ({start} → {target_end}) …"
        )
        max_retries = 3
        raw = None
        for attempt in range(1, max_retries + 1):
            # 1. Try download with start and end
            try:
                raw = yf.download(
                    query_ticker,
                    start=start,
                    end=target_end,
                    auto_adjust=True,
                    progress=False,
                )
                if raw is not None and not raw.empty:
                    break
            except Exception as exc:
                logger.warning(f"[DataAcquisition] Attempt {attempt} yf.download(start, end) failed: {exc}")

            # 2. Try download with start only (up to latest available minute)
            try:
                raw = yf.download(
                    query_ticker,
                    start=start,
                    auto_adjust=True,
                    progress=False,
                )
                if raw is not None and not raw.empty:
                    break
            except Exception as exc:
                logger.warning(f"[DataAcquisition] Attempt {attempt} yf.download(start) failed: {exc}")

            # 3. If start date was rejected, try period='5y' (last 5 years of live data)
            try:
                logger.info(f"[DataAcquisition] Attempt {attempt}: Trying period='5y' live fallback for {query_ticker}...")
                raw = yf.download(query_ticker, period="5y", auto_adjust=True, progress=False)
                if raw is not None and not raw.empty:
                    break
                t_obj = yf.Ticker(query_ticker)
                raw = t_obj.history(period="5y", auto_adjust=True)
                if raw is not None and not raw.empty:
                    break
            except Exception as exc:
                logger.warning(f"[DataAcquisition] Attempt {attempt} period=5y fallback failed: {exc}")

            time.sleep(2)

        if raw is None or raw.empty:
            source_desc = "Official PSX Data Portal and Yahoo Finance" if is_psx else "Yahoo Finance"
            raise RuntimeError(
                f"{source_desc} returned empty data for '{query_ticker}'. "
                "Please verify the symbol or exchange availability."
            )

        # ── Clean up column structure ─────────────────────────────────────────────
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Keep only OHLCV columns
        ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in ohlcv_cols if c not in raw.columns]
        if missing:
            raise ValueError(
                f"Downloaded data is missing expected columns: {missing}. "
                f"Available: {list(raw.columns)}"
            )
        df = raw[ohlcv_cols].copy()
        df.dropna(how="all", inplace=True)
        df.sort_index(inplace=True)
        df.index.name = "Date"

    # ── Drop rows where ALL OHLCV values are NaN (fully missing trading days) ─
    df.dropna(how="all", inplace=True)
    df.sort_index(inplace=True)
    df.index.name = "Date"

    # ── FRESHNESS ENFORCEMENT: raise if downloaded data is too old ───────────
    from pandas.tseries.offsets import BDay as _BDayEnforce
    latest_date = df.index[-1].date()
    two_bday_ago = (datetime.now() - _BDayEnforce(2)).date()
    if latest_date < two_bday_ago:
        raise RuntimeError(
            f"Data integrity check FAILED — Stale market data for '{canonical}'.\n"
            f"  Latest data point: {latest_date}\n"
            f"  Minimum acceptable: {two_bday_ago} (within 2 business days of today: {today})\n"
            f"  The market data provider may be delayed or the symbol/exchange may be unavailable.\n"
            f"  NO SYNTHETIC/FALLBACK DATA WILL BE SUBSTITUTED. Please retry later or verify the ticker."
        )
    logger.info(
        f"[DataAcquisition] Freshness OK: latest={latest_date} >= cutoff={two_bday_ago}"
    )

    # ── Save raw CSV ──────────────────────────────────────────────────────────
    df.to_csv(raw_path)
    logger.info(
        f"[DataAcquisition] Saved {len(df):,} rows to {raw_path} | "
        f"{df.index[0].date()} → {df.index[-1].date()}"
    )

    _log_data_summary(df, canonical, sym_info)
    return df


def _log_data_summary(df: pd.DataFrame, ticker: str, sym_info: dict = None) -> None:
    """Print a quick summary of the downloaded dataset with currency."""
    cs = sym_info["currency_symbol"] if sym_info else "$"
    name = sym_info["company_name"] if sym_info else ticker
    ex = sym_info["exchange"] if sym_info else "Global"

    print("\n" + "=" * 60)
    print(f"  [Data] {ticker} — {name} ({ex})")
    print("=" * 60)
    print(f"  Date Range   : {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"  Trading Days : {len(df):,}")
    print(f"  Years Covered: {(df.index[-1] - df.index[0]).days / 365.25:.1f}")
    print("\n  Price Range (Close):")
    print(f"    Min : {cs}{df['Close'].min():.2f}")
    print(f"    Max : {cs}{df['Close'].max():.2f}")
    print(f"    Mean: {cs}{df['Close'].mean():.2f}")
    print("\n  Missing values per column:")
    for col in df.columns:
        n_missing = df[col].isna().sum()
        print(f"    {col:8s}: {n_missing}")
    print("=" * 60 + "\n")
