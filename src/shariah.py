"""
shariah.py — Shariah-compliant stock universe management & symbol resolution.

Handles:
- Loading the 241 Shariah-compliant symbols from unique_symbols.csv
- Mapping PSX ticker symbols to company names and sectors
- Resolving user queries (e.g. 'MEBL', 'OGDC.KA', 'AAPL') to Yahoo Finance tickers,
  exchange metadata, currency, and compliance status.
"""

import json
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from src.config import POPULAR_INTERNATIONAL

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "unique_symbols.csv"
COMPANIES_JSON_PATH = PROJECT_ROOT / "data" / "psx_companies.json"

# Load base pre-configured companies
PSX_COMPANIES: Dict[str, Dict[str, str]] = {}
if COMPANIES_JSON_PATH.exists():
    try:
        with open(COMPANIES_JSON_PATH, "r", encoding="utf-8") as f:
            PSX_COMPANIES = json.load(f)
    except Exception:
        pass

_SHARIAH_SYMBOLS_SET = set()
_DPS_CACHE: Dict[str, dict] = {}
_FORCE_PSX_VERIFY: bool = False
_INTERNATIONAL_SYMBOLS_SET = None


def load_shariah_symbols() -> List[str]:
    """Load and cache the unique Shariah-compliant symbols from CSV."""
    global _SHARIAH_SYMBOLS_SET
    if _SHARIAH_SYMBOLS_SET:
        return sorted(list(_SHARIAH_SYMBOLS_SET))

    symbols = []
    if CSV_PATH.exists():
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip().upper()
                if s and s != "SYMBOL" and not s.startswith("#"):
                    symbols.append(s)
    _SHARIAH_SYMBOLS_SET = set(symbols)
    return sorted(symbols)


def is_shariah_compliant(symbol: str) -> bool:
    """Return True if the given symbol is in the PSX Shariah universe."""
    load_shariah_symbols()
    clean = symbol.upper().strip().replace(".KA", "")
    return clean in _SHARIAH_SYMBOLS_SET


def _query_live_psx_dps(symbol: str) -> Optional[dict]:
    """Query live PSX Data Portal System to check if symbol is a listed PSX equity.

    NO SILENT FAILURES: If the module-level _FORCE_PSX_VERIFY flag is True (set
    when the caller explicitly expects a PSX listing, e.g. .KA suffix or
    Shariah-list membership), a DPS API failure is raised as RuntimeError
    instead of silently returning None.
    """
    global _DPS_CACHE, _FORCE_PSX_VERIFY
    sym_upper = symbol.upper().strip()
    if sym_upper in _DPS_CACHE:
        return _DPS_CACHE[sym_upper]

    dps_error = None
    try:
        req = urllib.request.Request(
            "https://dps.psx.com.pk/symbols",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for item in data:
                s = item.get("symbol", "").upper().strip()
                _DPS_CACHE[s] = {
                    "symbol": s,
                    "name": item.get("name", f"{s} Limited"),
                    "sector": item.get("sectorName", "PSX Equities"),
                    "is_equity": not item.get("isDebt", False) and not item.get("isETF", False)
                }
    except Exception as e:
        dps_error = str(e)

    if dps_error is not None:
        msg = (
            f"[Shariah::DPS] Live PSX Data Portal query failed for symbol '{sym_upper}': {dps_error}. "
            "PSX metadata (company name, sector, equity verification) cannot be confirmed."
        )
        if _FORCE_PSX_VERIFY:
            raise RuntimeError(
                msg + " EXPLICIT PSX VERIFICATION WAS REQUIRED but the DPS API was unreachable. "
                "No synthetic or fallback metadata will be used. Check network connectivity or "
                "disable forced PSX verification for this symbol."
            )
        print(f"  !! {msg}")

    return _DPS_CACHE.get(sym_upper)


def register_new_shariah_symbol(symbol: str, name: str = "", sector: str = "") -> None:
    """Dynamically register a newly discovered Shariah symbol into memory and unique_symbols.csv."""
    load_shariah_symbols()
    clean = symbol.upper().strip().replace(".KA", "")
    if clean not in _SHARIAH_SYMBOLS_SET:
        _SHARIAH_SYMBOLS_SET.add(clean)
        try:
            with open(CSV_PATH, "a", encoding="utf-8") as f:
                f.write(f"\n{clean}")
        except Exception:
            pass

    if clean not in PSX_COMPANIES:
        PSX_COMPANIES[clean] = {
            "name": name or f"{clean} Limited",
            "sector": sector or "PSX Shariah Constituent"
        }
        try:
            with open(COMPANIES_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(PSX_COMPANIES, f, indent=2, ensure_ascii=False)
        except Exception:
            pass


def resolve_symbol(raw_input: str) -> dict:
    """
    Resolve a user input ticker into a canonical symbol, Yahoo Finance query ticker,
    storage identifier, exchange, currency, and Shariah compliance status.

    Examples:
      'MEBL'    -> query: 'MEBL.KA', canonical: 'MEBL', exchange: 'PSX', PKR (Rs.), Shariah: True
      'OGDC.KA' -> query: 'OGDC.KA', canonical: 'OGDC', exchange: 'PSX', PKR (Rs.), Shariah: True
      'AAPL'    -> query: 'AAPL',    canonical: 'AAPL', exchange: 'US',  USD ($),   Shariah: False
    """
    load_shariah_symbols()
    clean = raw_input.upper().strip()

    has_ka_suffix = clean.endswith(".KA")
    base_symbol = clean[:-3] if has_ka_suffix else clean

    # Build lookup set of known international tickers ONCE at first call
    global _INTERNATIONAL_SYMBOLS_SET
    if _INTERNATIONAL_SYMBOLS_SET is None:
        _INTERNATIONAL_SYMBOLS_SET = {
            str(e.get("symbol", "")).strip().upper()
            for e in POPULAR_INTERNATIONAL
            if str(e.get("symbol", "")).strip()
        }
    in_international_list = base_symbol in _INTERNATIONAL_SYMBOLS_SET

    # Disambiguation rule (critical for name collisions like META, COST):
    #   - If user explicitly added .KA suffix → always PSX, never override.
    #   - If bare ticker is in POPULAR_INTERNATIONAL curated list → always US/USD.
    #     User typed/curated it as international (quick pick chip), so it beats
    #     accidental name collisions with Pakistani Shariah symbols.
    #   - Otherwise proceed to Shariah list / live DPS lookup (original PSX branch).
    skip_psx_detection = (not has_ka_suffix) and in_international_list

    # Check if base_symbol is in the Shariah-compliant PSX list
    in_shariah_list = (base_symbol in _SHARIAH_SYMBOLS_SET) and not skip_psx_detection

    global _FORCE_PSX_VERIFY
    _FORCE_PSX_VERIFY = has_ka_suffix or in_shariah_list

    dps_info = None
    if not in_shariah_list and not skip_psx_detection:
        dps_info = _query_live_psx_dps(base_symbol)

    _FORCE_PSX_VERIFY = False

    is_psx = has_ka_suffix or in_shariah_list or (dps_info is not None)
    canonical = base_symbol

    if is_psx:
        query_ticker = f"{canonical}.KA"
        storage_dir = canonical
        exchange = "Pakistan Stock Exchange (PSX)"
        currency = "PKR"
        currency_symbol = "Rs. "

        company_name = ""
        sector = ""

        if canonical in PSX_COMPANIES:
            company_name = PSX_COMPANIES[canonical]["name"]
            sector = PSX_COMPANIES[canonical]["sector"]
        elif dps_info:
            company_name = dps_info["name"]
            sector = dps_info["sector"]
        else:
            company_name = f"{canonical} (PSX Listed)"
            sector = "Pakistan Stock Exchange"

        # If it was verified via DPS or is in the Shariah universe
        is_shariah = in_shariah_list or (dps_info is not None and dps_info.get("is_equity"))
        if is_shariah and canonical not in _SHARIAH_SYMBOLS_SET:
            register_new_shariah_symbol(canonical, company_name, sector)

    else:
        query_ticker = clean
        storage_dir = clean
        exchange = "US / International"
        currency = "USD"
        currency_symbol = "$"
        is_shariah = False
        company_name = clean
        sector = "International Equity"

    return {
        "raw_input": raw_input,
        "canonical_symbol": canonical,
        "query_ticker": query_ticker,
        "storage_dir": storage_dir,
        "is_psx": is_psx,
        "is_shariah": is_shariah,
        "exchange": exchange,
        "currency": currency,
        "currency_symbol": currency_symbol,
        "company_name": company_name,
        "sector": sector,
    }


def get_all_shariah_symbols() -> List[dict]:
    """Return full list of all 453+ Shariah symbols with company names and sectors."""
    load_shariah_symbols()
    result = []
    for sym in sorted(list(_SHARIAH_SYMBOLS_SET)):
        info = PSX_COMPANIES.get(sym, {
            "name": f"{sym} Limited",
            "sector": "PSX Shariah Constituent"
        })
        result.append({
            "symbol": sym,
            "query_ticker": f"{sym}.KA",
            "name": info["name"],
            "sector": info["sector"],
            "is_shariah": True,
            "currency": "PKR",
            "currency_symbol": "Rs. ",
        })
    return result


def get_popular_international_symbols() -> List[dict]:
    """
    Return the curated list of popular international (US-listed, USD)
    large-cap tickers with company names and sectors.

    Every symbol in this list is independently verified via a live
    Yahoo Finance 5-day probe returning >= 1 row of real OHLCV data
    (no fake or hardcoded symbols that 404).
    """
    result = []
    seen = set()
    for entry in POPULAR_INTERNATIONAL:
        sym = str(entry.get("symbol", "")).strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        result.append({
            "symbol": sym,
            "query_ticker": sym,
            "name": str(entry.get("name", sym)),
            "sector": str(entry.get("sector", "International Equity")),
            "is_shariah": False,
            "currency": "USD",
            "currency_symbol": "$",
        })
    return result

