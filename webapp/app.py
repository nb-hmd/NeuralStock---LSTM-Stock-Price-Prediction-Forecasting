"""
app.py — FastAPI web server for the Stock Price Prediction UI.

Endpoints:
  GET  /                          → Serve HTML frontend
  POST /api/predict               → Start prediction job for a ticker
  GET  /api/status/{job_id}       → Poll job progress / results
  GET  /api/cached/{ticker}       → Check if ticker results are cached
  GET  /api/plots/{ticker}/{file} → Serve a plot PNG image
"""

import io
import json
import os
import sys
import threading
import uuid
from pathlib import Path

# Force UTF-8 output on Windows
os.environ["PYTHONUTF8"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from webapp.pipeline_runner import (
    get_ticker_paths,
    is_cached,
    load_cached_results,
    run_pipeline,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="NeuralStock — LSTM Stock Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store: {job_id: {...status dict...}}
JOBS: dict[str, dict] = {}

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Startup: migrate existing AAPL root-level artifacts to AAPL/ subdir
# ---------------------------------------------------------------------------

@app.on_event("startup")
def _migrate_aapl():
    """Move root-level AAPL artifacts into outputs/plots/AAPL/ on first start."""
    import shutil
    root_plots = PROJECT_ROOT / "outputs" / "plots"
    aapl_plots = root_plots / "AAPL"
    aapl_plots.mkdir(parents=True, exist_ok=True)

    aapl_reports = PROJECT_ROOT / "outputs" / "reports" / "AAPL"
    aapl_reports.mkdir(parents=True, exist_ok=True)

    aapl_models = PROJECT_ROOT / "models" / "AAPL"
    aapl_models.mkdir(parents=True, exist_ok=True)

    # Move PNGs from root plots dir
    for png in root_plots.glob("*.png"):
        dest = aapl_plots / png.name
        if not dest.exists():
            shutil.copy2(str(png), str(dest))

    # Copy root metrics
    for name in ("evaluation_metrics.json", "evaluation_metrics.csv", "training_history.csv"):
        src = PROJECT_ROOT / "outputs" / "reports" / name
        if src.exists() and not (aapl_reports / name).exists():
            shutil.copy2(str(src), str(aapl_reports / name))

    # Copy root model / scaler
    for name in ("lstm_model.keras", "scaler.pkl"):
        src = PROJECT_ROOT / "models" / name
        if src.exists() and not (aapl_models / name).exists():
            shutil.copy2(str(src), str(aapl_models / name))

    print("[Startup] AAPL artifacts migrated to ticker-specific directories.")


import re
from src.shariah import resolve_symbol, get_all_shariah_symbols, get_popular_international_symbols

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    ticker: str
    force_retrain: bool = False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML frontend."""
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve the favicon."""
    ico_path = STATIC_DIR / "favicon.ico"
    if ico_path.exists():
        return FileResponse(str(ico_path), media_type="image/x-icon")
    svg_path = STATIC_DIR / "favicon.svg"
    if svg_path.exists():
        return FileResponse(str(svg_path), media_type="image/svg+xml")
    raise HTTPException(status_code=404, detail="Favicon not found")


@app.get("/api/shariah_symbols")
async def shariah_symbols():
    """Return list of all 241 Shariah-compliant PSX symbols with names and sectors."""
    return JSONResponse(get_all_shariah_symbols())


@app.get("/api/popular_international_symbols")
async def popular_international_symbols():
    """Return curated list of popular international (US, USD) large-cap tickers."""
    return JSONResponse(get_popular_international_symbols())


@app.get("/api/cached/{ticker}")
async def check_cached(ticker: str):
    """Return whether results for this ticker are already cached."""
    sym_info = resolve_symbol(ticker)
    t = sym_info["canonical_symbol"]
    cached = is_cached(t)
    result = {
        "ticker": t,
        "query_ticker": sym_info["query_ticker"],
        "cached": cached,
        "sym_info": sym_info
    }
    if cached:
        result.update(load_cached_results(t))
    return JSONResponse(result)


@app.post("/api/predict")
async def start_predict(req: PredictRequest):
    """
    Start the LSTM prediction pipeline for the requested ticker.
    Returns immediately with a job_id for status polling.
    If results are already cached (and force_retrain=False), returns them directly.
    """
    raw_ticker = req.ticker.strip().upper()
    if not raw_ticker or not re.match(r'^[A-Z0-9.\-_]{1,15}$', raw_ticker):
        raise HTTPException(status_code=422, detail="Invalid ticker symbol.")

    sym_info = resolve_symbol(raw_ticker)
    ticker = sym_info["canonical_symbol"]

    # Always launch fresh real-time pipeline on user search
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status":   "running",
        "progress": 0,
        "stage":    "Connecting to live market feed...",
        "ticker":   ticker,
        "query_ticker": sym_info["query_ticker"],
        "sym_info": sym_info,
        "cached":   False,
    }
    thread = threading.Thread(
        target=run_pipeline,
        args=(ticker, JOBS, job_id),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "cached": False}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Poll the status of a prediction job."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JSONResponse(JOBS[job_id])


@app.get("/api/plots/{ticker}/{filename}")
async def serve_plot(ticker: str, filename: str):
    """Serve a PNG plot for the given ticker."""
    sym_info = resolve_symbol(ticker)
    paths = get_ticker_paths(sym_info["canonical_symbol"])
    plot_path = Path(paths["plots_dir"]) / filename

    if not plot_path.exists() or not plot_path.suffix == ".png":
        raise HTTPException(status_code=404, detail="Plot not found.")
    return FileResponse(str(plot_path), media_type="image/png")
