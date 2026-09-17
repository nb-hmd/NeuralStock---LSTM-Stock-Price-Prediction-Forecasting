"""
run_webapp.py — Launch the NeuralStock FastAPI server.

Usage:
    venv\\Scripts\\python.exe run_webapp.py

The server starts at http://127.0.0.1:8000
"""

import io
import os
import sys

# Force UTF-8 output on Windows before anything else
os.environ["PYTHONUTF8"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer") and sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("  NeuralStock  -  LSTM Stock Prediction UI")
    print("=" * 60)
    print("  Server URL : http://127.0.0.1:8000")
    print("  API Docs   : http://127.0.0.1:8000/docs")
    print("=" * 60)
    print("  Open http://127.0.0.1:8000 in your browser")
    print("  Press CTRL+C to stop the server")
    print("=" * 60 + "\n")

    uvicorn.run(
        "webapp.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        workers=1,
        log_level="info",
    )
