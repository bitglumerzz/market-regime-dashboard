"""Streamlit-side client for the host TV MCP bridge.

Writes JSON request files into `tv_bridge_queue/`. The bridge daemon (running
on macOS host) picks them up, calls TV MCP CLI, and writes results to the
specified `out` paths.

Two modes:
  - submit_request(req): fire-and-forget. Returns request id immediately.
  - submit_and_wait(req, timeout): blocks until .done.ok or .done.error.

The queue is bind-mounted into Docker so writes are visible to the bridge.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path


QUEUE_DIR = "tv_bridge_queue"


# Yahoo → TV symbol mapping for known tickers (specific exchange preference)
TICKER_TO_TV: dict[str, str] = {
    "BTC-USD":  "BINANCE:BTCUSDT",
    "ETH-USD":  "BINANCE:ETHUSDT",
    "HYPE-USD": "BYBIT:HYPEUSDT",
    "SOL-USD":  "BINANCE:SOLUSDT",
    "ADA-USD":  "BINANCE:ADAUSDT",
    "AVAX-USD": "BINANCE:AVAXUSDT",
    "DOGE-USD": "BINANCE:DOGEUSDT",
    "DOT-USD":  "BINANCE:DOTUSDT",
    "LINK-USD": "BINANCE:LINKUSDT",
    "MATIC-USD":"BINANCE:MATICUSDT",
    "XRP-USD":  "BINANCE:XRPUSDT",
    "LTC-USD":  "BINANCE:LTCUSDT",
}


def _to_tv_symbol(ticker: str) -> str:
    """Translate any Yahoo ticker to TV format. Fall back to BINANCE: prefix."""
    s = ticker.strip().upper()
    if s in TICKER_TO_TV:
        return TICKER_TO_TV[s]
    # Generic crypto fallback: BTC-USD → BINANCE:BTCUSDT
    if s.endswith("-USD"):
        return f"BINANCE:{s.replace('-USD', 'USDT')}"
    if s.endswith("-USDT"):
        return f"BINANCE:{s.replace('-', '')}"
    # Unknown — pass through as-is, TV may still recognize it
    return s
# our TF → TV CLI interval string
TF_TO_TV: dict[str, str] = {
    "1d": "1D", "1D": "1D",
    "4h": "240", "4H": "240",
    "1h": "60",  "1H": "60",
    "15m": "15",
    "5m":  "5",
}


def is_bridge_available() -> bool:
    """True if the queue directory exists (means bind mount is wired)."""
    return os.path.isdir(QUEUE_DIR)


def submit_request(req: dict) -> str:
    """Drop a request into the queue. Returns the request id."""
    os.makedirs(QUEUE_DIR, exist_ok=True)
    rid = uuid.uuid4().hex[:12]
    path = os.path.join(QUEUE_DIR, f"{rid}.req")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(req, f, ensure_ascii=False)
    os.replace(tmp, path)
    return rid


def submit_and_wait(req: dict, timeout: float = 45.0,
                     poll: float = 0.5) -> tuple[str, bool, str]:
    """Submit + block until .done.{status} appears (or timeout).

    Returns (status, ok, reason). status ∈ {ok, error, timeout}.
    """
    rid = submit_request(req)
    deadline = time.time() + timeout
    while time.time() < deadline:
        for suffix in (".done.ok", ".done.error"):
            done = os.path.join(QUEUE_DIR, f"{rid}{suffix}")
            if os.path.isfile(done):
                if suffix == ".done.ok":
                    _cleanup(rid)
                    return "ok", True, ""
                try:
                    with open(done, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    reason = meta.get("reason", "")
                except Exception:
                    reason = ""
                _cleanup(rid)
                return "error", False, reason
        time.sleep(poll)
    return "timeout", False, f"no response in {timeout}s"


def _cleanup(rid: str) -> None:
    """Remove .done files for finished requests."""
    for suffix in (".done.ok", ".done.error"):
        p = os.path.join(QUEUE_DIR, f"{rid}{suffix}")
        try:
            if os.path.isfile(p):
                os.remove(p)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------
def request_screenshot(ticker: str, tf: str, out_path: str,
                        timeout: float = 200.0) -> bool:
    """Request a TV chart screenshot. Returns True on success.

    Default timeout 200s — TV chart switch + render + CDP capture can take
    60-120s depending on chart complexity.
    """
    if not is_bridge_available():
        return False
    req = {
        "type":     "screenshot",
        "symbol":   _to_tv_symbol(ticker),
        "interval": TF_TO_TV.get(tf, "1D"),
        "region":   "chart",
        "out":      os.path.abspath(out_path),
    }
    status, ok, _ = submit_and_wait(req, timeout=timeout)
    return ok


def request_pine_labels(ticker: str, tf: str, out_path: str,
                         study_filter: str = "", timeout: float = 30.0) -> bool:
    """Request Pine label data (e.g. Elliott Wave labels). Returns True on success."""
    if not is_bridge_available():
        return False
    req = {
        "type":         "pine_labels",
        "symbol":       _to_tv_symbol(ticker),
        "interval":     TF_TO_TV.get(tf, "1D"),
        "study_filter": study_filter,
        "out":          os.path.abspath(out_path),
    }
    status, ok, _ = submit_and_wait(req, timeout=timeout)
    return ok


def request_pine_lines(ticker: str, tf: str, out_path: str,
                        study_filter: str = "", timeout: float = 30.0) -> bool:
    if not is_bridge_available():
        return False
    req = {
        "type":         "pine_lines",
        "symbol":       _to_tv_symbol(ticker),
        "interval":     TF_TO_TV.get(tf, "1D"),
        "study_filter": study_filter,
        "out":          os.path.abspath(out_path),
    }
    status, ok, _ = submit_and_wait(req, timeout=timeout)
    return ok
