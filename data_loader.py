"""Data ingestion: OHLCV fetch with multi-source fallback.

Primary source : yfinance (with curl_cffi Chrome impersonation if installed)
Fallback source: Stooq.com  — free public CSV endpoint, no API key,
                              very stable, supports US stocks/ETFs and FX.

The two sources cover each other: yfinance handles crypto and exotic
symbols Stooq doesn't, while Stooq handles the case where Yahoo
rate-limits or blocks the container IP.
"""
from __future__ import annotations

import io
import time
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
import yfinance as yf


_CACHE_TTL_SECONDS: int = 3600
_REQUIRED_COLS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")
_MAX_RETRIES: int = 2
_RETRY_BACKOFF_SECONDS: float = 1.5
_HTTP_TIMEOUT_SECONDS: int = 15

# Symbols that Stooq doesn't carry well (crypto, etc.) — skip the fallback
# for these and let the yfinance error surface clearly.
_CRYPTO_SUFFIXES: tuple[str, ...] = ("-USD", "-USDT", "-BTC", "-ETH")


# ---------------------------------------------------------------------------
# Source 1 — yfinance (with Chrome-impersonating session if curl_cffi present)
# ---------------------------------------------------------------------------
def _make_yf_session():
    try:
        from curl_cffi import requests as cffi_requests
        return cffi_requests.Session(impersonate="chrome")
    except Exception:
        return None


def _fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Try yfinance via both endpoints with a Chrome-impersonating session."""
    session = _make_yf_session()

    # First: bulk download endpoint.
    try:
        df = yf.download(
            ticker, start=start, end=end,
            auto_adjust=True, progress=False, threads=False,
            session=session,
        )
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    # Second: per-ticker history endpoint (often more reliable).
    try:
        t = yf.Ticker(ticker, session=session) if session else yf.Ticker(ticker)
        df = t.history(start=start, end=end, auto_adjust=True)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Source 2 — Stooq CSV endpoint (free, no key, very reliable)
# ---------------------------------------------------------------------------
# Stooq uses lowercase symbols with a market suffix:
#   US equities/ETFs:  "spy.us", "aapl.us", "qqq.us"
#   FX / indices:      "eurusd", "^spx"
# yfinance-style tickers need translation.
def _to_stooq_symbol(yf_symbol: str) -> str:
    s = yf_symbol.strip().lower()
    if any(s.endswith(suf.lower()) for suf in _CRYPTO_SUFFIXES):
        # e.g. btc-usd  →  btcusd  (Stooq style)
        return s.replace("-", "")
    # Plain alphanumeric → assume US listing.
    if s.isalpha() or all(c.isalnum() or c == "." for c in s):
        return s if "." in s else f"{s}.us"
    return s


def _fetch_stooq(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch daily OHLCV from Stooq as CSV."""
    symbol = _to_stooq_symbol(ticker)
    try:
        d1 = datetime.fromisoformat(start).strftime("%Y%m%d")
        d2 = datetime.fromisoformat(end).strftime("%Y%m%d")
    except ValueError:
        return None

    url = f"https://stooq.com/q/d/l/?s={symbol}&d1={d1}&d2={d2}&i=d"
    try:
        resp = requests.get(
            url,
            timeout=_HTTP_TIMEOUT_SECONDS,
            headers={"User-Agent": "Mozilla/5.0 (regime-dashboard)"},
        )
        resp.raise_for_status()
        text = resp.text.strip()
        if not text or "No data" in text or text.lower().startswith("<"):
            return None

        df = pd.read_csv(io.StringIO(text))
        if df.empty or "Date" not in df.columns:
            return None

        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
        # Stooq columns: Date,Open,High,Low,Close,Volume — already adjusted.
        # If Volume is missing (some FX/indices), fill with 0.
        if "Volume" not in df.columns:
            df["Volume"] = 0.0
        return df[list(_REQUIRED_COLS)]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner=False)
def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV with yfinance → Stooq fallback.

    Parameters
    ----------
    ticker : str
        yfinance-style symbol (e.g. "SPY", "AAPL", "BTC-USD").
    start, end : str
        ISO date strings (YYYY-MM-DD). `end` exclusive in yfinance.

    Returns
    -------
    pd.DataFrame
        OHLCV columns, sorted ascending by date.

    Raises
    ------
    ValueError
        If neither source returns data after retries.
    """
    if not ticker or not ticker.strip():
        raise ValueError("Ticker symbol is empty.")
    ticker_clean = ticker.strip().upper()

    df: pd.DataFrame | None = None
    used_source: str = ""

    # Try yfinance first — preferred because it auto-adjusts properly and
    # supports the widest universe (incl. crypto).
    for attempt in range(_MAX_RETRIES):
        df = _fetch_yfinance(ticker_clean, start, end)
        if df is not None and not df.empty:
            used_source = "yfinance"
            break
        time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))

    # If yfinance failed and this isn't a crypto symbol, try Stooq.
    if (df is None or df.empty) and not any(
        ticker_clean.endswith(suf) for suf in _CRYPTO_SUFFIXES
    ):
        df = _fetch_stooq(ticker_clean, start, end)
        if df is not None and not df.empty:
            used_source = "stooq"

    if df is None or df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker_clean}' between {start} and {end} "
            "from either Yahoo Finance or Stooq. Check the symbol and date range. "
            "Yahoo Finance may be rate-limiting this IP — wait a minute and try again. "
            "Stooq covers US stocks/ETFs and most FX; for exotic symbols only yfinance works."
        )

    # Normalize.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Data response for '{ticker_clean}' is missing columns: {missing}"
        )

    df = df[list(_REQUIRED_COLS)].copy()
    df.sort_index(ascending=True, inplace=True)
    df.index.name = "Date"

    # Strip timezone (yfinance often returns tz-aware UTC; Stooq is naive).
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Stash which source was used so the UI can show it (read via .attrs).
    df.attrs["source"] = used_source
    return df
