"""Reader for the 5m cache populated by `refresh_5m.py`.

The sidecar refresher writes parquet files to `cache_5m/{TICKER}.parquet`
every 5 minutes. This module lets the streamlit app read those files
moment-to-moment without going out to the exchange on every page load.

The reader returns None (not raises) when:
  - the cache file doesn't exist (refresher hasn't run yet)
  - the file is older than `max_age_minutes` (stale — fall back to live)
  - the file is unreadable for any reason

Callers should treat a None return as "go fetch live."
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

CACHE_DIR = "cache_5m"
DEFAULT_MAX_AGE_MIN = 10            # >5min stale = fall back to live


@dataclass
class CachedSeries:
    close: pd.Series                 # Close prices (what data_provider returns)
    df: pd.DataFrame                 # Full OHLCV (useful for verification later)
    fetched_at: datetime             # mtime of the parquet
    age_minutes: float
    coin: str


def _ticker_to_cache_name(ticker: str) -> str:
    """Translate Yahoo-style ticker → cache filename used by refresher."""
    s = ticker.strip().upper()
    # Same convention the refresher uses
    return s.replace("/USDT", "-USD").replace("/USD", "-USD").replace("-USDT", "-USD")


def load_cached_5m(
    ticker: str,
    *, max_age_minutes: float = DEFAULT_MAX_AGE_MIN,
    cache_dir: str | None = None,
) -> CachedSeries | None:
    """Return cached 5m data if present and fresh; otherwise None.

    `cache_dir=None` resolves to the module-level CACHE_DIR at call time —
    this is important for tests that monkeypatch CACHE_DIR.
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR
    coin = _ticker_to_cache_name(ticker)
    path = os.path.join(cache_dir, f"{coin}.parquet")
    if not os.path.isfile(path):
        return None
    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
        age_min = (datetime.now(timezone.utc) - mtime).total_seconds() / 60.0
        if age_min > max_age_minutes:
            return None
        df = pd.read_parquet(path)
        if df.empty or "Close" not in df.columns:
            return None
        close = df["Close"].astype(float)
        if close.index.tz is not None:
            close.index = close.index.tz_convert(None)
        return CachedSeries(close=close, df=df, fetched_at=mtime,
                             age_minutes=age_min, coin=coin)
    except Exception:
        # Reader must never raise — caller will fall back to live
        return None


def cache_status() -> dict[str, dict]:
    """Diagnostic: report which coins are cached, when, how old."""
    out: dict[str, dict] = {}
    if not os.path.isdir(CACHE_DIR):
        return out
    for fname in sorted(os.listdir(CACHE_DIR)):
        if not fname.endswith(".parquet"):
            continue
        path = os.path.join(CACHE_DIR, fname)
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
            age_min = (datetime.now(timezone.utc) - mtime).total_seconds() / 60.0
            size_kb = os.path.getsize(path) / 1024.0
            coin = fname[:-len(".parquet")]
            out[coin] = {
                "path": path,
                "fetched_at": mtime.isoformat(),
                "age_minutes": round(age_min, 1),
                "size_kb": round(size_kb, 1),
                "fresh": age_min <= DEFAULT_MAX_AGE_MIN,
            }
        except Exception:
            continue
    return out
