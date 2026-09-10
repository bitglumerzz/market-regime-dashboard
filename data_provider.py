"""Pluggable OHLCV data source for the Elliott Waves page.

Two backends:
  1. CCXT (Binance spot/futures) — for crypto. Free, no API key, years of
     true 4h / 15m / 1m data without resampling. Strongly preferred for
     crypto tickers like BTC-USD, ETH-USD.
  2. yfinance — for US stocks/ETFs. Intraday history is limited to 60 days
     (15m) / 730 days (1h), and 4h must be resampled from 1h.

The picker `fetch_close()` auto-routes: crypto tickers go to CCXT (if
installed), everything else to yfinance.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

import pandas as pd


Source = Literal["auto", "ccxt", "yfinance"]


# Map our timeframe names to CCXT's exchange-native timeframe codes.
_CCXT_TIMEFRAMES: dict[str, str] = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "12h": "12h",
    "1d": "1d", "1w": "1w",
}

# Crypto-style ticker suffixes we route to CCXT by default.
_CRYPTO_SUFFIXES: tuple[str, ...] = ("-USD", "-USDT", "-BTC", "-ETH", "/USD",
                                       "/USDT")

# Per-coin exchange fallback chain. Binance lists most majors but not all
# alt-coins (e.g. HYPE never listed on Binance spot); KuCoin / Bybit / native
# DEX (Hyperliquid) cover the gaps. The first exchange to return data wins.
# Anything not in this map defaults to ["binance"] with the auto-derived symbol.
_EXCHANGE_CHAIN: dict[str, list[tuple[str, str]]] = {
    "BTC-USD":  [("binance", "BTC/USDT"),  ("kucoin", "BTC/USDT"),  ("bybit", "BTC/USDT")],
    "ETH-USD":  [("binance", "ETH/USDT"),  ("kucoin", "ETH/USDT"),  ("bybit", "ETH/USDT")],
    "HYPE-USD": [("binance", "HYPE/USDT"), ("kucoin", "HYPE/USDT"),
                  ("bybit",   "HYPE/USDT"), ("hyperliquid", "HYPE/USDC")],
    "SOL-USD":  [("binance", "SOL/USDT"),  ("kucoin", "SOL/USDT"),  ("bybit", "SOL/USDT")],
}


def _exchange_chain_for(ticker: str) -> list[tuple[str, str]]:
    """Resolve the (exchange, symbol) chain for a crypto ticker."""
    s = ticker.strip().upper()
    if s in _EXCHANGE_CHAIN:
        return _EXCHANGE_CHAIN[s]
    # Default — try Binance, KuCoin, Bybit with auto-translated symbol
    symbol = _to_ccxt_symbol(s)
    return [("binance", symbol), ("kucoin", symbol), ("bybit", symbol)]


def _is_crypto(ticker: str) -> bool:
    s = ticker.strip().upper()
    return any(s.endswith(suf) for suf in _CRYPTO_SUFFIXES)


def _to_ccxt_symbol(yf_ticker: str) -> str:
    """Translate 'BTC-USD' → 'BTC/USDT' (Binance pair format).

    Yahoo uses '-USD' (USD spot), Binance lists USDT pairs by default.
    For crypto, USD ≈ USDT for practical OHLCV purposes.
    """
    s = yf_ticker.strip().upper()
    if "/" in s:
        return s
    if s.endswith("-USD"):
        return s.replace("-USD", "/USDT")
    if s.endswith("-USDT"):
        return s.replace("-", "/")
    if s.endswith("-BTC"):
        return s.replace("-", "/")
    if s.endswith("-ETH"):
        return s.replace("-", "/")
    return s


# ---------------------------------------------------------------------------
# CCXT backend
# ---------------------------------------------------------------------------
def _fetch_one_exchange(
    exchange_name: str, symbol: str, timeframe: str, lookback_days: int,
) -> pd.Series:
    """Pull OHLCV from one specific exchange. Returns a Close-price Series.

    Raises on any failure — caller is responsible for trying the next exchange
    in the chain.
    """
    import ccxt
    if timeframe not in _CCXT_TIMEFRAMES:
        raise ValueError(
            f"timeframe {timeframe!r} not supported; "
            f"choose from {list(_CCXT_TIMEFRAMES)}"
        )
    ex_cls = getattr(ccxt, exchange_name, None)
    if ex_cls is None:
        raise ValueError(f"unknown exchange {exchange_name!r}")
    exchange = ex_cls({"enableRateLimit": True})

    tf = _CCXT_TIMEFRAMES[timeframe]
    end_ms = int(datetime.utcnow().timestamp() * 1000)
    start_ms = int((datetime.utcnow() - timedelta(days=lookback_days)).timestamp() * 1000)

    bars: list[list] = []
    cursor = start_ms
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=tf, since=cursor, limit=1000)
        if not batch:
            break
        bars.extend(batch)
        last_ts = batch[-1][0]
        if last_ts >= end_ms or last_ts == cursor:
            break
        cursor = last_ts + 1
        if len(bars) > 50_000:
            break

    if not bars:
        raise ValueError(f"{exchange_name} returned no bars for {symbol} {tf}")

    df = pd.DataFrame(bars, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("Date").drop(columns=["ts"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df["Close"].astype(float)


def _fetch_ccxt(
    ticker: str, timeframe: str, lookback_days: int,
    exchange_name: str | None = None,
) -> pd.Series:
    """Pull OHLCV from CCXT, trying each exchange in the ticker's fallback chain.

    For coins like HYPE that aren't on Binance, this falls back to KuCoin /
    Bybit / Hyperliquid. The first exchange to return data wins.
    """
    import ccxt  # noqa — surface ImportError early

    if exchange_name is not None:
        # Caller wants a specific exchange — use that, no fallback.
        return _fetch_one_exchange(exchange_name, _to_ccxt_symbol(ticker),
                                    timeframe, lookback_days)

    chain = _exchange_chain_for(ticker)
    last_error: Exception | None = None
    for exchange_id, symbol in chain:
        try:
            return _fetch_one_exchange(exchange_id, symbol, timeframe,
                                        lookback_days)
        except Exception as exc:
            last_error = exc
            continue
    raise ValueError(
        f"All CCXT exchanges failed for {ticker} {timeframe}. "
        f"Tried: {[ex for ex, _ in chain]}. "
        f"Last error: {type(last_error).__name__}: {last_error}"
    )


# ---------------------------------------------------------------------------
# yfinance backend (mirrors multi_tf's existing logic for non-crypto)
# ---------------------------------------------------------------------------
def _fetch_yfinance(
    ticker: str, timeframe: str, lookback_days: int,
    resample: str | None = None,
) -> pd.Series:
    import yfinance as yf

    interval_map = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "1d": "1d",
    }
    interval = interval_map.get(timeframe, timeframe)

    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=True,
        progress=False, threads=False,
    )
    if df is None or df.empty:
        raise ValueError(f"yfinance returned no data for {ticker} at {interval}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if resample:
        ohlc = df[["Open", "High", "Low", "Close"]]
        try:
            rs = ohlc.resample(resample.lower(), label="right", closed="right")
        except ValueError:
            rs = ohlc.resample(resample.upper(), label="right", closed="right")
        close = (
            rs.agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
            .dropna()["Close"]
            .astype(float)
        )
    else:
        close = df["Close"].astype(float)

    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    return close


# ---------------------------------------------------------------------------
# Public router
# ---------------------------------------------------------------------------
def fetch_close(
    ticker: str, timeframe: str, lookback_days: int,
    *, source: Source = "auto", resample: str | None = None,
) -> tuple[pd.Series, str]:
    """Fetch close prices for one ticker/timeframe.

    Parameters
    ----------
    ticker : symbol in Yahoo-style ('BTC-USD', 'SPY')
    timeframe : '15m', '1h', '4h', '1d' etc.
    lookback_days : how far back to pull
    source : 'auto' (router), 'ccxt' (force crypto), 'yfinance' (force)
    resample : optional pandas freq for downsampling (yfinance only)

    Returns
    -------
    (close_series, source_used)
    """
    if source == "auto":
        source = "ccxt" if _is_crypto(ticker) else "yfinance"

    # FAST PATH: for 5m crypto, prefer the sidecar's cache. The refresher
    # writes fresh parquet every 5min — no network call per page load.
    if source == "ccxt" and timeframe == "5m":
        try:
            from cache_5m import load_cached_5m
            cached = load_cached_5m(ticker)
            if cached is not None:
                return cached.close, f"cache_5m({cached.age_minutes:.1f}min)"
        except Exception:
            # cache layer never blocks — fall through to live
            pass

    if source == "ccxt":
        try:
            close = _fetch_ccxt(ticker, timeframe, lookback_days)
            return close, "ccxt"
        except ImportError:
            # ccxt not installed — fall through to yfinance
            pass
        except Exception:
            # data unavailable on Binance → fall back to yfinance
            pass

    close = _fetch_yfinance(ticker, timeframe, lookback_days, resample=resample)
    return close, "yfinance"
