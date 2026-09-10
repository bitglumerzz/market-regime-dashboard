"""Background refresher for 5m OHLCV data.

Runs continuously inside the `refresher` sidecar container. Every 5 minutes,
fetches the latest 5m bars for BTC / ETH / HYPE from CCXT and writes atomic
parquet files to `cache_5m/`. The main streamlit app reads from this cache,
so chart refreshes are instant — no network call on each page load.

Usage:
    python refresh_5m.py --once    # fetch one cycle and exit (good for tests)
    python refresh_5m.py --loop    # run forever, sleeping 5min between cycles

Cache format: one parquet file per coin at `cache_5m/{TICKER}.parquet` with
columns Open/High/Low/Close/Volume and a UTC DatetimeIndex.

Atomicity: writes to a `.tmp` file first, then os.replace() — guarantees the
streamlit reader never sees a half-written file.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CACHE_DIR = "cache_5m"
CACHE_LOOKBACK_DAYS = 30          # ~8.6k bars at 5m, plenty for Elliott
SLEEP_BETWEEN_CYCLES_S = 300      # 5 minutes
TIMEFRAME = "5m"
SAFETY_BAR_LIMIT = 50_000


@dataclass
class CoinConfig:
    """Where a coin can be sourced. Try `sources` in order until one works."""
    cache_name: str                      # filename in cache_5m/
    sources: list[tuple[str, str]]       # [(exchange_id, symbol)]


COINS: list[CoinConfig] = [
    CoinConfig(
        cache_name="BTC-USD",
        sources=[
            ("binance", "BTC/USDT"),
            ("kucoin",  "BTC/USDT"),
            ("bybit",   "BTC/USDT"),
        ],
    ),
    CoinConfig(
        cache_name="ETH-USD",
        sources=[
            ("binance", "ETH/USDT"),
            ("kucoin",  "ETH/USDT"),
            ("bybit",   "ETH/USDT"),
        ],
    ),
    CoinConfig(
        cache_name="HYPE-USD",
        sources=[
            ("binance",      "HYPE/USDT"),
            ("kucoin",       "HYPE/USDT"),
            ("bybit",        "HYPE/USDT"),
            ("hyperliquid",  "HYPE/USDC"),   # native exchange, last resort
        ],
    ),
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    log = logging.getLogger("refresh_5m")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        log.addHandler(h)
    return log


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------
def _fetch_one_source(
    exchange_id: str, symbol: str, lookback_days: int, log: logging.Logger,
) -> pd.DataFrame:
    """Pull 5m OHLCV from one specific exchange. Raises on failure."""
    import ccxt
    ex_cls = getattr(ccxt, exchange_id, None)
    if ex_cls is None:
        raise ValueError(f"unknown exchange {exchange_id!r}")

    exchange = ex_cls({"enableRateLimit": True})
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int(
        (datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp() * 1000
    )

    bars: list[list] = []
    cursor = start_ms
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME,
                                      since=cursor, limit=1000)
        if not batch:
            break
        bars.extend(batch)
        last_ts = batch[-1][0]
        if last_ts >= end_ms or last_ts == cursor:
            break
        cursor = last_ts + 1
        if len(bars) > SAFETY_BAR_LIMIT:
            break

    if not bars:
        raise ValueError(f"{exchange_id} returned no bars for {symbol}")

    df = pd.DataFrame(bars,
                      columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("Date").drop(columns=["ts"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def fetch_with_fallback(coin: CoinConfig, log: logging.Logger) -> pd.DataFrame:
    """Try each exchange in coin.sources until one succeeds."""
    last_err: Exception | None = None
    for exchange_id, symbol in coin.sources:
        try:
            df = _fetch_one_source(exchange_id, symbol,
                                    CACHE_LOOKBACK_DAYS, log)
            log.info(
                "  ✓ %s ← %s on %s (%d bars, %s → %s)",
                coin.cache_name, symbol, exchange_id, len(df),
                df.index[0].strftime("%Y-%m-%d %H:%M"),
                df.index[-1].strftime("%Y-%m-%d %H:%M"),
            )
            return df
        except Exception as exc:
            last_err = exc
            log.warning("  ✗ %s on %s failed: %s",
                        symbol, exchange_id, type(exc).__name__)
            continue
    raise RuntimeError(
        f"All sources failed for {coin.cache_name}: {type(last_err).__name__}: "
        f"{last_err}"
    ) from last_err


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------
def write_atomic_parquet(df: pd.DataFrame, target_path: str) -> None:
    """Write parquet via temp+rename so concurrent readers never see partial data."""
    os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
    tmp_path = target_path + ".tmp"
    df.to_parquet(tmp_path, engine="pyarrow", compression="snappy")
    os.replace(tmp_path, target_path)


# ---------------------------------------------------------------------------
# One cycle
# ---------------------------------------------------------------------------
def run_one_cycle(log: logging.Logger) -> dict:
    """Refresh all coins once. Returns per-coin status dict."""
    cycle_started_at = time.monotonic()
    log.info("=" * 60)
    log.info("Starting refresh cycle at %s",
              datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    status: dict[str, str] = {}

    for coin in COINS:
        target_path = os.path.join(CACHE_DIR, f"{coin.cache_name}.parquet")
        try:
            df = fetch_with_fallback(coin, log)
            write_atomic_parquet(df, target_path)
            status[coin.cache_name] = "ok"
        except Exception as exc:
            log.error("  ✗ %s — refresh failed: %s",
                      coin.cache_name, exc)
            status[coin.cache_name] = f"error: {exc}"

    elapsed = time.monotonic() - cycle_started_at
    ok_count = sum(1 for v in status.values() if v == "ok")
    log.info("Cycle done in %.1fs · %d/%d coins refreshed",
              elapsed, ok_count, len(COINS))

    # Heartbeat file lets a healthcheck see when last successful cycle ran
    heartbeat = os.path.join(CACHE_DIR, ".heartbeat")
    try:
        with open(heartbeat, "w", encoding="utf-8") as f:
            f.write(datetime.now(timezone.utc).isoformat())
    except Exception:
        pass

    return status


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------
def run_loop(log: logging.Logger) -> None:
    """Run forever, sleeping SLEEP_BETWEEN_CYCLES_S between cycles."""
    log.info("refresh_5m starting in --loop mode "
              "(interval=%ds, coins=%s)",
              SLEEP_BETWEEN_CYCLES_S,
              [c.cache_name for c in COINS])
    while True:
        try:
            run_one_cycle(log)
        except Exception as exc:
            # Never let the loop die — log and continue.
            log.exception("Unhandled error in cycle: %s", exc)
        # Sleep until the next 5-minute boundary so cycles align with bar closes
        now = time.time()
        next_boundary = (int(now) // SLEEP_BETWEEN_CYCLES_S + 1) * SLEEP_BETWEEN_CYCLES_S
        sleep_s = max(30.0, next_boundary - now)
        log.info("Sleeping %.0fs until next cycle…", sleep_s)
        time.sleep(sleep_s)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--once", action="store_true",
                    help="run one refresh cycle and exit (good for tests)")
    g.add_argument("--loop", action="store_true",
                    help="run forever, sleeping between cycles")
    args = parser.parse_args()

    log = _setup_logging()
    if args.once:
        status = run_one_cycle(log)
        # Exit non-zero if any coin failed (useful for CI/healthcheck)
        return 0 if all(v == "ok" for v in status.values()) else 1
    if args.loop:
        run_loop(log)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
