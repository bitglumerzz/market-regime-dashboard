"""TradingView chart screenshot harvester.

Triggered after a Claude holistic prediction is logged: navigates to the
relevant TV chart for that ticker+timeframe and saves a snapshot image
to `predictions/screenshots/{prediction_id}.png` for visual verification.

Two backends are supported (auto-selected):
  1. tradingview-mcp (if installed) — uses Chrome DevTools Protocol bridge
     for headless TV navigation
  2. tv-snapshot REST proxies (CoinGecko, CryptoCompare) — fallback if MCP
     is absent

For our use this is BEST-EFFORT — if neither backend is available, we
silently skip without breaking the prediction flow.

The captured PNG goes alongside the prediction's JSONL record so when you
inspect a setup later you have BOTH:
  - the structured prediction (entry/stop/target/reasoning)
  - what the chart looked like at that exact moment
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests

SHOTS_DIR = "predictions/screenshots"

# Yahoo-style → TV URL
TICKER_TO_TV_PATH = {
    "BTC-USD":   "BINANCE:BTCUSDT",
    "ETH-USD":   "BINANCE:ETHUSDT",
    "HYPE-USD":  "BYBIT:HYPEUSDT",
    "SOL-USD":   "BINANCE:SOLUSDT",
    "ADA-USD":   "BINANCE:ADAUSDT",
    "AVAX-USD":  "BINANCE:AVAXUSDT",
    "DOGE-USD":  "BINANCE:DOGEUSDT",
    "DOT-USD":   "BINANCE:DOTUSDT",
    "LINK-USD":  "BINANCE:LINKUSDT",
    "MATIC-USD": "BINANCE:MATICUSDT",
    "XRP-USD":   "BINANCE:XRPUSDT",
    "LTC-USD":   "BINANCE:LTCUSDT",
    "BNB-USD":   "BINANCE:BNBUSDT",
    "TRX-USD":   "BINANCE:TRXUSDT",
    "ATOM-USD":  "BINANCE:ATOMUSDT",
    "NEAR-USD":  "BINANCE:NEARUSDT",
    "TON-USD":   "BINANCE:TONUSDT",
    "ARB-USD":   "BINANCE:ARBUSDT",
    "OP-USD":    "BINANCE:OPUSDT",
    "APT-USD":   "BINANCE:APTUSDT",
    "SUI-USD":   "BINANCE:SUIUSDT",
    "INJ-USD":   "BINANCE:INJUSDT",
    "SEI-USD":   "BINANCE:SEIUSDT",
    "WLD-USD":   "BINANCE:WLDUSDT",
    "FET-USD":   "BINANCE:FETUSDT",
    "RNDR-USD":  "BINANCE:RNDRUSDT",
}


def _resolve_tv_symbol(ticker: str) -> str:
    """Translate any Yahoo ticker to a TradingView-compatible symbol path.

    1) Exact map lookup (highest fidelity, can use BYBIT/KuCoin overrides)
    2) Generic crypto fallback: XXX-USD → BINANCE:XXXUSDT
    3) Generic crypto with USDT suffix: XXX-USDT → BINANCE:XXXUSDT
    4) Pass-through (lets TV resolve raw symbols like 'EURUSD', 'AAPL')
    """
    s = ticker.strip().upper()
    if s in TICKER_TO_TV_PATH:
        return TICKER_TO_TV_PATH[s]
    if s.endswith("-USD"):
        return f"BINANCE:{s.replace('-USD', 'USDT')}"
    if s.endswith("-USDT"):
        return f"BINANCE:{s.replace('-', '')}"
    return s
# Our TF → TV interval string
TF_TO_TV_INTERVAL = {
    "1d":  "1D",  "1D":  "1D",
    "4h":  "240", "4H":  "240",
    "1h":  "60",  "1H":  "60",
    "15m": "15",
    "5m":  "5",
}


def _ensure_dir() -> None:
    os.makedirs(SHOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Backend 1: tradingview-mcp (Chrome DevTools Protocol bridge)
# ---------------------------------------------------------------------------
def _capture_via_mcp(ticker: str, tf: str, out_path: str) -> bool:
    """If a TV MCP is wired locally, navigate the chart and screenshot it.

    The user's setup typically exposes TV MCP via stdio. We discover it by
    looking for the canonical CLI helper `tv-screenshot` or by importing
    the user's local tv_mcp_client module if present. Returns True on success.
    """
    # Try a thin local CLI helper first (most setups have this exposed)
    helper_candidates = ["tv-screenshot", "tv_mcp_screenshot"]
    for helper in helper_candidates:
        if _command_exists(helper):
            try:
                import subprocess
                tv_symbol = TICKER_TO_TV_PATH.get(ticker, ticker)
                interval = TF_TO_TV_INTERVAL.get(tf, "1D")
                result = subprocess.run(
                    [helper, "--symbol", tv_symbol, "--interval", interval,
                     "--out", out_path],
                    capture_output=True, timeout=60,
                )
                if result.returncode == 0 and os.path.isfile(out_path):
                    return True
            except Exception:
                continue
    return False


def _command_exists(cmd: str) -> bool:
    from shutil import which
    return which(cmd) is not None


# ---------------------------------------------------------------------------
# Backend 2: chart-img.com public API (free tier, no auth)
# ---------------------------------------------------------------------------
def _capture_via_chart_img(ticker: str, tf: str, out_path: str) -> bool:
    """Free fallback: chart-img.com renders a TradingView-style chart image
    for any public symbol. Decent quality, no Pine indicator overlay.
    """
    try:
        tv_symbol = _resolve_tv_symbol(ticker)
        interval = TF_TO_TV_INTERVAL.get(tf, "1D")
        url = (
            f"https://api.chart-img.com/v1/tradingview/advanced-chart"
            f"?symbol={tv_symbol}&interval={interval}&theme=dark"
            f"&width=1280&height=720"
        )
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and r.content:
            with open(out_path, "wb") as f:
                f.write(r.content)
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------
def _is_tv_desktop_alive(timeout: float = 2.0) -> bool:
    """Quick probe of the Chrome DevTools port that TradingView Desktop
    must expose for the bridge to work. Without TV running with
    --remote-debugging-port=9222, every bridge request will timeout after
    ~200s — better to detect this fast and fall back to chart-img.com.

    Probes the host where the bridge daemon runs. From inside Docker that
    is host.docker.internal on macOS/Windows, falling back to localhost.
    """
    for host in ("host.docker.internal", "localhost"):
        try:
            r = requests.get(f"http://{host}:9222/json/version",
                              timeout=timeout)
            if r.status_code == 200:
                return True
        except Exception:
            continue
    return False


def _capture_via_host_bridge(ticker: str, tf: str, out_path: str) -> bool:
    """Best backend: file-based bridge to tv_bridge.py running on host.

    Bridge calls TV MCP via Chrome DevTools to TradingView Desktop, producing
    a Pine-indicator-rich screenshot. Requires `tv_bridge.py` running on the
    Mac host with TV Desktop launched on port 9222.

    Skips immediately when TV Desktop's CDP port is unreachable — otherwise
    every TF would block for ~200s before failing.
    """
    try:
        from tv_bridge_client import request_screenshot, is_bridge_available
        if not is_bridge_available():
            return False
        # Skip the bridge entirely if TV Desktop isn't listening on the CDP
        # port — there's nothing for the bridge to talk to.
        if not _is_tv_desktop_alive(timeout=2.0):
            return False
        # Wait up to 200s — chart switch + Pine indicator rebuild + screenshot
        return request_screenshot(ticker, tf, out_path, timeout=200.0)
    except Exception:
        return False


def _capture_via_plotly(ticker: str, tf: str, out_path: str,
                          prices=None, major_swings=None,
                          minor_swings=None) -> bool:
    """Fully-local fallback: render our own Plotly chart to PNG via kaleido.

    Works without TV Desktop and without external API. Coordinates produced
    here are identical to what tv_annotate expects (both use the same OHLC
    window), so PIL overlay is pixel-accurate.
    """
    if prices is None or len(prices) == 0:
        return False
    try:
        from plotly_snapshot import render_tf_snapshot
        result = render_tf_snapshot(
            ticker=ticker, tf=tf, prices=prices,
            major_swings=major_swings, minor_swings=minor_swings,
            out_path=out_path,
        )
        return result is not None
    except Exception:
        return False


def diagnose_capture(ticker: str, tf: str,
                      prices=None, major_swings=None,
                      minor_swings=None) -> dict:
    """Return per-backend diagnostic info — what failed and why.

    Useful when capture() returns None and you need to surface the real
    reason in the UI instead of a generic ‘all backends failed’ message.
    """
    diag: dict[str, str] = {}

    # Backend 1 — host bridge
    try:
        from tv_bridge_client import is_bridge_available
        if not is_bridge_available():
            diag["bridge"] = "tv_bridge_queue/ not mounted in container"
        elif not _is_tv_desktop_alive(timeout=2.0):
            diag["bridge"] = ("TV Desktop CDP port 9222 unreachable "
                              "(TV not open OR not launched with "
                              "--remote-debugging-port=9222)")
        else:
            diag["bridge"] = "would attempt (bridge alive)"
    except Exception as e:
        diag["bridge"] = f"{type(e).__name__}: {e}"

    # Backend 2 — local CLI
    diag["local_cli"] = ("not installed" if not _command_exists("tv-screenshot")
                         else "available")

    # Backend 3 — chart-img.com
    try:
        tv_symbol = _resolve_tv_symbol(ticker)
        interval = TF_TO_TV_INTERVAL.get(tf, "1D")
        url = (f"https://api.chart-img.com/v1/tradingview/advanced-chart"
               f"?symbol={tv_symbol}&interval={interval}&theme=dark"
               f"&width=400&height=300")
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and r.content:
            diag["chart_img"] = f"OK ({len(r.content)} bytes)"
        else:
            diag["chart_img"] = (f"HTTP {r.status_code}, "
                                   f"body: {r.text[:120]}")
    except Exception as e:
        diag["chart_img"] = f"{type(e).__name__}: {e}"

    # Backend 4 — Plotly + kaleido + matplotlib fallback chain
    try:
        from plotly_snapshot import _has_kaleido, _has_matplotlib
        kaleido_ok = _has_kaleido()
        matplotlib_ok = _has_matplotlib()

        if prices is None or len(prices) == 0:
            diag["plotly"] = (f"kaleido={kaleido_ok}, matplotlib={matplotlib_ok}"
                               " — BUT no prices supplied by caller")
        else:
            # Try the actual render
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                test_path = f.name
            try:
                from plotly_snapshot import render_tf_snapshot
                result = render_tf_snapshot(
                    ticker=ticker, tf=tf, prices=prices,
                    major_swings=major_swings, minor_swings=minor_swings,
                    out_path=test_path,
                )
                if result and os.path.isfile(test_path) \
                        and os.path.getsize(test_path) > 1024:
                    size_kb = os.path.getsize(test_path) // 1024
                    used = "kaleido" if kaleido_ok else "matplotlib"
                    diag["plotly"] = (f"OK ({used}) renders {size_kb}KB "
                                       f"PNG  · kaleido={kaleido_ok} "
                                       f"matplotlib={matplotlib_ok}")
                else:
                    diag["plotly"] = (f"render returned no usable file  · "
                                       f"kaleido={kaleido_ok} "
                                       f"matplotlib={matplotlib_ok}")
            finally:
                try:
                    os.remove(test_path)
                except OSError:
                    pass
    except Exception as e:
        diag["plotly"] = f"{type(e).__name__}: {e}"

    return diag


def capture(prediction_id: str, ticker: str, tf: str = "1d",
             force: bool = False, *,
             prices=None, major_swings=None,
             minor_swings=None) -> Optional[str]:
    """Capture a chart screenshot for the given prediction. Best-effort.

    Args:
        prediction_id: used to form filename {prediction_id}.png
        ticker: e.g. BTC-USD
        tf: 1d / 4h / 15m / 5m
        force: if True, re-capture even when a file already exists
               (removes the old file first so we get a fresh one)
        prices, major_swings, minor_swings: optional Plotly fallback data —
            when external backends fail, we render our own Plotly chart as
            PNG (requires kaleido). Pass r.prices/r.major_swings/r.minor_swings
            from a TFAnalysis to enable this.

    Returns the saved file path on success, None on failure.
    Backend priority (best fidelity first):
      1. Host bridge → TV MCP → TradingView Desktop  (Pine indicators visible)
      2. Local tv-screenshot CLI helper              (legacy fallback)
      3. chart-img.com REST                          (basic TV chart, no Pine)
      4. Plotly snapshot via kaleido                 (always works offline)
    """
    _ensure_dir()
    out_path = os.path.join(SHOTS_DIR, f"{prediction_id}.png")
    if os.path.isfile(out_path):
        if not force:
            return out_path
        # force=True: delete old PNG so bridge/chart-img produces a fresh one
        try:
            os.remove(out_path)
        except OSError:
            pass
    if _capture_via_host_bridge(ticker, tf, out_path):
        return out_path
    if _capture_via_mcp(ticker, tf, out_path):
        return out_path
    if _capture_via_chart_img(ticker, tf, out_path):
        return out_path
    if _capture_via_plotly(ticker, tf, out_path,
                            prices=prices, major_swings=major_swings,
                            minor_swings=minor_swings):
        return out_path
    return None


def screenshot_exists(prediction_id: str) -> bool:
    return os.path.isfile(os.path.join(SHOTS_DIR, f"{prediction_id}.png"))


def screenshot_path(prediction_id: str) -> str:
    return os.path.join(SHOTS_DIR, f"{prediction_id}.png")
