"""Run ZigZag + Elliott across three timeframes for one ticker.

The intended use mirrors the Andreev workflow (top-down):
  - Top TF (daily): identify the macro wave structure ("we're in wave 3
    of a larger impulse").
  - Mid TF (4-hour): confirm the local sub-structure inside the daily wave.
  - Low TF (15-minute): time the entry inside the 4h sub-wave.

Each timeframe uses TWO ZigZags of different "degree":
  - MAJOR: large threshold, fewer swings. This is what the Elliott
    classifier evaluates — major impulses on this TF.
  - MINOR: smaller threshold, more swings. Visual context for the user —
    the inner structure inside each major leg.

yfinance has hard limits on intraday history:
  - 15m → last 60 days
  - 1h  → last 730 days
  - 1d  → unlimited
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

import pandas as pd
import yfinance as yf

import numpy as np

from calibrate_from_tv import (
    load_calibration, load_per_ticker, threshold_for, vol_based_threshold,
)
from data_provider import fetch_close
from elliott import WaveCandidate, classify
from forecast import WaveForecast, forecast_next
from zigzag import Swing, detect_zigzag


# Built-in defaults (used when no calibration file exists). Higher major
# thresholds give cleaner Elliott structures; lower minor thresholds
# reveal the sub-wave detail inside each major leg.
#
# 4 timeframes covering: macro (1d) → meso (4h) → micro (15m) → entry (5m).
# 5m is the live-decision timeframe; it inherits 15m's data limits and
# tightens thresholds further for fine entry timing.
_BUILTIN_DEFAULTS: tuple[dict, ...] = (
    {
        "name": "1d", "interval": "1d", "lookback_days": 730,
        "major_threshold_pct": 0.12,
        "minor_threshold_pct": 0.05,
    },
    {
        "name": "4h", "interval": "1h", "lookback_days": 180,
        "major_threshold_pct": 0.06,
        "minor_threshold_pct": 0.025,
        "resample": "4h",
    },
    {
        "name": "15m", "interval": "15m", "lookback_days": 50,
        "major_threshold_pct": 0.025,
        "minor_threshold_pct": 0.010,
    },
    {
        "name": "5m", "interval": "5m", "lookback_days": 30,
        "major_threshold_pct": 0.012,
        "minor_threshold_pct": 0.005,
    },
)


def _apply_calibration() -> tuple[dict, ...]:
    """Overlay calibrated thresholds (from elliott_calibration.json) onto defaults.

    If a TV-calibrated value exists for a given timeframe, use it as the
    major threshold. The minor threshold is scaled proportionally to keep
    the major/minor ratio close to the built-in 2.4×.
    """
    cal = load_calibration()
    if not cal:
        return _BUILTIN_DEFAULTS
    out: list[dict] = []
    for cfg in _BUILTIN_DEFAULTS:
        new = dict(cfg)
        tf_name = cfg["name"]
        if tf_name in cal:
            calibrated_major = cal[tf_name]
            ratio = cfg["minor_threshold_pct"] / cfg["major_threshold_pct"]
            new["major_threshold_pct"] = calibrated_major
            new["minor_threshold_pct"] = round(calibrated_major * ratio, 4)
            new["_calibrated"] = True
        out.append(new)
    return tuple(out)


# DEFAULT_TF_CONFIGS is the public config — calibrated overrides applied
# automatically on import.
DEFAULT_TF_CONFIGS: tuple[dict, ...] = _apply_calibration()


@dataclass
class TFAnalysis:
    """Result for one timeframe."""
    name: str                          # display label
    interval: str                      # yfinance interval string
    prices: pd.Series                  # close prices
    major_swings: list[Swing]          # large ZigZag — drives classification
    minor_swings: list[Swing]          # small ZigZag — for visual context
    candidates: list[WaveCandidate]    # ranked Elliott hypotheses on majors
    major_threshold_pct: float
    minor_threshold_pct: float
    source: str = "yfinance"           # 'ccxt' for crypto, 'yfinance' otherwise
    threshold_source: str = "builtin"  # per_ticker / vol_formula / aggregate / builtin
    forecast: WaveForecast | None = None  # what comes next after the labeled structure
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def top(self) -> WaveCandidate | None:
        return self.candidates[0] if self.candidates else None


def _fetch_intraday(
    ticker: str, interval: str, lookback_days: int,
    resample: str | None = None,
) -> pd.Series:
    """Pull intraday data with yfinance's tight constraints, then resample."""
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise ValueError(
            f"No data for {ticker} at {interval} (last {lookback_days}d). "
            "Intraday history is limited by Yahoo."
        )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if resample:
        # Resample to a coarser bar (e.g. 1h → 4h). pandas ≥ 2.2 deprecates
        # uppercase frequency aliases ('H' → 'h'); pandas 1.x prefers
        # uppercase. We try lowercase first (modern) and fall back to upper.
        ohlc = df[["Open", "High", "Low", "Close"]]
        try:
            resampled = ohlc.resample(resample.lower(),
                                        label="right", closed="right")
        except ValueError:
            resampled = ohlc.resample(resample.upper(),
                                        label="right", closed="right")
        close = (
            resampled
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
            .dropna()["Close"]
            .astype(float)
        )
    else:
        close = df["Close"].astype(float)
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    return close


def _fetch_daily(ticker: str, lookback_days: int) -> pd.Series:
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise ValueError(f"No daily data for {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df["Close"].astype(float)
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    return close


def analyze_ticker(
    ticker: str,
    *,
    tf_configs: Iterable[dict] = DEFAULT_TF_CONFIGS,
    source: str = "auto",
) -> list[TFAnalysis]:
    """Run two-degree ZigZag + Elliott classification on each TF.

    Parameters
    ----------
    source : 'auto' | 'ccxt' | 'yfinance'
        'auto' = use CCXT for crypto tickers (BTC-USD, ETH-USD, …), else yfinance.
        'ccxt' = force CCXT (only works for crypto pairs on Binance).
        'yfinance' = force yfinance (works for stocks/ETFs and some crypto,
        but with worse intraday history for crypto).
    """
    results: list[TFAnalysis] = []
    ticker = ticker.strip().upper()

    for cfg in tf_configs:
        name = cfg["name"]
        interval = cfg["interval"]
        lookback = cfg["lookback_days"]
        major_thr = cfg["major_threshold_pct"]
        minor_thr = cfg["minor_threshold_pct"]
        resample = cfg.get("resample")

        # Threshold resolution waterfall:
        #   1) per-ticker exact match
        #   2) vol-formula (Stage 5) — linear fit on realized volatility
        #   3) aggregate by TF
        #   4) built-in default (already in cfg)
        per_ticker_data = load_per_ticker()
        threshold_source = "builtin"
        ratio = minor_thr / max(major_thr, 1e-9)

        # 1) per-ticker
        for candidate_name in (ticker, ticker.replace("-USD", "-USDT")):
            if (candidate_name in per_ticker_data
                    and name in per_ticker_data[candidate_name]):
                t_thr = per_ticker_data[candidate_name][name]
                major_thr = t_thr
                minor_thr = max(t_thr * ratio, 0.001)
                threshold_source = "per_ticker"
                break

        try:
            tf_request = name if resample else interval
            close, used_source = fetch_close(
                ticker, tf_request, lookback,
                source=source,  # type: ignore[arg-type]
                resample=resample,
            )
        except Exception as exc:
            results.append(TFAnalysis(
                name=name, interval=interval, prices=pd.Series(dtype=float),
                major_swings=[], minor_swings=[], candidates=[],
                major_threshold_pct=major_thr, minor_threshold_pct=minor_thr,
                threshold_source=threshold_source,
                forecast=None,
                error=f"{type(exc).__name__}: {exc}",
            ))
            continue

        # 2) Vol-formula fallback — only fires if per-ticker didn't match
        # AND we have enough bars to compute a stable daily-vol estimate.
        if threshold_source == "builtin" and len(close) >= 30:
            try:
                log_ret = np.log(close / close.shift(1)).dropna()
                if len(log_ret) >= 20:
                    # Convert any TF's stddev to a 1d-equivalent volatility:
                    # log returns scale by sqrt of bars-per-day.
                    bars_per_day_lookup = {
                        "1d": 1, "4h": 6, "1h": 24, "15m": 96, "5m": 288,
                    }
                    bpd = bars_per_day_lookup.get(name, 1)
                    daily_vol_pct = float(log_ret.std() * np.sqrt(bpd) * 100)
                    v_thr = vol_based_threshold(daily_vol_pct, "1d")
                    if v_thr is not None:
                        major_thr = v_thr
                        minor_thr = max(v_thr * ratio, 0.001)
                        threshold_source = "vol_formula"
            except Exception:
                pass  # never let vol-formula failure block analysis

        # 3) Aggregate-by-TF fallback (the cfg already carries it if any
        # calibration was loaded at module import; otherwise it's the builtin).
        # Resolution-source tagging: if cfg was flagged _calibrated and we
        # haven't already routed via per_ticker/vol_formula, the current
        # threshold is the aggregate.
        if threshold_source == "builtin" and cfg.get("_calibrated"):
            threshold_source = "aggregate"

        major_swings = detect_zigzag(close, threshold_pct=major_thr)
        minor_swings = detect_zigzag(close, threshold_pct=minor_thr)
        candidates = classify(major_swings, top_k=3)

        # Forecast next move from the top hypothesis.
        forecast = forecast_next(candidates[0]) if candidates else None

        results.append(TFAnalysis(
            name=name, interval=interval, prices=close,
            major_swings=major_swings, minor_swings=minor_swings,
            candidates=candidates,
            major_threshold_pct=major_thr, minor_threshold_pct=minor_thr,
            source=used_source,
            threshold_source=threshold_source,
            forecast=forecast,
        ))

    return results


# ---------------------------------------------------------------------------
# Top-down synthesis — what does the combined picture say?
# ---------------------------------------------------------------------------
def top_down_synthesis(results: list[TFAnalysis]) -> dict:
    """Synthesize the trading setup from up to 4 TFs in classic top-down order.

    Returns a dict with:
      - macro_view : 1d  (bias)
      - meso_view  : 4h  (sub-structure)
      - micro_view : 15m (mid-structure)
      - entry_view : 5m  (entry timing, if available — None if 5m TF absent)
      - setup      : one-liner verdict ("LONG opportunity", "stand aside", etc.)
      - color      : 'long' | 'short' | 'wait'
    """
    by_name = {r.name: r for r in results if r.ok and r.top is not None}
    top_1d  = by_name.get("1d", None)
    top_4h  = by_name.get("4h", None)
    top_15m = by_name.get("15m", None)
    top_5m  = by_name.get("5m", None)

    macro = top_1d.top  if top_1d  else None
    meso  = top_4h.top  if top_4h  else None
    micro = top_15m.top if top_15m else None
    entry = top_5m.top  if top_5m  else None

    def short(c: WaveCandidate | None) -> str:
        if c is None:
            return "—"
        return f"{c.pattern.replace('_', ' ')} (score {c.score:.0f})"

    macro_view = short(macro)
    meso_view  = short(meso)
    micro_view = short(micro)
    entry_view = short(entry) if entry is not None else None

    # Trading logic — 4-stage funnel:
    #   1. macro defines BIAS         (must be a clean impulse, score ≥ 50)
    #   2. meso confirms SUB-STRUCTURE (pull-back or continuation)
    #   3. micro times the SETUP      (correction completing or already running)
    #   4. entry (5m) is the LIVE TIMING — only used if 5m TF is provided
    setup = "Stand aside — picture unclear"
    color = "wait"

    def _entry_confirms(direction: str) -> bool | None:
        """Return True if the 5m TF supports the direction, False if against,
        None if entry TF not available / inconclusive."""
        if entry is None:
            return None
        # For LONG: 5m should be either a completed down-correction OR an up impulse.
        if direction == "up":
            if "correction_down" in entry.pattern and "completed" in entry.current_position:
                return True
            if "impulse_up" in entry.pattern and entry.score >= 50:
                return True
            return False
        else:  # short
            if "correction_up" in entry.pattern and "completed" in entry.current_position:
                return True
            if "impulse_down" in entry.pattern and entry.score >= 50:
                return True
            return False

    if macro and "impulse_up" in macro.pattern and macro.score >= 50:
        # Long bias from the top.
        if (meso and meso.score >= 40
                and ("correction_down" in meso.pattern
                     or "impulse_up" in meso.pattern)):
            if (micro and "correction_down" in micro.pattern
                    and "completed" in micro.current_position):
                # 15m correction done — check 5m for final timing.
                ec = _entry_confirms("up")
                if ec is True:
                    setup = "🎯 LONG ENTRY NOW: 1d up, 4h pull-back, 15m done, 5m confirms"
                    color = "long"
                elif ec is None:
                    setup = "LONG opportunity: 1d up, 4h pull-back, 15m correction completed"
                    color = "long"
                else:
                    setup = "Close to LONG: macro aligned, wait for 5m to confirm"
                    color = "wait"
            elif micro and "impulse_up" in micro.pattern:
                ec = _entry_confirms("up")
                if ec is False:
                    setup = "Running long but 5m showing correction — partial profit?"
                    color = "wait"
                else:
                    setup = "Already running long — 1d/4h/15m all up"
                    color = "long"
            else:
                setup = "Wait for 15m correction to complete"
                color = "wait"
        else:
            setup = "1d up but 4h structure unclear — wait"
            color = "wait"

    elif macro and "impulse_down" in macro.pattern and macro.score >= 50:
        if (meso and meso.score >= 40
                and ("correction_up" in meso.pattern
                     or "impulse_down" in meso.pattern)):
            if (micro and "correction_up" in micro.pattern
                    and "completed" in micro.current_position):
                ec = _entry_confirms("down")
                if ec is True:
                    setup = "🎯 SHORT ENTRY NOW: 1d down, 4h bounce, 15m done, 5m confirms"
                    color = "short"
                elif ec is None:
                    setup = "SHORT opportunity: 1d down, 4h bounce, 15m correction completed"
                    color = "short"
                else:
                    setup = "Close to SHORT: macro aligned, wait for 5m to confirm"
                    color = "wait"
            elif micro and "impulse_down" in micro.pattern:
                ec = _entry_confirms("down")
                if ec is False:
                    setup = "Running short but 5m showing bounce — partial profit?"
                    color = "wait"
                else:
                    setup = "Already running short — 1d/4h/15m all down"
                    color = "short"
            else:
                setup = "Wait for 15m correction to complete"
                color = "wait"
        else:
            setup = "1d down but 4h structure unclear — wait"
            color = "wait"

    elif macro and "correction" in macro.pattern and macro.score >= 50:
        setup = (
            "1d is in a correction — historically a poor environment for "
            "trend-following. Wait for the next clean impulse."
        )
        color = "wait"

    return {
        "macro_view": macro_view,
        "meso_view": meso_view,
        "micro_view": micro_view,
        "entry_view": entry_view,
        "setup": setup,
        "color": color,
    }
