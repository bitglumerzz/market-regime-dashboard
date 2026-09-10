"""Run the full regime-pipeline across multiple tickers and compare.

The point of this module: a regime model that works great on one asset class
may fail on another. Some tickers have cleanly separable vol regimes (SPY,
GLD); some are too noisy (single small-cap stocks); some have a single
regime most of the time (TLT in a calm rate environment). This lets you
see at a glance which assets the framework actually helps on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from backtest import BacktestResult, run_backtest
from data_loader import load_data
from feature_engineering import compute_features, feature_matrix
from hmm_model import forward_filter, train_best_hmm
from regime_labeler import apply_stability_filter, label_regimes


# Default basket — broad-strokes coverage of asset classes.
DEFAULT_TICKERS: tuple[str, ...] = (
    "SPY",      # US large-cap equity
    "QQQ",      # US tech
    "GLD",      # Gold
    "TLT",      # Long US Treasuries
    "BTC-USD",  # Crypto
)


@dataclass
class AssetResult:
    ticker: str
    df: pd.DataFrame             # full features + regime + confidence
    strat: BacktestResult
    bh: BacktestResult
    best_n: int
    source: str
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def _run_single(
    ticker: str,
    start: str,
    end: str,
    n_range: tuple[int, int],
    criterion: str,
    long_regimes: list[str],
    cost_bps: float,
    slippage_bps: float,
    confidence_threshold: float,
    sizing: str,
) -> AssetResult:
    """Run the entire in-sample pipeline + backtest for one ticker."""
    try:
        raw = load_data(ticker, start, end)
        feats = compute_features(raw)
        if len(feats) < 60:
            return AssetResult(
                ticker=ticker, df=pd.DataFrame(),
                strat=None, bh=None,  # type: ignore[arg-type]
                best_n=0, source=raw.attrs.get("source", ""),
                error=f"Only {len(feats)} feature rows — need ≥60.",
            )
        X = feature_matrix(feats)
        Xs = StandardScaler().fit_transform(X)
        model, best_n = train_best_hmm(Xs, n_range=n_range, criterion=criterion.lower())
        post = forward_filter(model, Xs)
        state_to_label = label_regimes(model, best_n)
        raw_labels = pd.Series(
            [state_to_label[int(s)] for s in np.argmax(post, 1)],
            index=feats.index,
        )
        smooth = apply_stability_filter(raw_labels)
        df_full = feats.copy()
        df_full["regime"] = smooth
        df_full["confidence"] = np.max(post, axis=1)

        strat, bh = run_backtest(
            df_full,
            long_regimes=long_regimes,
            cost_bps=cost_bps, slippage_bps=slippage_bps,
            confidence_threshold=confidence_threshold,
            sizing=sizing,  # type: ignore[arg-type]
        )
        return AssetResult(
            ticker=ticker, df=df_full, strat=strat, bh=bh,
            best_n=best_n, source=raw.attrs.get("source", ""),
        )
    except Exception as exc:  # data fetch failure, degenerate model, etc.
        return AssetResult(
            ticker=ticker, df=pd.DataFrame(),
            strat=None, bh=None,  # type: ignore[arg-type]
            best_n=0, source="",
            error=f"{type(exc).__name__}: {exc}",
        )


def run_multi_asset(
    tickers: Iterable[str],
    start: str,
    end: str,
    *,
    n_range: tuple[int, int],
    criterion: str,
    long_regimes: list[str],
    cost_bps: float,
    slippage_bps: float,
    confidence_threshold: float,
    sizing: str,
    progress_cb: Callable[[float, str], None] | None = None,
) -> list[AssetResult]:
    """Run the pipeline for each ticker and return a list of AssetResults.

    Runs are independent and sequential (yfinance has per-thread quirks; the
    serial path is also kinder to Yahoo's rate-limits).
    """
    tickers_list = [t.strip().upper() for t in tickers if t.strip()]
    n = len(tickers_list)
    results: list[AssetResult] = []
    for i, t in enumerate(tickers_list):
        if progress_cb is not None:
            progress_cb(i / n, f"Fetching {t}…")
        res = _run_single(
            t, start, end, n_range, criterion,
            long_regimes, cost_bps, slippage_bps,
            confidence_threshold, sizing,
        )
        results.append(res)
    if progress_cb is not None:
        progress_cb(1.0, "Done.")
    return results
