"""Predefined crisis windows for stress-testing the regime strategy.

Each entry is a (label, start_date, end_date, description) tuple. The dashboard
runs the same backtest, restricted to each window, and reports the strategy's
behavior in known stressed environments.

Windows are deliberately a bit wider than the eyeball "crisis" because:
- HMM needs warmup bars BEFORE the crisis to classify it correctly
- post-crisis recovery is part of the story
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from backtest import BacktestResult, run_backtest


@dataclass(frozen=True)
class CrisisWindow:
    label: str
    start: str           # ISO date
    end: str             # ISO date
    description: str


# Curated list. Add more as needed.
CRISIS_WINDOWS: tuple[CrisisWindow, ...] = (
    CrisisWindow(
        label="GFC 2008",
        start="2008-09-01",
        end="2009-06-30",
        description="Lehman collapse → market bottom (Mar 2009) → early recovery.",
    ),
    CrisisWindow(
        label="EU Debt 2011",
        start="2011-07-01",
        end="2011-12-31",
        description="US debt-ceiling downgrade + European sovereign debt crisis.",
    ),
    CrisisWindow(
        label="Vol Spike 2018",
        start="2018-01-15",
        end="2018-04-15",
        description="February 2018 volmageddon: VIX spike, S&P −10% in days.",
    ),
    CrisisWindow(
        label="COVID Crash 2020",
        start="2020-02-15",
        end="2020-06-30",
        description="−34% S&P in ~5 weeks; fastest bear market on record + V-recovery.",
    ),
    CrisisWindow(
        label="Inflation 2022",
        start="2022-01-01",
        end="2022-10-31",
        description="Fed hiking cycle, S&P −25% peak-to-trough, bond rout.",
    ),
    CrisisWindow(
        label="SVB / Banks 2023",
        start="2023-03-01",
        end="2023-05-15",
        description="SVB and Credit Suisse failures, regional banking stress.",
    ),
    CrisisWindow(
        label="Tariff Shock 2025",
        start="2025-03-15",
        end="2025-05-15",
        description="Liberation-Day tariffs (Apr 2–9), S&P ~−18% in a week.",
    ),
)


def slice_window(df: pd.DataFrame, window: CrisisWindow) -> pd.DataFrame:
    """Return rows of `df` inside the crisis window. Empty if no overlap."""
    if df.empty:
        return df.iloc[0:0]
    start = pd.Timestamp(window.start)
    end = pd.Timestamp(window.end)
    return df.loc[(df.index >= start) & (df.index <= end)]


def run_stress_tests(
    df: pd.DataFrame,
    long_regimes: Iterable[str],
    *,
    cost_bps: float,
    slippage_bps: float,
    confidence_threshold: float,
    sizing: str,
    windows: tuple[CrisisWindow, ...] = CRISIS_WINDOWS,
    min_bars: int = 20,
) -> list[dict]:
    """Run backtest on each crisis window that has enough data.

    Returns a list of dicts (one per window) ready to be turned into a Streamlit
    table. Windows without enough overlapping bars are silently skipped.
    """
    long_list = list(long_regimes)
    rows: list[dict] = []
    for w in windows:
        sub = slice_window(df, w)
        if len(sub) < min_bars or not long_list:
            continue
        try:
            strat, bh = run_backtest(
                sub,
                long_regimes=long_list,
                cost_bps=cost_bps,
                slippage_bps=slippage_bps,
                confidence_threshold=confidence_threshold,
                sizing=sizing,  # type: ignore[arg-type]
            )
        except (ValueError, RuntimeError):
            continue
        rows.append({
            "window": w,
            "bars": len(sub),
            "start": sub.index.min().date().isoformat(),
            "end": sub.index.max().date().isoformat(),
            "strat_total": strat.total_return,
            "bh_total": bh.total_return,
            "outperformance": strat.total_return - bh.total_return,
            "strat_dd": strat.max_drawdown,
            "bh_dd": bh.max_drawdown,
            "dd_saved": strat.max_drawdown - bh.max_drawdown,  # positive = strat had smaller DD
            "strat_sharpe": strat.sharpe,
            "bh_sharpe": bh.sharpe,
            "time_in_market": strat.time_in_market,
            "n_trades": strat.n_trades,
            "strat": strat,
            "bh": bh,
        })
    return rows
