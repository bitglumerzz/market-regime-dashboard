"""Regime-based long/cash backtest with realistic frictions.

Key features
------------
* **No look-ahead**: position[t] = signal(regime[t-1], confidence[t-1]).
* **Transaction costs**: round-trip cost charged on every position change.
* **Slippage**: extra basis points charged the same way.
* **Confidence gating**: positions only opened when posterior confidence ≥ threshold.
* **Sizing modes**:
    - "binary"     — position is 0 or 1
    - "confidence" — position equals the lagged confidence (when in long regime
                     and above threshold)
* **Rich metrics**: Sharpe, Sortino, Calmar, win rate, avg trade return, etc.

All math is in log-return space:
    log_strat_return[t] = position[t] * log_return[t] - cost_indicator[t] * cost_log
where `cost_log = log(1 - cost_fraction) ≈ -cost_fraction` for small costs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np
import pandas as pd


ANNUALIZATION_FACTOR: int = 252

# Defaults — chosen to be conservative-realistic for liquid US ETFs.
DEFAULT_COST_BPS: float = 5.0        # 0.05% per round-trip leg (commission)
DEFAULT_SLIPPAGE_BPS: float = 1.0    # 0.01% slippage per leg
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.0  # off by default

SizingMode = Literal["binary", "confidence"]


@dataclass
class BacktestResult:
    """All results from a single backtest run."""
    equity: pd.Series
    daily_return: pd.Series       # log return per bar (already net of costs)
    position: pd.Series           # actual exposure on each bar
    trades: pd.Series             # 1.0 on bars where position changed

    # Core P&L stats
    total_return: float
    ann_return: float
    ann_vol: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float

    # Trading-quality stats
    win_rate: float               # % of in-position bars with positive net return
    avg_trade_return: float       # mean log return per completed round-trip
    n_trades: int                 # total position changes (entries + exits)
    n_round_trips: int            # full open→close cycles
    time_in_market: float
    total_cost: float             # cumulative cost paid (as fraction)

    label: str = ""
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "total_return": self.total_return,
            "ann_return": self.ann_return,
            "ann_vol": self.ann_vol,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "avg_trade_return": self.avg_trade_return,
            "n_trades": self.n_trades,
            "n_round_trips": self.n_round_trips,
            "time_in_market": self.time_in_market,
            "total_cost": self.total_cost,
        }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _annualized_sortino(simple_ret: pd.Series, ann_return: float) -> float:
    """Sortino = ann_return / (downside std * sqrt(252)).
    Downside std uses only negative simple returns.
    """
    downside = simple_ret[simple_ret < 0]
    if len(downside) < 2:
        return 0.0
    dd_std = float(downside.std(ddof=0))
    if dd_std <= 0:
        return 0.0
    return float(ann_return / (dd_std * np.sqrt(ANNUALIZATION_FACTOR)))


def _round_trip_returns(
    log_strat_return: pd.Series, position: pd.Series
) -> tuple[list[float], float]:
    """Sum of log returns over each contiguous in-position run = round-trip log P&L.

    Returns
    -------
    (list_of_round_trip_log_returns, mean_log_return)
    """
    if position.empty:
        return [], 0.0

    # A "round trip" = entry → exit. Find runs of position > 0.
    pos = position.astype(float).values
    rt_returns: list[float] = []
    run_sum = 0.0
    in_pos = False
    for i, p in enumerate(pos):
        if p > 0 and not in_pos:
            in_pos = True
            run_sum = 0.0
        if in_pos:
            run_sum += float(log_strat_return.iloc[i])
            if p == 0 or i == len(pos) - 1:
                # close out at end-of-run or end-of-series
                rt_returns.append(run_sum)
                in_pos = False
                run_sum = 0.0
    if not rt_returns:
        return [], 0.0
    return rt_returns, float(np.mean(rt_returns))


def _compute_metrics(
    log_strat_return: pd.Series,
    position: pd.Series,
    trades: pd.Series,
    total_cost: float,
    label: str,
    params: dict,
) -> BacktestResult:
    """Turn a (returns, position, trades) triple into a full BacktestResult."""
    # Equity is end-of-bar: equity[t] = exp(sum_{i<=t} log_strat_return[i]).
    # Capital starts at 1.0 *before* bar 0, so equity[-1] == exp(sum(log_returns)).
    # We do NOT normalize equity[0] to 1.0 — that would double-count by stripping
    # off the first bar's return and break the buy-and-hold identity test.
    equity = np.exp(log_strat_return.cumsum())

    simple_ret = np.expm1(log_strat_return)

    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    n_obs = max(len(log_strat_return), 1)
    ann_return = float(
        np.exp(log_strat_return.sum() * ANNUALIZATION_FACTOR / n_obs) - 1.0
    )
    ann_vol = float(simple_ret.std(ddof=0) * np.sqrt(ANNUALIZATION_FACTOR))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0
    sortino = _annualized_sortino(simple_ret, ann_return)

    # Max drawdown.
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0
    calmar = float(ann_return / abs(max_dd)) if max_dd < 0 else 0.0

    # Win rate — only over bars where we held a position.
    held = position > 0
    if held.any():
        held_returns = log_strat_return[held]
        win_rate = float((held_returns > 0).mean())
    else:
        win_rate = 0.0

    # Round-trip stats.
    rt_returns, avg_rt = _round_trip_returns(log_strat_return, position)
    n_round_trips = len(rt_returns)

    n_trades = int(trades.sum())
    time_in_market = float((position > 0).mean()) if len(position) else 0.0

    return BacktestResult(
        equity=equity,
        daily_return=log_strat_return,
        position=position,
        trades=trades,
        total_return=total_return,
        ann_return=ann_return,
        ann_vol=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_dd,
        win_rate=win_rate,
        avg_trade_return=avg_rt,
        n_trades=n_trades,
        n_round_trips=n_round_trips,
        time_in_market=time_in_market,
        total_cost=total_cost,
        label=label,
        params=params,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_backtest(
    df: pd.DataFrame,
    long_regimes: Iterable[str],
    *,
    regime_col: str = "regime",
    log_return_col: str = "log_return",
    confidence_col: str = "confidence",
    cost_bps: float = DEFAULT_COST_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    sizing: SizingMode = "binary",
) -> tuple[BacktestResult, BacktestResult]:
    """Run a regime-gated long/cash strategy alongside buy-and-hold.

    Parameters
    ----------
    df : DataFrame with `regime_col`, `log_return_col`, `confidence_col`.
        Index must be a DatetimeIndex.
    long_regimes : iterable of str
        Regime labels in which the strategy goes long.
    cost_bps, slippage_bps : float
        Round-trip cost in basis points (1 bps = 0.01%). Total per-trade
        cost = (cost_bps + slippage_bps) * 1e-4, applied each time the
        position changes.
    confidence_threshold : float in [0, 1]
        Position is forced to 0 when lagged confidence < threshold.
    sizing : "binary" | "confidence"
        How to size the position when in a long regime.

    Returns
    -------
    (strategy_result, buy_and_hold_result)
    """
    if regime_col not in df.columns:
        raise ValueError(f"Missing column '{regime_col}' in dataframe.")
    if log_return_col not in df.columns:
        raise ValueError(f"Missing column '{log_return_col}' in dataframe.")
    if confidence_col not in df.columns and sizing == "confidence":
        raise ValueError(
            f"Missing column '{confidence_col}'; required for sizing='confidence'."
        )
    long_set = set(long_regimes)
    if not long_set:
        raise ValueError("long_regimes must contain at least one label.")
    if sizing not in ("binary", "confidence"):
        raise ValueError(f"Unknown sizing mode: {sizing!r}.")

    # --- align series, drop initial NaN bar (if any)
    log_ret = df[log_return_col].astype(float).dropna()
    regimes = df.loc[log_ret.index, regime_col]
    conf = (
        df.loc[log_ret.index, confidence_col].astype(float)
        if confidence_col in df.columns
        else pd.Series(1.0, index=log_ret.index)
    )

    # --- raw signal (today's decision, based on today's regime+confidence)
    in_long_regime = regimes.isin(long_set)
    gate = conf >= float(confidence_threshold)
    if sizing == "binary":
        raw_signal = (in_long_regime & gate).astype(float)
    else:  # confidence sizing
        raw_signal = np.where(in_long_regime & gate, conf, 0.0)
        raw_signal = pd.Series(raw_signal, index=log_ret.index)

    # --- lag by one bar to enforce no-look-ahead
    position = raw_signal.shift(1).fillna(0.0)
    position.name = "position"

    # --- trade flag = 1.0 whenever position changes
    pos_change = position.diff().fillna(position.iloc[0])
    trades = (pos_change.abs() > 1e-9).astype(float)
    trades.name = "trades"

    # --- cost in log space (signed only by magnitude — every change costs)
    cost_fraction_per_trade = (cost_bps + slippage_bps) * 1e-4
    # Use exact log(1 - c) — accurate at any plausible cost.
    cost_log_per_trade = np.log1p(-cost_fraction_per_trade)
    cost_series = trades * cost_log_per_trade
    cost_series.name = "cost_log"

    # --- strategy log return
    strat_log_ret = (log_ret * position) + cost_series
    strat_log_ret.name = "strat_log_return"

    total_cost = float(-cost_series.sum())  # positive number = fraction lost

    params = {
        "long_regimes": sorted(long_set),
        "cost_bps": cost_bps,
        "slippage_bps": slippage_bps,
        "confidence_threshold": confidence_threshold,
        "sizing": sizing,
    }

    strat = _compute_metrics(
        log_strat_return=strat_log_ret,
        position=position,
        trades=trades,
        total_cost=total_cost,
        label="Regime-Gated Long/Cash",
        params=params,
    )

    # Buy-and-hold: always long, no trades after initial entry, no cost.
    bh_position = pd.Series(1.0, index=log_ret.index, name="position")
    bh_trades = pd.Series(0.0, index=log_ret.index, name="trades")
    bh_trades.iloc[0] = 1.0  # one notional entry trade
    bh_log_ret = log_ret.copy()
    bh = _compute_metrics(
        log_strat_return=bh_log_ret,
        position=bh_position,
        trades=bh_trades,
        total_cost=0.0,
        label="Buy & Hold",
        params={"long_regimes": ["always"], "cost_bps": 0, "slippage_bps": 0,
                "confidence_threshold": 0, "sizing": "binary"},
    )

    return strat, bh
