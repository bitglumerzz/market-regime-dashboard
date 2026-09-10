"""Monte Carlo bootstrap of a backtest's return distribution.

Given a series of per-bar strategy log-returns, we resample with replacement
to construct N alternative equity paths. The point: any single backtest is
just *one* sample from the distribution of possible outcomes. Bootstrap
gives the full distribution.

Two resampling modes are supported:
* "iid"   — sample bars independently. Standard but breaks autocorrelation.
* "block" — sample contiguous blocks of bars. Preserves serial dependence
            (the right choice when returns are autocorrelated or there is
             volatility clustering).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


DEFAULT_N_SIMULATIONS: int = 1000
DEFAULT_BLOCK_SIZE: int = 5
PERCENTILES: tuple[float, ...] = (5, 25, 50, 75, 95)

ResampleMode = Literal["iid", "block"]


@dataclass
class MonteCarloResult:
    """Container for a bootstrap run."""
    equity_paths: np.ndarray          # (n_simulations, n_bars + 1) — starts at 1.0
    final_values: np.ndarray          # (n_simulations,) — terminal equity
    drawdowns: np.ndarray             # (n_simulations,) — max DD per path
    sharpe_ratios: np.ndarray         # (n_simulations,) — annualized
    n_simulations: int
    n_bars: int
    mode: str
    block_size: int

    def quantiles(self, q: tuple[float, ...] = PERCENTILES) -> dict:
        """Return final-value, drawdown, and Sharpe percentiles."""
        return {
            "final_pct": dict(zip(q, np.percentile(self.final_values, q))),
            "dd_pct":    dict(zip(q, np.percentile(self.drawdowns, q))),
            "sharpe_pct": dict(zip(q, np.percentile(self.sharpe_ratios, q))),
        }

    def equity_band(self, q: tuple[float, ...] = PERCENTILES) -> np.ndarray:
        """Per-bar percentile band of equity paths.

        Returns array of shape (len(q), n_bars + 1).
        """
        return np.percentile(self.equity_paths, q, axis=0)

    def probability_of_loss(self) -> float:
        """Share of paths ending below starting capital."""
        return float((self.final_values < 1.0).mean())

    def probability_below(self, threshold: float) -> float:
        """Share of paths ending below `threshold` × starting capital."""
        return float((self.final_values < threshold).mean())


def _resample_iid(log_returns: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(log_returns)
    return log_returns[rng.integers(0, n, size=n)]


def _resample_blocks(
    log_returns: np.ndarray, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Stationary bootstrap with fixed block length.

    Sample starting indices uniformly, copy `block_size` consecutive bars,
    wrap around if necessary. Concatenate blocks to length n.
    """
    n = len(log_returns)
    out = np.empty(n, dtype=log_returns.dtype)
    i = 0
    while i < n:
        start = int(rng.integers(0, n))
        take = min(block_size, n - i)
        for k in range(take):
            out[i + k] = log_returns[(start + k) % n]
        i += take
    return out


_ANN = 252


def _path_metrics(log_path: np.ndarray) -> tuple[float, float, float]:
    """Return (terminal_equity, max_drawdown_fraction, annualized_sharpe)."""
    eq = np.exp(np.cumsum(log_path))
    eq_with_anchor = np.concatenate([[1.0], eq])
    peak = np.maximum.accumulate(eq_with_anchor)
    dd = (eq_with_anchor / peak - 1.0).min()
    # Sharpe using simple returns derived from logs.
    simple = np.expm1(log_path)
    if simple.std(ddof=0) > 0:
        sharpe = (np.expm1(log_path.sum() * _ANN / len(log_path)) /
                  (simple.std(ddof=0) * np.sqrt(_ANN)))
    else:
        sharpe = 0.0
    return float(eq[-1]), float(dd), float(sharpe)


def run_monte_carlo(
    returns: pd.Series | np.ndarray,
    *,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    mode: ResampleMode = "block",
    block_size: int = DEFAULT_BLOCK_SIZE,
    seed: int | None = 0,
) -> MonteCarloResult:
    """Bootstrap the equity curve.

    Parameters
    ----------
    returns : log-return series (or array) — typically the daily_return field
        of a BacktestResult.
    n_simulations : number of bootstrap paths.
    mode : "iid" or "block".
    block_size : only used when mode == "block".
    seed : RNG seed for reproducibility (None for fresh randomness).
    """
    arr = np.asarray(returns, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 30:
        raise ValueError(f"Need at least 30 return bars; got {len(arr)}.")

    rng = np.random.default_rng(seed)
    n = len(arr)
    equity_paths = np.empty((n_simulations, n + 1), dtype=np.float64)
    finals = np.empty(n_simulations, dtype=np.float64)
    drawdowns = np.empty(n_simulations, dtype=np.float64)
    sharpes = np.empty(n_simulations, dtype=np.float64)

    for s in range(n_simulations):
        if mode == "iid":
            sampled = _resample_iid(arr, rng)
        else:
            sampled = _resample_blocks(arr, block_size, rng)
        cum = np.cumsum(sampled)
        equity_paths[s, 0] = 1.0
        equity_paths[s, 1:] = np.exp(cum)
        finals[s], drawdowns[s], sharpes[s] = _path_metrics(sampled)

    return MonteCarloResult(
        equity_paths=equity_paths,
        final_values=finals,
        drawdowns=drawdowns,
        sharpe_ratios=sharpes,
        n_simulations=n_simulations,
        n_bars=n,
        mode=mode,
        block_size=block_size,
    )
