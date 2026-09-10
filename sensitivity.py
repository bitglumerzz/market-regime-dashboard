"""Sensitivity analysis — sweep key parameters and measure robustness.

Idea
----
A strategy that works only at one specific parameter setting is overfit.
We sweep `n_components` × `confidence_threshold` and record several
performance metrics for each combo. The resulting heatmap shows whether
performance is stable across parameter neighborhoods (robust) or peaks
on a single cell (fragile).

We re-use the SAME standardized feature matrix across all combinations,
so the only thing varying is the HMM fit and the confidence threshold.
The HMM is refit once per n_components (cheap caching).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from backtest import run_backtest
from feature_engineering import LOG_RETURN_COL
from hmm_model import GaussianHMM, forward_filter, train_best_hmm
from regime_labeler import apply_stability_filter, label_regimes


# Default grid — kept small so the analysis finishes in well under a minute.
DEFAULT_N_GRID: tuple[int, ...] = (3, 4, 5, 6)
DEFAULT_CONFIDENCE_GRID: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 0.90)


@dataclass
class SensitivityCell:
    n_components: int
    confidence_threshold: float
    sharpe: float
    sortino: float
    calmar: float
    total_return: float
    max_drawdown: float
    time_in_market: float
    n_round_trips: int


def _fit_in_sample_pipeline(
    Xs: np.ndarray, feats: pd.DataFrame, n: int
) -> tuple[pd.Series, pd.Series]:
    """Fit HMM with fixed n_components, return (regime_labels, confidence)."""
    model, _ = train_best_hmm(Xs, n_range=(n, n))
    post = forward_filter(model, Xs)
    state_to_label = label_regimes(model, n)
    raw_labels = pd.Series(
        [state_to_label[int(s)] for s in np.argmax(post, axis=1)],
        index=feats.index,
    )
    smooth_labels = apply_stability_filter(raw_labels)
    confidence = pd.Series(np.max(post, axis=1), index=feats.index)
    return smooth_labels, confidence


def run_sensitivity_grid(
    feats: pd.DataFrame,
    Xs: np.ndarray,
    long_regimes: list[str],
    *,
    n_grid: tuple[int, ...] = DEFAULT_N_GRID,
    confidence_grid: tuple[float, ...] = DEFAULT_CONFIDENCE_GRID,
    cost_bps: float = 5.0,
    slippage_bps: float = 1.0,
    sizing: str = "binary",
    progress_cb: Callable[[float, str], None] | None = None,
) -> list[SensitivityCell]:
    """Sweep (n_components, confidence_threshold) and record metrics per cell.

    Re-uses each fitted HMM across all confidence thresholds for speed.
    """
    cells: list[SensitivityCell] = []
    total_steps = len(n_grid)
    for i, n in enumerate(n_grid):
        if progress_cb is not None:
            progress_cb(i / total_steps, f"Fitting HMM with n={n}…")
        try:
            regime_labels, conf = _fit_in_sample_pipeline(Xs, feats, n)
        except Exception:
            # Degenerate fit — fill the row with NaN-ish values.
            for ct in confidence_grid:
                cells.append(SensitivityCell(
                    n_components=n, confidence_threshold=ct,
                    sharpe=np.nan, sortino=np.nan, calmar=np.nan,
                    total_return=np.nan, max_drawdown=np.nan,
                    time_in_market=np.nan, n_round_trips=0,
                ))
            continue

        df_full = feats.copy()
        df_full["regime"] = regime_labels
        df_full["confidence"] = conf

        for ct in confidence_grid:
            try:
                strat, _ = run_backtest(
                    df_full,
                    long_regimes=long_regimes,
                    cost_bps=cost_bps,
                    slippage_bps=slippage_bps,
                    confidence_threshold=float(ct),
                    sizing=sizing,  # type: ignore[arg-type]
                )
                cells.append(SensitivityCell(
                    n_components=n, confidence_threshold=ct,
                    sharpe=strat.sharpe, sortino=strat.sortino,
                    calmar=strat.calmar,
                    total_return=strat.total_return,
                    max_drawdown=strat.max_drawdown,
                    time_in_market=strat.time_in_market,
                    n_round_trips=strat.n_round_trips,
                ))
            except Exception:
                cells.append(SensitivityCell(
                    n_components=n, confidence_threshold=ct,
                    sharpe=np.nan, sortino=np.nan, calmar=np.nan,
                    total_return=np.nan, max_drawdown=np.nan,
                    time_in_market=np.nan, n_round_trips=0,
                ))

    if progress_cb is not None:
        progress_cb(1.0, "Sensitivity grid complete.")
    return cells


def cells_to_matrix(
    cells: list[SensitivityCell],
    metric: str,
    n_grid: tuple[int, ...],
    confidence_grid: tuple[float, ...],
) -> np.ndarray:
    """Reshape flat cell list into a 2-D matrix [n × confidence]."""
    matrix = np.full((len(n_grid), len(confidence_grid)), np.nan)
    n_to_row = {n: i for i, n in enumerate(n_grid)}
    ct_to_col = {ct: j for j, ct in enumerate(confidence_grid)}
    for c in cells:
        i = n_to_row.get(c.n_components)
        j = ct_to_col.get(c.confidence_threshold)
        if i is None or j is None:
            continue
        matrix[i, j] = getattr(c, metric)
    return matrix


def robustness_score(cells: list[SensitivityCell], metric: str = "sharpe") -> float:
    """Single-number summary: median / IQR of the chosen metric across the grid.

    Higher = the strategy delivers consistent performance across parameter
    neighborhoods. Lower = fragile / overfit.
    """
    vals = np.array([getattr(c, metric) for c in cells], dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 4:
        return 0.0
    q25, q50, q75 = np.percentile(vals, [25, 50, 75])
    iqr = q75 - q25
    if iqr <= 0:
        return float(q50)
    return float(q50 / (1.0 + iqr))  # bounded; 0 ≤ score < median if iqr > 0
