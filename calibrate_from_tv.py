"""Calibrate the Elliott detector against TradingView Pine-indicator ground truth.

Workflow
--------
1. User runs Claude Code with `tradesdontlie/tradingview-mcp` installed locally,
   loads a TradingView Elliott Wave Pine indicator, and dumps its labels to a CSV
   under `tv_ground_truth/<ticker>_<tf>.csv`. See `tv_ground_truth/README.md`.

2. This module loads those CSVs, sweeps our `major_threshold_pct` across a grid,
   and finds the value that maximizes F1 against the TV labeling.

3. Best parameters per timeframe are saved to `elliott_calibration.json`.

4. multi_tf.py reads that JSON at import time and uses the calibrated thresholds.

Ground truth CSV format
-----------------------
    date,close,wave_label
    2024-01-15,500.00,
    2024-01-17,502.00,
    2024-01-19,498.00,0
    2024-02-01,520.00,1
    2024-02-09,510.00,2
    ...

`wave_label` is empty on most bars; non-empty only at swing endpoints (0..5 or
0/A/B/C). Empty cells are read as NaN.

What "match" means
------------------
A TV-labeled bar matches one of our detected swings if there's a swing within
± `match_tolerance_bars` of that date. Recall = matched_tv / total_tv. Precision
= matched_tv / total_our_swings (capped). F1 = harmonic mean.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from zigzag import Swing, detect_zigzag


CALIBRATION_FILE: str = "elliott_calibration.json"

# Grid we sweep when looking for the best threshold. Adjustable by callers.
DEFAULT_THRESHOLD_GRID: tuple[float, ...] = tuple(
    round(x / 100.0, 4)
    for x in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 22.0)
)

# Bars of tolerance when matching detector swings to TV labels.
DEFAULT_MATCH_TOLERANCE: int = 2


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------
@dataclass
class GroundTruth:
    """Parsed ground truth for one (ticker, timeframe) pair."""
    ticker: str
    timeframe: str
    prices: pd.Series              # close, indexed by date
    label_bars: pd.Series          # subset where wave_label is non-empty
    raw: pd.DataFrame

    @property
    def n_labels(self) -> int:
        return len(self.label_bars)


def load_ground_truth(path: str | Path) -> GroundTruth:
    """Load a TV ground-truth CSV into a GroundTruth record."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ground-truth CSV not found: {p}")

    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    needed = {"date", "close", "wave_label"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            f"{p.name}: missing columns {missing}. "
            "Required header row: date,close,wave_label"
        )

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])

    # Parse ticker / timeframe from filename: e.g. "ETH-USD_1d.csv".
    stem = p.stem
    if "_" in stem:
        ticker, tf = stem.rsplit("_", 1)
    else:
        ticker, tf = stem, "?"

    labeled = df["wave_label"].astype(str).str.strip()
    label_mask = labeled.notna() & (labeled != "") & (labeled != "nan")
    label_bars = df.loc[label_mask, "close"].copy()
    label_bars.name = "label_close"

    return GroundTruth(
        ticker=ticker, timeframe=tf,
        prices=df["close"].astype(float),
        label_bars=label_bars,
        raw=df.reset_index(),
    )


def load_all_ground_truth(directory: str | Path) -> list[GroundTruth]:
    """Load every CSV in `directory` as a GroundTruth."""
    d = Path(directory)
    if not d.exists():
        return []
    out: list[GroundTruth] = []
    for p in sorted(d.glob("*.csv")):
        try:
            out.append(load_ground_truth(p))
        except Exception as exc:
            # Skip malformed files but don't crash the whole calibration.
            print(f"  ! skipping {p.name}: {exc}")
    return out


# ---------------------------------------------------------------------------
# Matching detector swings to TV labels
# ---------------------------------------------------------------------------
def _match_swings_to_labels(
    swings: list[Swing],
    label_dates: pd.DatetimeIndex,
    price_index: pd.DatetimeIndex,
    tolerance_bars: int,
) -> tuple[int, int, int]:
    """Compute (matched, n_tv, n_ours).

    matched = TV labels that have at least one detector swing within
              `tolerance_bars` of them.
    """
    if len(swings) == 0 or len(label_dates) == 0:
        return 0, len(label_dates), len(swings)

    # Position of each label date in the price index.
    pos_label = pd.Series(np.arange(len(price_index)), index=price_index)
    label_pos = pos_label.reindex(label_dates).dropna().astype(int).to_numpy()

    # Position of each detector swing in the same index.
    swing_pos = np.array([s.bar_index for s in swings], dtype=int)

    if len(label_pos) == 0 or len(swing_pos) == 0:
        return 0, len(label_dates), len(swings)

    # For each label, check if any swing is within tolerance.
    matched = 0
    for lp in label_pos:
        if np.min(np.abs(swing_pos - lp)) <= tolerance_bars:
            matched += 1
    return matched, len(label_pos), len(swing_pos)


def _score_for_threshold(
    gt: GroundTruth, threshold_pct: float, tolerance_bars: int,
) -> dict:
    """Run the detector at one threshold and grade it against TV labels."""
    swings = detect_zigzag(gt.prices, threshold_pct=threshold_pct)
    matched, n_tv, n_ours = _match_swings_to_labels(
        swings, gt.label_bars.index, gt.prices.index, tolerance_bars,
    )
    recall = matched / n_tv if n_tv > 0 else 0.0
    precision = matched / n_ours if n_ours > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {
        "threshold_pct": threshold_pct,
        "n_tv_labels": n_tv,
        "n_our_swings": n_ours,
        "matched": matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Calibration sweep
# ---------------------------------------------------------------------------
@dataclass
class CalibrationResult:
    """Aggregate result for one (ticker, timeframe)."""
    ticker: str
    timeframe: str
    best_threshold_pct: float
    best_f1: float
    grid_results: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "timeframe": self.timeframe,
            "best_threshold_pct": self.best_threshold_pct,
            "best_f1": self.best_f1,
            "grid_results": self.grid_results,
        }


def calibrate_one(
    gt: GroundTruth,
    *,
    grid: Iterable[float] = DEFAULT_THRESHOLD_GRID,
    tolerance_bars: int = DEFAULT_MATCH_TOLERANCE,
) -> CalibrationResult:
    """Sweep thresholds for one ground-truth file."""
    results: list[dict] = []
    for thr in grid:
        results.append(_score_for_threshold(gt, thr, tolerance_bars))
    # Best by F1; tie-break by smaller threshold (denser swings = more options).
    best = max(results, key=lambda r: (r["f1"], -r["threshold_pct"]))
    return CalibrationResult(
        ticker=gt.ticker, timeframe=gt.timeframe,
        best_threshold_pct=float(best["threshold_pct"]),
        best_f1=float(best["f1"]),
        grid_results=results,
    )


def calibrate_all(
    ground_truths: list[GroundTruth],
    *,
    grid: Iterable[float] = DEFAULT_THRESHOLD_GRID,
    tolerance_bars: int = DEFAULT_MATCH_TOLERANCE,
    min_labels: int = 5,
) -> dict[str, CalibrationResult]:
    """Run calibration for every ground-truth file and return per-TF aggregates.

    Returns a dict keyed by timeframe: '1d', '4h', '15m', etc. For each TF we
    average the best-threshold across tickers (weighted by F1) so the saved
    setting generalizes, not over-fits one ticker.

    Files with fewer than `min_labels` labels are silently EXCLUDED from
    the aggregate — their F1 would be near-zero from noise alone and they'd
    pull the weighted average towards meaningless values.
    """
    # Only files with enough labels participate. Bars-only files are skipped.
    calibratable = [g for g in ground_truths if g.n_labels >= min_labels]
    per_file: list[CalibrationResult] = [
        calibrate_one(g, grid=grid, tolerance_bars=tolerance_bars)
        for g in calibratable
    ]

    by_tf: dict[str, list[CalibrationResult]] = {}
    for r in per_file:
        by_tf.setdefault(r.timeframe, []).append(r)

    aggregated: dict[str, CalibrationResult] = {}
    for tf, group in by_tf.items():
        # Drop any result with F1=0 — those failed to converge meaningfully.
        good = [r for r in group if r.best_f1 > 0]
        if not good:
            good = group
        # F1-weighted average of best thresholds.
        weights = np.array([max(r.best_f1, 1e-3) for r in good])
        thresholds = np.array([r.best_threshold_pct for r in good])
        avg_threshold = float(np.average(thresholds, weights=weights))
        avg_f1 = float(np.mean([r.best_f1 for r in good]))
        aggregated[tf] = CalibrationResult(
            ticker=f"<aggregate of {len(good)}>",
            timeframe=tf,
            best_threshold_pct=avg_threshold,
            best_f1=avg_f1,
            grid_results=[],
        )
    return aggregated


def per_ticker_thresholds(
    ground_truths: list[GroundTruth],
    *,
    grid: Iterable[float] = DEFAULT_THRESHOLD_GRID,
    tolerance_bars: int = DEFAULT_MATCH_TOLERANCE,
    min_labels: int = 5,
) -> dict[str, dict[str, float]]:
    """Per-ticker, per-timeframe best thresholds.

    Returns:
        {ticker: {timeframe: best_threshold_pct}}
    Useful for multi_tf.py to pick a ticker-specific threshold when one exists,
    falling back to the aggregate otherwise.
    """
    out: dict[str, dict[str, float]] = {}
    for g in ground_truths:
        if g.n_labels < min_labels:
            continue
        r = calibrate_one(g, grid=grid, tolerance_bars=tolerance_bars)
        out.setdefault(g.ticker, {})[g.timeframe] = r.best_threshold_pct
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_calibration(
    aggregated: dict[str, CalibrationResult],
    per_file: list[CalibrationResult] | None = None,
    per_ticker: dict[str, dict[str, float]] | None = None,
    vol_formula: dict[str, dict[str, float]] | None = None,
    path: str | Path = CALIBRATION_FILE,
) -> None:
    """Write calibration JSON next to multi_tf.py.

    Parameters
    ----------
    aggregated : per-TF F1-weighted best thresholds.
    per_ticker : {ticker: {tf: threshold}} for ticker-level overrides.
    vol_formula : {tf: {slope, intercept, r_squared, min_threshold, max_threshold}}
        Linear-fit coefficients of `threshold_pct ≈ slope × daily_vol_pct + intercept`.
        This is the Stage-5 fallback when neither per-ticker nor aggregate
        match — gives any unseen ticker a sensible threshold based on its
        realized volatility alone.
    """
    payload = {
        "aggregated_by_timeframe": {tf: r.as_dict() for tf, r in aggregated.items()},
        "per_file": [r.as_dict() for r in (per_file or [])],
        "per_ticker": per_ticker or {},
        "vol_formula": vol_formula or {},
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def load_calibration(path: str | Path = CALIBRATION_FILE) -> dict[str, float] | None:
    """Load calibrated aggregate thresholds. Returns {tf: threshold_pct} or None."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except Exception:
        return None
    agg = data.get("aggregated_by_timeframe", {})
    return {tf: float(v["best_threshold_pct"]) for tf, v in agg.items()}


def load_per_ticker(path: str | Path = CALIBRATION_FILE) -> dict[str, dict[str, float]]:
    """Load per-ticker thresholds. Returns {ticker: {tf: threshold}} or empty dict."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except Exception:
        return {}
    raw = data.get("per_ticker", {})
    return {
        ticker: {tf: float(thr) for tf, thr in tfs.items()}
        for ticker, tfs in raw.items()
    }


def threshold_for(
    ticker: str, timeframe: str,
    *, path: str | Path = CALIBRATION_FILE,
) -> float | None:
    """Look up the best threshold for (ticker, tf). Falls back to aggregate.

    Returns None if neither per-ticker nor aggregate is available.
    Does NOT apply the vol-based formula — that's a separate step done in
    multi_tf.py where the realized-vol of the loaded series is available.
    """
    per = load_per_ticker(path)
    if ticker in per and timeframe in per[ticker]:
        return per[ticker][timeframe]
    agg = load_calibration(path)
    if agg and timeframe in agg:
        return agg[timeframe]
    return None


def load_vol_formula(path: str | Path = CALIBRATION_FILE) -> dict[str, dict[str, float]]:
    """Load {tf: {slope, intercept, ...}} formula from elliott_calibration.json.

    Returns empty dict if no formula exists yet.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except Exception:
        return {}
    raw = data.get("vol_formula", {})
    out: dict[str, dict[str, float]] = {}
    for tf, params in raw.items():
        try:
            out[tf] = {k: float(v) for k, v in params.items()}
        except (TypeError, ValueError):
            continue
    return out


def vol_based_threshold(
    daily_vol_pct: float, timeframe: str,
    *, path: str | Path = CALIBRATION_FILE,
) -> float | None:
    """Apply the saved linear formula to a realized-volatility number.

    Returns threshold as a fraction (e.g. 0.06 for 6%) or None if no formula
    is available for `timeframe`. Clamped to [min_threshold, max_threshold].
    """
    formulas = load_vol_formula(path)
    if timeframe not in formulas:
        return None
    f = formulas[timeframe]
    slope = float(f.get("slope", 0.0))
    intercept = float(f.get("intercept", 0.0))
    raw_pct = slope * daily_vol_pct + intercept
    raw_frac = raw_pct / 100.0
    lo = float(f.get("min_threshold", 0.005))
    hi = float(f.get("max_threshold", 0.25))
    return max(lo, min(hi, raw_frac))
