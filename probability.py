"""Compute empirical probabilities of forecast outcomes from history.

The rule-based forecaster says: "after a completed impulse, expect ABC
correction reaching 38.2% retrace (high prob), 50% (medium), C=A (high),
1.618×A (low)". Those qualitative labels are our priors. With enough
ground-truth data we can replace them with **measured** frequencies:

  P(price retraces 38.2% within 25 bars after wave 5 ends) ≈ ?
  P(price retraces 50%) ≈ ?
  P(price retraces 61.8%) ≈ ?

We compute these by walking each ground-truth file:
  1. Find every completed 5-wave impulse the detector identifies in history
  2. For each, measure what the price did over the next `lookahead_bars` bars
  3. Record which Fibonacci retracement levels were touched
  4. Aggregate into probabilities

The probabilities are saved under `empirical_probabilities` in
`elliott_calibration.json` and read by `forecast.py` to refine target
labels (`high` / `medium` / `low` → real percentages).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from calibrate_from_tv import CALIBRATION_FILE, GroundTruth, load_all_ground_truth
from elliott import _score_impulse_window
from zigzag import detect_zigzag


# Fibonacci levels we measure as targets after a completed impulse.
FIB_LEVELS: tuple[float, ...] = (0.236, 0.382, 0.500, 0.618, 0.786, 1.000)


@dataclass
class FibOutcome:
    """One impulse case + which Fib levels its subsequent move reached."""
    direction: Literal["up", "down"]
    wave5_price: float
    wave0_price: float
    impulse_size: float
    bars_after: int
    levels_reached: dict[float, bool]
    max_retrace_pct: float          # actual maximum counter-move as fraction of impulse


def _evaluate_impulse_outcome(
    prices: pd.Series, w0_pos: int, w5_pos: int, direction: str,
    lookahead_bars: int,
) -> FibOutcome | None:
    """Look forward N bars from wave-5 endpoint, record which Fib retraces hit."""
    if w5_pos + 1 >= len(prices):
        return None
    end_pos = min(w5_pos + lookahead_bars, len(prices) - 1)
    if end_pos <= w5_pos:
        return None

    p0 = float(prices.iloc[w0_pos])
    p5 = float(prices.iloc[w5_pos])
    impulse_size = abs(p5 - p0)
    if impulse_size <= 0:
        return None

    forward = prices.iloc[w5_pos + 1: end_pos + 1].values

    if direction == "up":
        # After up impulse, expect price to FALL toward Fib retracements.
        max_counter_move = float(p5 - forward.min())
    else:
        # After down impulse, expect price to RISE.
        max_counter_move = float(forward.max() - p5)
    max_retrace_pct = max(0.0, max_counter_move) / impulse_size

    levels: dict[float, bool] = {
        fib: max_retrace_pct >= fib for fib in FIB_LEVELS
    }
    return FibOutcome(
        direction=direction, wave5_price=p5, wave0_price=p0,
        impulse_size=impulse_size, bars_after=end_pos - w5_pos,
        levels_reached=levels, max_retrace_pct=max_retrace_pct,
    )


def scan_outcomes_for_one(
    gt: GroundTruth,
    threshold_pct: float,
    *,
    lookahead_bars: int = 30,
    min_impulse_score: float = 60.0,
) -> list[FibOutcome]:
    """Find every completed impulse in this ground-truth series and grade what
    happened after each."""
    swings = detect_zigzag(gt.prices, threshold_pct=threshold_pct)
    if len(swings) < 6:
        return []

    outcomes: list[FibOutcome] = []
    # Scan every 6-swing window in the past (skip the provisional last swing).
    for i in range(6, len(swings)):
        window = swings[i - 6: i]
        if any(s.provisional for s in window):
            continue
        for direction in ("up", "down"):
            cand = _score_impulse_window(window, direction)  # type: ignore[arg-type]
            if cand.score < min_impulse_score or cand.rule_violations:
                continue
            w0_pos = window[0].bar_index
            w5_pos = window[-1].bar_index
            out = _evaluate_impulse_outcome(
                gt.prices, w0_pos, w5_pos, direction, lookahead_bars,
            )
            if out is not None:
                outcomes.append(out)
                break  # impulse already scored — don't double-count opposite direction
    return outcomes


def compute_empirical_probabilities(
    ground_truths: list[GroundTruth],
    *,
    threshold_pct: float = 0.05,
    lookahead_bars: int = 30,
    min_impulse_score: float = 60.0,
) -> dict:
    """Aggregate Fib-level reach probabilities across all ground-truth files.

    Returns
    -------
    {
        "n_impulses": 47,
        "lookahead_bars": 30,
        "p_reaches": {
            "0.382": 0.72,        # 72% of impulses had a >=38.2% retrace
            "0.500": 0.55,
            "0.618": 0.41,
            "1.000": 0.18,
            ...
        },
        "avg_max_retrace": 0.49,
    }
    """
    all_outcomes: list[FibOutcome] = []
    for gt in ground_truths:
        # We can compute outcomes even on bars-only files — TV labels not needed,
        # we run our detector on raw prices.
        outs = scan_outcomes_for_one(
            gt, threshold_pct=threshold_pct,
            lookahead_bars=lookahead_bars,
            min_impulse_score=min_impulse_score,
        )
        all_outcomes.extend(outs)

    if not all_outcomes:
        return {
            "n_impulses": 0,
            "lookahead_bars": lookahead_bars,
            "p_reaches": {},
            "avg_max_retrace": 0.0,
        }

    p_reaches: dict[str, float] = {}
    for fib in FIB_LEVELS:
        hits = sum(1 for o in all_outcomes if o.levels_reached.get(fib, False))
        p_reaches[f"{fib:.3f}"] = round(hits / len(all_outcomes), 4)

    avg_retrace = float(np.mean([o.max_retrace_pct for o in all_outcomes]))

    return {
        "n_impulses": len(all_outcomes),
        "lookahead_bars": lookahead_bars,
        "threshold_pct": threshold_pct,
        "p_reaches": p_reaches,
        "avg_max_retrace": round(avg_retrace, 4),
    }


# ---------------------------------------------------------------------------
# Persistence (extends elliott_calibration.json)
# ---------------------------------------------------------------------------
def save_probabilities(
    probabilities: dict, path: str | Path = CALIBRATION_FILE
) -> None:
    """Merge probabilities into the existing calibration JSON without
    clobbering the other fields."""
    p = Path(path)
    data: dict = {}
    if p.exists():
        try:
            data = json.loads(p.read_text())
        except Exception:
            data = {}
    data["empirical_probabilities"] = probabilities
    p.write_text(json.dumps(data, indent=2))


def load_probabilities(
    path: str | Path = CALIBRATION_FILE,
) -> dict:
    """Return the empirical_probabilities block or {} if not present."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except Exception:
        return {}
    return data.get("empirical_probabilities", {})


# ---------------------------------------------------------------------------
# Convenience: probability of reaching a specific Fib level
# ---------------------------------------------------------------------------
def probability_of_reaching(
    fib_ratio: float, *, path: str | Path = CALIBRATION_FILE,
) -> float | None:
    """Return P(price reaches this Fib retracement after a completed impulse).

    `fib_ratio` should be one of the FIB_LEVELS. Returns None if no
    empirical data is loaded.
    """
    data = load_probabilities(path)
    reaches = data.get("p_reaches", {})
    if not reaches:
        return None
    # Find the nearest measured level
    measured = sorted([float(k) for k in reaches.keys()])
    if not measured:
        return None
    nearest = min(measured, key=lambda x: abs(x - fib_ratio))
    return float(reaches.get(f"{nearest:.3f}", 0.0))
