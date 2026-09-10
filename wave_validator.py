"""Backtest the Elliott detector against historical data.

The trick: every time the detector identifies a completed 5-wave impulse,
Elliott theory says the next move is an A-B-C correction in the opposite
direction. We can VERIFY this on history.

For each completed impulse found in the past:
  - Note the price at the end of wave 5 (`p5`) and the direction.
  - Look at the price action over the next `lookahead_bars` bars.
  - Did the price move counter-trend by at least `move_threshold_pct`?
    → HIT  (Elliott was right)
  - Did it instead continue in the same direction by `move_threshold_pct`?
    → MISS (Elliott was wrong)
  - Anything in between? → AMBIGUOUS

The aggregate hit-rate is a quantitative measure of how well our detector
is finding real Elliott patterns versus random alternation. If the hit
rate is well above 50%, our detector is finding signal; if it's around 50%,
we're seeing noise.

This is also a feedback loop for tuning: try different `major_threshold_pct`
values and see which one maximizes hit rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from elliott import _score_impulse_window
from zigzag import Swing, detect_zigzag


@dataclass
class ValidationCase:
    """One detected impulse + how the market actually behaved after it."""
    direction: Literal["up", "down"]
    end_timestamp: pd.Timestamp
    end_price: float
    score: float                       # detector's own score for the impulse
    rule_violations: int
    lookahead_bars: int
    final_price: float                 # price `lookahead_bars` later
    counter_move_pct: float            # signed: + means counter-trend (HIT)
    outcome: Literal["hit", "miss", "ambiguous"]


@dataclass
class ValidationReport:
    """Aggregate results across all detected impulses in history."""
    n_cases: int
    n_hits: int
    n_misses: int
    n_ambiguous: int
    hit_rate: float                    # hits / (hits + misses) — excludes ambiguous
    raw_hit_rate: float                # hits / n_cases — includes ambiguous in denominator
    avg_counter_move_pct: float
    cases: list[ValidationCase] = field(default_factory=list)

    @property
    def total_decisive(self) -> int:
        return self.n_hits + self.n_misses


def validate_detector(
    prices: pd.Series,
    *,
    threshold_pct: float = 0.05,
    lookahead_bars: int = 20,
    move_threshold_pct: float = 0.05,
    min_score: float = 50.0,
) -> ValidationReport:
    """Sweep through history, find completed impulses, and grade them.

    Parameters
    ----------
    prices : Close price series.
    threshold_pct : ZigZag threshold (same as in detect_zigzag).
    lookahead_bars : how far forward to look after a detected impulse.
    move_threshold_pct : how big the counter-move must be to count as a HIT.
    min_score : ignore impulses below this score (likely noise).

    Returns
    -------
    ValidationReport with per-case detail and aggregate hit rate.
    """
    if prices.empty:
        return ValidationReport(0, 0, 0, 0, 0.0, 0.0, 0.0)

    # Get the full swing list once.
    all_swings = detect_zigzag(prices, threshold_pct=threshold_pct)

    cases: list[ValidationCase] = []

    # Slide a 6-swing window across history (NOT just the tail).
    # For each window, score it as both up and down impulse, take the
    # better-scoring direction, and grade only completed impulses.
    for i in range(6, len(all_swings)):
        window = all_swings[i - 6: i]
        # Skip any window that contains the provisional swing.
        if any(s.provisional for s in window):
            continue

        cand_up = _score_impulse_window(window, "up")
        cand_dn = _score_impulse_window(window, "down")
        best = cand_up if cand_up.score >= cand_dn.score else cand_dn
        direction = "up" if best is cand_up else "down"

        if best.score < min_score:
            continue
        # We only validate impulses that are RECENT enough to have data after them
        end_swing = window[-1]
        end_bar = end_swing.bar_index
        future_end = end_bar + lookahead_bars
        if future_end >= len(prices):
            continue

        end_price = float(prices.iloc[end_bar])
        future_price = float(prices.iloc[future_end])

        # Counter-move = price movement IN THE OPPOSITE DIRECTION of the impulse.
        if direction == "up":
            # We expect price to FALL after an up impulse (A-B-C correction).
            counter_move_pct = (end_price - future_price) / end_price
        else:
            # We expect price to RISE after a down impulse.
            counter_move_pct = (future_price - end_price) / end_price

        if counter_move_pct >= move_threshold_pct:
            outcome = "hit"
        elif counter_move_pct <= -move_threshold_pct:
            outcome = "miss"
        else:
            outcome = "ambiguous"

        cases.append(ValidationCase(
            direction=direction,
            end_timestamp=end_swing.index,
            end_price=end_price,
            score=best.score,
            rule_violations=len(best.rule_violations),
            lookahead_bars=lookahead_bars,
            final_price=future_price,
            counter_move_pct=counter_move_pct,
            outcome=outcome,
        ))

    n_cases = len(cases)
    n_hits = sum(1 for c in cases if c.outcome == "hit")
    n_misses = sum(1 for c in cases if c.outcome == "miss")
    n_amb = n_cases - n_hits - n_misses
    decisive = n_hits + n_misses
    hit_rate = (n_hits / decisive) if decisive > 0 else 0.0
    raw_hit_rate = (n_hits / n_cases) if n_cases > 0 else 0.0
    avg_counter = float(np.mean([c.counter_move_pct for c in cases])) if cases else 0.0

    return ValidationReport(
        n_cases=n_cases, n_hits=n_hits, n_misses=n_misses, n_ambiguous=n_amb,
        hit_rate=hit_rate, raw_hit_rate=raw_hit_rate,
        avg_counter_move_pct=avg_counter, cases=cases,
    )
