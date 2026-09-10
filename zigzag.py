"""ZigZag indicator — filter price noise into a sequence of swing points.

A "swing point" is a local extreme that the price has retraced away from by
at least `threshold_pct`. ZigZag drops anything smaller than that, leaving
only the meaningful turning points. These are the candidate Elliott wave
endpoints.

Algorithm
---------
1. Track running max and running min from the starting bar.
2. When the price retraces from the running max by ≥ threshold_pct, the
   starting bar is confirmed as a LOW and the running max becomes a HIGH.
   Symmetric for the down direction.
3. After the initial direction is set, run the standard ZigZag loop:
   update the running extreme; when price retraces by threshold, emit the
   extreme as a confirmed swing and flip direction.
4. The trailing extreme is appended as `provisional=True` since price
   could still extend it further.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


SwingKind = Literal["high", "low"]


@dataclass
class Swing:
    index: pd.Timestamp
    price: float
    kind: SwingKind
    bar_index: int
    provisional: bool = False


def detect_zigzag(prices: pd.Series, threshold_pct: float = 0.05) -> list[Swing]:
    """Find ZigZag swings with the given retracement threshold.

    Parameters
    ----------
    prices : pd.Series
        Close prices indexed by datetime.
    threshold_pct : float
        Minimum retracement (e.g. 0.05 = 5%) before a swing is confirmed.

    Returns
    -------
    list[Swing]
        Alternating highs and lows in chronological order. The last entry
        may have `provisional=True` (live, not-yet-confirmed extreme).
    """
    if prices.empty:
        return []
    if threshold_pct <= 0:
        raise ValueError("threshold_pct must be positive.")

    p = prices.dropna().values.astype(float)
    idx = prices.dropna().index
    n = len(p)
    if n < 2:
        return []

    swings: list[Swing] = []

    # Phase 1: direction unknown — track running max and running min
    # separately from the starting bar.
    running_max = p[0]
    running_max_pos = 0
    running_min = p[0]
    running_min_pos = 0

    direction: SwingKind | None = None
    current_extreme: float | None = None
    current_extreme_pos: int | None = None

    i = 1
    while i < n and direction is None:
        price = p[i]
        if price > running_max:
            running_max = price
            running_max_pos = i
        if price < running_min:
            running_min = price
            running_min_pos = i

        down_from_max = (running_max - price) / running_max if running_max > 0 else 0.0
        up_from_min = (price - running_min) / running_min if running_min > 0 else 0.0

        # The starting bar gets emitted as a swing ONLY if it was an extreme.
        if down_from_max >= threshold_pct and running_max_pos > 0:
            # Bar 0 is a low; running_max is the confirmed first high.
            if running_min_pos < running_max_pos:
                swings.append(Swing(
                    index=idx[running_min_pos], price=running_min,
                    kind="low", bar_index=running_min_pos,
                ))
            swings.append(Swing(
                index=idx[running_max_pos], price=running_max,
                kind="high", bar_index=running_max_pos,
            ))
            direction = "low"  # next swing we're looking for is a low
            current_extreme = price
            current_extreme_pos = i
            i += 1
            break
        if up_from_min >= threshold_pct and running_min_pos > 0:
            if running_max_pos < running_min_pos:
                swings.append(Swing(
                    index=idx[running_max_pos], price=running_max,
                    kind="high", bar_index=running_max_pos,
                ))
            swings.append(Swing(
                index=idx[running_min_pos], price=running_min,
                kind="low", bar_index=running_min_pos,
            ))
            direction = "high"
            current_extreme = price
            current_extreme_pos = i
            i += 1
            break
        i += 1

    # Phase 2: standard ZigZag with known direction.
    while i < n and direction is not None:
        price = p[i]
        if direction == "low":
            # Looking for the next low — track running minimum.
            if current_extreme is None or price < current_extreme:
                current_extreme = price
                current_extreme_pos = i
            else:
                rise = (price - current_extreme) / current_extreme if current_extreme > 0 else 0.0
                if rise >= threshold_pct:
                    swings.append(Swing(
                        index=idx[current_extreme_pos], price=current_extreme,
                        kind="low", bar_index=current_extreme_pos,
                    ))
                    direction = "high"
                    current_extreme = price
                    current_extreme_pos = i
        else:  # direction == "high"
            if current_extreme is None or price > current_extreme:
                current_extreme = price
                current_extreme_pos = i
            else:
                drawdown = (current_extreme - price) / current_extreme if current_extreme > 0 else 0.0
                if drawdown >= threshold_pct:
                    swings.append(Swing(
                        index=idx[current_extreme_pos], price=current_extreme,
                        kind="high", bar_index=current_extreme_pos,
                    ))
                    direction = "low"
                    current_extreme = price
                    current_extreme_pos = i
        i += 1

    # Append the trailing extreme as provisional.
    if direction is not None and current_extreme is not None and current_extreme_pos is not None:
        provisional_kind: SwingKind = direction
        # current_extreme is the extreme in the direction we're now tracking
        # — i.e. if direction == "low", current_extreme is the running minimum
        # → the trailing swing is a LOW.
        swings.append(Swing(
            index=idx[current_extreme_pos], price=current_extreme,
            kind=provisional_kind, bar_index=current_extreme_pos,
            provisional=True,
        ))

    return swings


def swings_to_dataframe(swings: list[Swing]) -> pd.DataFrame:
    """Convert a swing list to a DataFrame for plotting / inspection."""
    if not swings:
        return pd.DataFrame(columns=["price", "kind", "bar_index", "provisional"])
    return pd.DataFrame(
        [
            {
                "index": s.index, "price": s.price, "kind": s.kind,
                "bar_index": s.bar_index, "provisional": s.provisional,
            }
            for s in swings
        ]
    ).set_index("index")
