"""Regime sorting, naming, and the stability post-filter.

The HMM emits unordered integer states. We sort them by *realized
volatility* (the first feature in the HMM matrix) so labels are intuitive:
state 0 is always the calmest regime, state N-1 the wildest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


# Stability filter parameters — pulled out for transparency.
CONFIRMATION_BARS: int = 3      # bars a regime must persist before becoming "active"
FLIP_WINDOW: int = 20           # trailing window for chop detection
MAX_FLIPS_BEFORE_UNCERTAIN: int = 4  # > this many flips → "Uncertain"

# Reserved label for the chop / low-confidence override.
UNCERTAIN_LABEL: str = "Uncertain"


def label_regimes(
    model: GaussianHMM,
    n_components: int,
    scaler_mean: np.ndarray | None = None,
    scaler_scale: np.ndarray | None = None,
) -> dict[int, str]:
    """Map raw HMM state integers to human-readable volatility labels.

    States are sorted by their mean of feature index 0 (= realized_vol in
    feature_engineering.HMM_FEATURE_COLS). If a StandardScaler was used,
    pass its mean_ and scale_ so we can invert the standardization for a
    sanity-checkable ordering — though for *ordering* alone the raw means
    suffice (monotone transform).

    Parameters
    ----------
    model : fitted GaussianHMM
    n_components : int
        Same as model.n_components, accepted for explicitness.
    scaler_mean, scaler_scale : optional
        StandardScaler attributes (not used for ordering, only available
        if future callers want true-scale means).

    Returns
    -------
    dict[int, str]
        {raw_state_id: label}
    """
    # model.means_ is (n_components, n_features) in standardized space.
    vol_means = model.means_[:, 0]  # index 0 == realized_vol
    sorted_states = np.argsort(vol_means)  # ascending: calmest first

    labels: list[str]
    if n_components == 3:
        labels = ["Low Vol", "Medium Vol", "High Vol"]
    elif n_components == 4:
        labels = ["Low Vol", "Medium-Low Vol", "Medium-High Vol", "High Vol"]
    else:
        labels = ["Low Vol"]
        for i in range(1, n_components - 1):
            labels.append(f"Vol Regime {i + 1}")
        labels.append("High Vol")

    return {int(state): labels[rank] for rank, state in enumerate(sorted_states)}


def apply_stability_filter(raw_labels: pd.Series) -> pd.Series:
    """Smooth and chop-filter the raw regime label series.

    Rules (in order):

    1. **Persistence confirmation.** A new label is only adopted after it has
       held for `CONFIRMATION_BARS` consecutive bars. Before confirmation we
       carry forward the last confirmed label. (For the very first bars,
       before any confirmation has happened, we adopt the first run-of-3
       label retroactively — pre-warmup bars share that label.)

    2. **Chop override.** Within any trailing `FLIP_WINDOW` of confirmed
       labels, if the number of label *changes* exceeds
       `MAX_FLIPS_BEFORE_UNCERTAIN`, the current bar's label is overridden
       to `Uncertain`. This catches whipsawing regions where the underlying
       state is genuinely ambiguous.

    Returns
    -------
    pd.Series with the same index as the input.
    """
    if raw_labels.empty:
        return raw_labels.copy()

    raw = raw_labels.tolist()
    T = len(raw)
    confirmed: list[str] = [None] * T  # type: ignore[list-item]

    # --- Pass 1: persistence-confirmed labels ---
    current = raw[0]
    run_len = 1
    last_confirmed: str | None = None

    for t in range(T):
        if t == 0:
            run_len = 1
        else:
            if raw[t] == raw[t - 1]:
                run_len += 1
            else:
                run_len = 1
        # Becomes "active" only after CONFIRMATION_BARS in a row.
        if run_len >= CONFIRMATION_BARS:
            last_confirmed = raw[t]
        confirmed[t] = last_confirmed  # may be None during the very first warmup

    # If the series never had a 3-bar run at the start, backfill with the
    # first non-None confirmation (better than leaving NaN/None).
    first_real = next((c for c in confirmed if c is not None), raw[0])
    for t in range(T):
        if confirmed[t] is None:
            confirmed[t] = first_real

    # --- Pass 2: chop / flip-count override ---
    final: list[str] = list(confirmed)
    for t in range(T):
        lo = max(0, t - FLIP_WINDOW + 1)
        flips = 0
        for k in range(lo + 1, t + 1):
            if confirmed[k] != confirmed[k - 1]:
                flips += 1
        if flips > MAX_FLIPS_BEFORE_UNCERTAIN:
            final[t] = UNCERTAIN_LABEL

    return pd.Series(final, index=raw_labels.index, name=raw_labels.name)
