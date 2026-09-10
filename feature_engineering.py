"""Feature computation pipeline for the HMM.

All features use only past data at each timestep (e.g. rolling windows are
left-aligned/trailing), so the resulting feature matrix is causally safe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# All windows are in *trading bars* (not calendar days).
REALIZED_VOL_WINDOW: int = 20
VOLUME_MA_WINDOW: int = 20

# Columns produced by compute_features.
LOG_RETURN_COL: str = "log_return"
REALIZED_VOL_COL: str = "realized_vol"
VOLUME_RATIO_COL: str = "volume_ratio"
HL_RANGE_PCT_COL: str = "hl_range_pct"

# Columns fed into the HMM as the feature matrix X.
# log_return is intentionally excluded: it is too noisy / heavy-tailed to
# drive state inference directly. We keep it in the frame for downstream
# regime statistics (e.g. annualized mean return per regime).
HMM_FEATURE_COLS: tuple[str, ...] = (REALIZED_VOL_COL, VOLUME_RATIO_COL, HL_RANGE_PCT_COL)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features and return a NaN-free DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV frame with columns Open, High, Low, Close, Volume.

    Returns
    -------
    pd.DataFrame
        Original columns plus log_return, realized_vol, volume_ratio,
        hl_range_pct. Rows with any NaN (from rolling-window warmups) are
        dropped.
    """
    required = ("Close", "High", "Low", "Volume")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input frame is missing required columns: {missing}")

    out = df.copy()

    # Log returns — single-bar arithmetic, no look-ahead.
    out[LOG_RETURN_COL] = np.log(out["Close"] / out["Close"].shift(1))

    # Trailing realized volatility (std of recent log returns).
    out[REALIZED_VOL_COL] = out[LOG_RETURN_COL].rolling(REALIZED_VOL_WINDOW).std()

    # Volume relative to its own trailing average (regime feature, not absolute).
    vol_ma = out["Volume"].rolling(VOLUME_MA_WINDOW).mean()
    out[VOLUME_RATIO_COL] = out["Volume"] / vol_ma

    # Intra-bar range as fraction of close price.
    out[HL_RANGE_PCT_COL] = (out["High"] - out["Low"]) / out["Close"]

    # Drop NaN rows — no imputation, by spec.
    out.dropna(inplace=True)
    return out


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract the HMM feature matrix X from a computed-features frame."""
    missing = [c for c in HMM_FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"compute_features() must be called first; missing: {missing}")
    return df[list(HMM_FEATURE_COLS)].to_numpy(dtype=np.float64)
