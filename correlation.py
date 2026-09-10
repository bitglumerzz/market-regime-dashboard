"""Rolling-correlation break detector for pairs of assets.

The thesis (well-documented in the literature on contagion, regime change,
and factor rotation): when two normally-co-moving assets *stop* moving
together, something structural has shifted in the macro or sector regime.

Detection strategy
------------------
For a pair (A, B):
  1. Compute log returns for both legs.
  2. Compute a SHORT-window rolling correlation (default 20 bars).
  3. Compute a LONG-window rolling correlation (default 60 bars).
  4. Compare current short correlation to a rolling z-score baseline —
     "how unusual is today's short correlation relative to recent history?"
  5. Mark a "break" when the z-score crosses below a negative threshold
     (i.e. correlation collapsed beyond normal noise).

Output preserves the entire time series so the UI can render
chart overlays for break regions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from data_loader import load_data


SHORT_WINDOW_DEFAULT: int = 20
LONG_WINDOW_DEFAULT: int = 60
BASELINE_WINDOW_DEFAULT: int = 126        # kept for backward compat; unused now
# Detection: short correlation has dropped this many ABSOLUTE points below long.
# A 0.30 gap on a normally-coupled pair (long ≈ 0.85) means short fell to ≈ 0.55
# — a clear regime break, but not so strict as to miss real events.
BREAK_THRESHOLD_DEFAULT: float = -0.30
# kept as alias for older callers
BREAK_ZSCORE_DEFAULT: float = BREAK_THRESHOLD_DEFAULT
MIN_OBSERVATIONS: int = 80                 # need ≥ this many bars for a useful run

# A sensible default pair list — picks broad-stroke macro regimes.
DEFAULT_PAIRS: tuple[tuple[str, str], ...] = (
    ("SPY", "QQQ"),       # large-cap vs tech — normally highly correlated
    ("GLD", "TLT"),       # gold vs long bonds — risk-off twins
    ("SPY", "GLD"),       # stocks vs gold — usually weak/negative
    ("QQQ", "IWM"),       # tech vs small-cap — risk-on cohort
    ("XLE", "XLF"),       # energy vs financials — cyclicals
)


@dataclass
class PairResult:
    """All time-series produced for one pair."""
    ticker_a: str
    ticker_b: str
    prices_a: pd.Series
    prices_b: pd.Series
    short_corr: pd.Series        # rolling correlation, short window
    long_corr: pd.Series         # rolling correlation, long window
    z_score: pd.Series           # rolling z-score of short_corr's deviation
    in_break: pd.Series          # boolean: True where z below threshold
    current_short_corr: float
    current_long_corr: float
    current_z: float
    current_break: bool
    n_break_events: int          # number of distinct break episodes
    last_break_start: pd.Timestamp | None
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def label(self) -> str:
        return f"{self.ticker_a} / {self.ticker_b}"


def _rolling_correlation(
    log_ret_a: pd.Series, log_ret_b: pd.Series, window: int
) -> pd.Series:
    """Pearson rolling correlation, aligned on the inner index."""
    df = pd.concat([log_ret_a, log_ret_b], axis=1, join="inner").dropna()
    df.columns = ["a", "b"]
    return df["a"].rolling(window).corr(df["b"])


def _count_break_episodes(in_break: pd.Series) -> tuple[int, pd.Timestamp | None]:
    """Count distinct break runs and find when the last one started."""
    n = 0
    last_start: pd.Timestamp | None = None
    prev = False
    for ts, val in in_break.items():
        v = bool(val)
        if v and not prev:
            n += 1
            last_start = ts
        prev = v
    return n, last_start


def analyze_pair(
    ticker_a: str,
    ticker_b: str,
    start: str,
    end: str,
    *,
    short_window: int = SHORT_WINDOW_DEFAULT,
    long_window: int = LONG_WINDOW_DEFAULT,
    baseline_window: int = BASELINE_WINDOW_DEFAULT,
    break_zscore: float = BREAK_THRESHOLD_DEFAULT,  # threshold on (short - long)
) -> PairResult:
    """Run the full correlation-break analysis for one pair."""
    try:
        df_a = load_data(ticker_a, start, end)
        df_b = load_data(ticker_b, start, end)
    except Exception as exc:
        return PairResult(
            ticker_a=ticker_a, ticker_b=ticker_b,
            prices_a=pd.Series(dtype=float), prices_b=pd.Series(dtype=float),
            short_corr=pd.Series(dtype=float), long_corr=pd.Series(dtype=float),
            z_score=pd.Series(dtype=float), in_break=pd.Series(dtype=bool),
            current_short_corr=np.nan, current_long_corr=np.nan,
            current_z=np.nan, current_break=False,
            n_break_events=0, last_break_start=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    prices_a = df_a["Close"]
    prices_b = df_b["Close"]
    log_ret_a = np.log(prices_a / prices_a.shift(1))
    log_ret_b = np.log(prices_b / prices_b.shift(1))

    short = _rolling_correlation(log_ret_a, log_ret_b, short_window)
    long = _rolling_correlation(log_ret_a, log_ret_b, long_window)

    if len(short.dropna()) < MIN_OBSERVATIONS:
        return PairResult(
            ticker_a=ticker_a, ticker_b=ticker_b,
            prices_a=prices_a, prices_b=prices_b,
            short_corr=short, long_corr=long,
            z_score=pd.Series(dtype=float), in_break=pd.Series(dtype=bool),
            current_short_corr=np.nan, current_long_corr=np.nan,
            current_z=np.nan, current_break=False,
            n_break_events=0, last_break_start=None,
            error=(
                f"Only {len(short.dropna())} valid bars after correlation warm-up; "
                f"need ≥ {MIN_OBSERVATIONS}. Widen the date range."
            ),
        )

    # Detection signal: absolute spread between short- and long-window
    # correlations. A negative spread means the short-term coupling has
    # dropped below the longer-term baseline — that's the regime break.
    # We expose the spread under the `z_score` field for backwards-compat
    # with the UI; it's no longer a true z-score.
    diff = short - long
    z_score = diff
    in_break = (diff < break_zscore).fillna(False)

    # Current snapshot.
    current_short = float(short.dropna().iloc[-1]) if len(short.dropna()) else np.nan
    current_long = float(long.dropna().iloc[-1]) if len(long.dropna()) else np.nan
    # `current_z` is now the current absolute spread (short - long), kept
    # under the old name for API stability.
    current_z = (current_short - current_long
                 if np.isfinite(current_short) and np.isfinite(current_long)
                 else np.nan)
    current_break = bool(in_break.iloc[-1]) if len(in_break) else False

    n_events, last_start = _count_break_episodes(in_break)

    return PairResult(
        ticker_a=ticker_a, ticker_b=ticker_b,
        prices_a=prices_a, prices_b=prices_b,
        short_corr=short, long_corr=long,
        z_score=z_score, in_break=in_break,
        current_short_corr=current_short, current_long_corr=current_long,
        current_z=current_z, current_break=current_break,
        n_break_events=n_events, last_break_start=last_start,
    )


def parse_pair_list(text: str) -> list[tuple[str, str]]:
    """Parse a user-provided text into a list of pairs.

    Accepts either 'SPY/QQQ, GLD/TLT' or one pair per line, with '/' or ',' delim.
    Empty or malformed pairs are silently skipped.
    """
    pairs: list[tuple[str, str]] = []
    for raw_line in text.replace(";", "\n").splitlines():
        for entry in raw_line.split(","):
            entry = entry.strip()
            if not entry:
                continue
            # Try common separators in order.
            for sep in ("/", "|", ":", " vs ", " - "):
                if sep in entry:
                    parts = [p.strip().upper() for p in entry.split(sep) if p.strip()]
                    if len(parts) == 2:
                        pairs.append((parts[0], parts[1]))
                        break
    return pairs


def break_episodes(in_break: pd.Series, dates: pd.DatetimeIndex) -> list[dict]:
    """Compress a break boolean series into start/end episodes for display."""
    episodes: list[dict] = []
    if in_break.empty:
        return episodes
    in_run = False
    start = None
    last_ts = None
    for ts, val in in_break.items():
        v = bool(val)
        if v and not in_run:
            in_run = True
            start = ts
            last_ts = ts
        elif v and in_run:
            last_ts = ts
        elif not v and in_run:
            episodes.append({
                "start": start, "end": last_ts,
                "duration_bars": int(((dates >= start) & (dates <= last_ts)).sum()),
            })
            in_run = False
            start = None
    if in_run and start is not None:
        episodes.append({
            "start": start, "end": last_ts,
            "duration_bars": int(((dates >= start) & (dates <= last_ts)).sum()),
        })
    return episodes
