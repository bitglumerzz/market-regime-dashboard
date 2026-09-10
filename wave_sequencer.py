"""Wave Sequencer — segment full swing history into a chain of waves.

Bridges from the current `elliott.classify()` (which finds the best wave
in a single window) to a complete historical labeling: for every period
in the past we know what wave structure was in progress.

Algorithm (greedy, non-overlapping):
    j = 0
    while j < len(swings) - 3:
        try classify() on swings[j : j + W] for each candidate window size
        pick the highest-scoring acceptance (score >= min_accept_score)
        accept → emit WaveSegment, advance j to last swing of accepted wave
        reject → j += 1 (skip noise)

A wave shares its LAST swing with the start of the next wave (a fifth
peak is the same point as the first low of the correction that follows).

Why not just call classify() once at the end of the swings list?
  - classify() returns ONE best hypothesis for the current structure.
  - For visualizing full chart history we need to know what was happening
    in 2024, 2023, etc. — not just "what's the wave we're currently in".
  - Sequencer scans ALL of history and emits a sequence of completed waves
    plus one in-progress wave at the right edge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from elliott import classify, WaveCandidate
from zigzag import Swing


# Candidate window sizes to try at each starting position.
# Order matters: longest first so we prefer fuller structures when scores tie.
# Sizes correspond to Elliott patterns:
#   9 = WXY with extra X (or full WXYXZ if there's room)
#   8 = WXY (3 + linker + 3 + linker = 8 swings ≈ 7-9 endpoints)
#   6 = impulse / diagonal / triangle
#   4 = simple zigzag / flat
_WINDOW_SIZES: tuple[int, ...] = (9, 8, 6, 4)

# Minimum classify().score for a window to count as an accepted wave.
# Scores below this are usually noise (alternating swings without proper
# Fibonacci or rule compliance).
MIN_ACCEPT_SCORE: float = 40.0

# Minor-tier threshold is more lenient — minor waves are noisier by nature.
MIN_ACCEPT_SCORE_MINOR: float = 25.0


@dataclass
class WaveSegment:
    """One labeled wave occupying a contiguous slice of the swing history."""
    start_idx: int                  # index in swings list (inclusive)
    end_idx: int                    # exclusive
    candidate: WaveCandidate        # what classify() said about this window

    @property
    def t_start(self):
        return self.candidate.swings_used[0].index

    @property
    def t_end(self):
        return self.candidate.swings_used[-1].index

    @property
    def price_start(self) -> float:
        return self.candidate.swings_used[0].price

    @property
    def price_end(self) -> float:
        return self.candidate.swings_used[-1].price

    @property
    def pattern(self) -> str:
        return self.candidate.pattern

    @property
    def score(self) -> float:
        return self.candidate.score

    def as_dict(self) -> dict:
        """Lightweight dict — useful for caching to parquet/JSON."""
        return {
            "start_idx":  self.start_idx,
            "end_idx":    self.end_idx,
            "pattern":    self.pattern,
            "score":      self.score,
            "t_start":    self.t_start.isoformat() if self.t_start else None,
            "t_end":      self.t_end.isoformat() if self.t_end else None,
            "price_start": self.price_start,
            "price_end":   self.price_end,
            "labels":     list(self.candidate.labels),
            "swing_indices": [s.bar_index for s in self.candidate.swings_used],
            "swing_prices":  [s.price for s in self.candidate.swings_used],
            "swing_kinds":   [s.kind for s in self.candidate.swings_used],
            "is_extended":  self.candidate.is_extended,
            "is_diagonal":  self.candidate.is_diagonal,
            "diagonal_kind": self.candidate.diagonal_kind,
            "is_truncated": self.candidate.is_truncated,
            "triangle_kind": self.candidate.triangle_kind,
        }


def segment_history(
    swings: list[Swing],
    min_accept_score: float = MIN_ACCEPT_SCORE,
    window_sizes: tuple[int, ...] = _WINDOW_SIZES,
) -> list[WaveSegment]:
    """Greedy non-overlapping wave segmentation over the full swing history.

    Parameters
    ----------
    swings :
        Alternating low/high swings from `zigzag.detect_zigzag`.
    min_accept_score :
        Reject any window whose best candidate scored below this. Below 40
        usually means classify() couldn't find proper Fib retracements or
        Elliott rule compliance, i.e. noise.
    window_sizes :
        Candidate window sizes to try at each position. Default tries
        9/8/6/4 (WXY, WXY, impulse-like, zigzag-like).

    Returns
    -------
    list[WaveSegment]
        Chronologically ordered, non-overlapping. May not cover every swing
        — gaps are zones where no pattern scored above threshold (sparse
        history or chop). The rightmost segment is the "current" wave.
    """
    if not swings:
        return []

    segments: list[WaveSegment] = []
    n = len(swings)
    j = 0
    while j <= n - 4:                          # need ≥4 swings for any pattern
        best: Optional[tuple[float, int, WaveCandidate]] = None
        for w in window_sizes:
            if j + w > n:
                continue
            window = swings[j : j + w]
            cands = classify(window, top_k=1)
            if not cands:
                continue
            c = cands[0]
            if c.score <= 0:
                continue
            if best is None or c.score > best[0]:
                best = (c.score, w, c)

        if best is not None and best[0] >= min_accept_score:
            score, w, cand = best
            # The candidate may have used a sub-window of swings (e.g. classify
            # picked 6 from a 9-slice). Use its ACTUAL swings_used to find
            # where the wave really ends.
            last_swing = cand.swings_used[-1]
            try:
                end_offset = next(i for i, s in enumerate(swings[j:])
                                   if s.index == last_swing.index
                                   and s.price == last_swing.price)
            except StopIteration:
                end_offset = w - 1
            segments.append(WaveSegment(
                start_idx=j,
                end_idx=j + end_offset + 1,
                candidate=cand,
            ))
            # Next wave starts at the same swing the current one ended on
            # (a 5 of one impulse = 0 of the next correction).
            j += end_offset
            if j == segments[-1].start_idx:        # safety: no progress
                j += 1
        else:
            j += 1

    return segments


def segment_with_subwaves(
    major_swings: list[Swing],
    minor_swings: list[Swing],
    min_accept_score: float = MIN_ACCEPT_SCORE,
    min_accept_score_minor: float = MIN_ACCEPT_SCORE_MINOR,
) -> tuple[list[WaveSegment], list[WaveSegment]]:
    """Segment BOTH tiers — major + minor — independently.

    The renderer can then draw major as background zones (one color per
    pattern type) and minor as fine-grained 1-2-3-4-5/a-b-c labels on top.

    Minor uses a looser score threshold because finer-grained swings are
    noisier and rarely satisfy strict Elliott rules even when the human
    eye sees a clear sub-structure.
    """
    major = segment_history(major_swings,
                              min_accept_score=min_accept_score)
    minor = segment_history(minor_swings,
                              min_accept_score=min_accept_score_minor)
    return major, minor


# ---------------------------------------------------------------------------
# Diagnostics — useful for offline tuning
# ---------------------------------------------------------------------------
def coverage_pct(segments: list[WaveSegment], total_swings: int) -> float:
    """How many swings (by count, not bars) are covered by accepted waves."""
    if total_swings == 0:
        return 0.0
    covered = sum(s.end_idx - s.start_idx for s in segments)
    return 100.0 * covered / total_swings


def summarize(segments: list[WaveSegment]) -> dict:
    """Compact stats for one tier — used in logging / a CLI sanity check."""
    if not segments:
        return {"count": 0, "patterns": {}, "score_mean": 0.0}
    patterns: dict[str, int] = {}
    for seg in segments:
        patterns[seg.pattern] = patterns.get(seg.pattern, 0) + 1
    return {
        "count": len(segments),
        "patterns": patterns,
        "score_mean": sum(s.score for s in segments) / len(segments),
        "score_min":  min(s.score for s in segments),
        "score_max":  max(s.score for s in segments),
        "t_first":    segments[0].t_start.isoformat() if segments else None,
        "t_last":     segments[-1].t_end.isoformat() if segments else None,
    }


# ---------------------------------------------------------------------------
# CLI smoke test — `python wave_sequencer.py BTC-USD 1d`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from data_provider import fetch_close
    from zigzag import detect_zigzag

    ticker = sys.argv[1] if len(sys.argv) > 1 else "BTC-USD"
    tf = sys.argv[2] if len(sys.argv) > 2 else "1d"
    lookback = int(sys.argv[3]) if len(sys.argv) > 3 else 730

    print(f"Loading {ticker} {tf}, lookback={lookback}d…")
    series = fetch_close(ticker, tf, lookback_days=lookback)
    if series is None or series.empty:
        print(f"  ✗ no data for {ticker} {tf}")
        sys.exit(1)
    print(f"  loaded {len(series)} bars from {series.index[0]} to {series.index[-1]}")

    # Major + minor with different thresholds
    major_swings = detect_zigzag(series, threshold_pct=0.08)
    minor_swings = detect_zigzag(series, threshold_pct=0.03)
    print(f"  major_swings={len(major_swings)} minor_swings={len(minor_swings)}")

    major, minor = segment_with_subwaves(major_swings, minor_swings)
    print()
    print("MAJOR segments:")
    for i, seg in enumerate(major):
        print(f"  [{i:3d}] {seg.t_start.date()} → {seg.t_end.date()} "
              f"| {seg.pattern:<22} | score={seg.score:5.1f}")
    print()
    print("Summary:")
    print(f"  major: {summarize(major)}")
    print(f"  minor: {summarize(minor)}")
    print(f"  major coverage: {coverage_pct(major, len(major_swings)):.1f}%")
    print(f"  minor coverage: {coverage_pct(minor, len(minor_swings)):.1f}%")
