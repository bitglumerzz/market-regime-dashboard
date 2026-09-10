"""Smoke tests for multi_label_swings — the multi-pattern wave labeling
that covers the WHOLE swing series, not just the last detected pattern.
"""
from __future__ import annotations

import sys
import os

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _make_swing(price: float, ts_offset_days: int, bar: int, kind: str):
    from zigzag import Swing
    return Swing(
        index=pd.Timestamp("2024-01-01") + pd.Timedelta(days=ts_offset_days),
        price=price, kind=kind, bar_index=bar, provisional=False,
    )


def test_impulse_up_basic():
    """6-swing alternating up impulse — should label 0-1-2-3-4-5."""
    from elliott import multi_label_swings
    swings = [
        _make_swing(100.0, 0, 0, "low"),
        _make_swing(120.0, 5, 5, "high"),    # wave 1
        _make_swing(110.0, 10, 10, "low"),   # wave 2 (50% retrace)
        _make_swing(150.0, 15, 15, "high"),  # wave 3 (extended)
        _make_swing(135.0, 20, 20, "low"),   # wave 4
        _make_swing(160.0, 25, 25, "high"),  # wave 5
    ]
    labels = multi_label_swings(swings, min_score=20.0)  # low bar for synthetic data
    assert len(labels) >= 4, f"expected at least 4 labels, got {len(labels)}"
    # First few labels should be impulse 0,1,2,3,...
    first_labels = [ls.label for ls in labels[:6]]
    assert "0" in first_labels or "1" in first_labels, \
        f"expected wave labels, got {first_labels}"
    print(f"  ✓ basic impulse_up: {len(labels)} swings labeled, "
          f"labels={[ls.label for ls in labels]}")


def test_back_to_back_patterns():
    """Two impulses end-to-end → should get two labeled segments."""
    from elliott import multi_label_swings
    # Pattern 1: impulse up from 100 to 160
    # Then: zigzag correction A-B-C
    swings = [
        _make_swing(100.0, 0, 0, "low"),
        _make_swing(120.0, 5, 5, "high"),
        _make_swing(110.0, 10, 10, "low"),
        _make_swing(150.0, 15, 15, "high"),
        _make_swing(135.0, 20, 20, "low"),
        _make_swing(160.0, 25, 25, "high"),    # end of impulse 1
        _make_swing(140.0, 30, 30, "low"),     # A of correction
        _make_swing(155.0, 35, 35, "high"),    # B
        _make_swing(130.0, 40, 40, "low"),     # C
    ]
    labels = multi_label_swings(swings, min_score=20.0)
    # We should see at least one pattern, possibly two depending on threshold
    assert len(labels) >= 4, f"got {len(labels)} labels"
    # All emitted labels must reference real swings from the input
    swing_indices = {id(s) for s in swings}
    for ls in labels:
        assert id(ls.swing) in swing_indices, \
            "labeled swing not from input series"
    # Labels should be sorted by bar_index (chronological)
    bars = [ls.swing.bar_index for ls in labels]
    assert bars == sorted(bars), "labels not chronologically ordered"
    print(f"  ✓ back-to-back: {len(labels)} labels in chronological order")


def test_no_duplicate_swings():
    """A single swing cannot appear twice with different labels."""
    from elliott import multi_label_swings
    swings = [
        _make_swing(100.0 + i * 5, i, i, "low" if i % 2 == 0 else "high")
        for i in range(12)
    ]
    labels = multi_label_swings(swings, min_score=10.0)
    seen_swing_ids = set()
    for ls in labels:
        sid = id(ls.swing)
        assert sid not in seen_swing_ids, \
            f"swing at bar {ls.swing.bar_index} labeled twice"
        seen_swing_ids.add(sid)
    print(f"  ✓ no duplicates across {len(labels)} labels")


def test_empty_and_tiny_series():
    """Empty or too-short series should return empty list, no crash."""
    from elliott import multi_label_swings
    assert multi_label_swings([]) == []
    assert multi_label_swings([_make_swing(100.0, 0, 0, "low")]) == []
    assert multi_label_swings([
        _make_swing(100.0, 0, 0, "low"),
        _make_swing(110.0, 1, 1, "high"),
    ]) == []
    print("  ✓ empty/tiny series handled gracefully")


def test_high_min_score_filters_noise():
    """With a very high min_score, only the best-formed patterns survive."""
    from elliott import multi_label_swings
    swings = [
        _make_swing(100.0 + (i * 7 % 30), i, i, "low" if i % 2 == 0 else "high")
        for i in range(20)
    ]
    relaxed = multi_label_swings(swings, min_score=10.0)
    strict = multi_label_swings(swings, min_score=80.0)
    assert len(strict) <= len(relaxed), \
        f"strict={len(strict)} should be ≤ relaxed={len(relaxed)}"
    print(f"  ✓ min_score filter works: relaxed={len(relaxed)} strict={len(strict)}")


def test_labeled_swing_dataclass():
    """LabeledSwing has all expected fields."""
    from elliott import multi_label_swings
    swings = [
        _make_swing(100.0, 0, 0, "low"),
        _make_swing(120.0, 5, 5, "high"),
        _make_swing(110.0, 10, 10, "low"),
        _make_swing(150.0, 15, 15, "high"),
        _make_swing(135.0, 20, 20, "low"),
        _make_swing(160.0, 25, 25, "high"),
    ]
    labels = multi_label_swings(swings, min_score=20.0)
    if labels:
        ls = labels[0]
        assert hasattr(ls, "swing")
        assert hasattr(ls, "label")
        assert hasattr(ls, "pattern")
        assert hasattr(ls, "pattern_index")
        assert hasattr(ls, "is_up")
        assert hasattr(ls, "score")
        assert isinstance(ls.is_up, bool)
        assert isinstance(ls.score, (int, float))
    print(f"  ✓ LabeledSwing dataclass fields present")


if __name__ == "__main__":
    print("Running multi_label_swings smoke tests…")
    test_impulse_up_basic()
    test_back_to_back_patterns()
    test_no_duplicate_swings()
    test_empty_and_tiny_series()
    test_high_min_score_filters_noise()
    test_labeled_swing_dataclass()
    print("\nAll multi-label smoke tests passed ✓")
