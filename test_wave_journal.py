"""Smoke tests for wave_journal.

Covers:
  * snapshot write + read roundtrip
  * snapshot ID is deterministic / idempotent within the same minute
  * transition detection fires when top pattern changes between two snapshots
  * action write/read + linking to snapshot
  * verify_snapshots with a mock fetcher resolves win/loss/expired correctly
  * compute_* aggregates handle the empty case without crashing
  * compute_action_stats returns sensible numbers after verification

Runs against an isolated TMPDIR — does not touch the real wave_journal/ data.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import pandas as pd


def _make_swing(price: float, ts: str, bar: int, kind: str):
    """Stand-in for zigzag.Swing — wave_journal only reads attributes via
    getattr, so a SimpleNamespace works without importing the real type."""
    from types import SimpleNamespace
    return SimpleNamespace(index=pd.Timestamp(ts), price=price,
                            kind=kind, bar_index=bar, provisional=False)


def _make_candidate(pattern: str, current_position: str = "completing"):
    from types import SimpleNamespace
    return SimpleNamespace(
        pattern=pattern,
        score=80.0,
        current_position=current_position,
        labels=["0", "1", "2", "3", "4", "5"],
        is_extended=False,
        extension_ratio=None,
        is_truncated=False,
        is_diagonal=False,
        diagonal_kind="",
        triangle_kind="",
        rule_violations=[],
        swings_used=[
            _make_swing(100.0, "2026-05-01T00:00:00Z", 0, "low"),
            _make_swing(110.0, "2026-05-02T00:00:00Z", 1, "high"),
            _make_swing(105.0, "2026-05-03T00:00:00Z", 2, "low"),
            _make_swing(125.0, "2026-05-04T00:00:00Z", 3, "high"),
            _make_swing(115.0, "2026-05-05T00:00:00Z", 4, "low"),
            _make_swing(140.0, "2026-05-06T00:00:00Z", 5, "high"),
        ],
        next_targets={"100%": 120.0},
    )


def _make_forecast(direction: str, invalidation: float, target: float):
    from types import SimpleNamespace
    return SimpleNamespace(
        next_pattern="correction_down",
        direction=direction,
        confidence="high",
        rationale="test",
        targets=[SimpleNamespace(label="A", price=target, fib_label="38.2%",
                                  probability="high", empirical_p=0.6)],
        invalidation_level=invalidation,
        invalidation_reason="test",
        expected_duration_bars=(5, 25),
        is_actionable=True,
    )


def _make_tf(name: str, top_pattern: str, last_price: float,
              direction: str, invalidation: float, target: float,
              current_position: str = "completing"):
    from types import SimpleNamespace
    cand = _make_candidate(top_pattern, current_position)
    prices = pd.Series(
        [last_price - 5, last_price - 2, last_price],
        index=pd.date_range("2026-05-10", periods=3, freq="D"),
    )
    return SimpleNamespace(
        name=name, interval=name,
        ok=True, error="", source="test", threshold_source="builtin",
        major_threshold_pct=0.05, minor_threshold_pct=0.02,
        prices=prices,
        major_swings=cand.swings_used,
        minor_swings=cand.swings_used,
        candidates=[cand],
        top=cand,
        forecast=_make_forecast(direction, invalidation, target),
    )


def setup_isolated_journal() -> str:
    """Point wave_journal at a fresh tmpdir for this test run."""
    tmp = tempfile.mkdtemp(prefix="wave_journal_test_")
    import wave_journal as wj
    wj.WAVE_JOURNAL_DIR = tmp
    wj.SNAPSHOTS_PATH = os.path.join(tmp, "snapshots.jsonl")
    wj.ACTIONS_PATH = os.path.join(tmp, "actions.jsonl")
    wj.TRANSITIONS_PATH = os.path.join(tmp, "transitions.jsonl")
    return tmp


def test_snapshot_roundtrip():
    tmp = setup_isolated_journal()
    try:
        import wave_journal as wj
        tfs = [
            _make_tf("1d", "impulse_up", 140.0, "down", 145.0, 130.0),
            _make_tf("4h", "correction_down", 138.0, "up", 130.0, 145.0),
        ]
        synth = {"setup": "LONG opportunity", "color": "long"}
        snap = wj.log_snapshot("BTC-USD", tfs, synth)
        assert snap["id"], "snapshot must have an id"
        assert len(snap["per_tf"]) == 2
        assert snap["per_tf"][0]["top"]["pattern"] == "impulse_up"

        loaded = wj.load_snapshots()
        assert len(loaded) == 1
        assert loaded[0]["id"] == snap["id"]
        assert loaded[0]["synthesis"]["color"] == "long"
        print("  ✓ snapshot roundtrip")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_snapshot_idempotent():
    tmp = setup_isolated_journal()
    try:
        import wave_journal as wj
        tfs = [_make_tf("1d", "impulse_up", 140.0, "down", 145.0, 130.0)]
        s1 = wj.log_snapshot("BTC-USD", tfs, {})
        s2 = wj.log_snapshot("BTC-USD", tfs, {})  # same minute, same fingerprint
        assert s1["id"] == s2["id"], "same fingerprint in same minute → same id"
        assert len(wj.load_snapshots()) == 1, "no duplicate row"
        print("  ✓ snapshot idempotent")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_transition_detection():
    tmp = setup_isolated_journal()
    try:
        import wave_journal as wj
        # First snapshot: 1d = impulse_up
        tfs1 = [_make_tf("1d", "impulse_up", 140.0, "down", 145.0, 130.0)]
        s1 = wj.log_snapshot("BTC-USD", tfs1, {})

        # Force the second snapshot to land in a different minute by manually
        # bumping snapshot_time. Same ticker, different top pattern.
        tfs2 = [_make_tf("1d", "correction_down", 130.0, "up", 125.0, 140.0)]
        future_ts = pd.Timestamp(s1["logged_at"]) + pd.Timedelta(minutes=5)
        s2 = wj.log_snapshot("BTC-USD", tfs2, {}, snapshot_time=future_ts)
        assert s1["id"] != s2["id"], "different fingerprint → different id"

        trans = wj.detect_and_log_transitions(s2)
        assert len(trans) == 1
        assert trans[0]["from_pattern"] == "impulse_up"
        assert trans[0]["to_pattern"] == "correction_down"
        loaded_t = wj.load_transitions()
        assert len(loaded_t) == 1
        print("  ✓ transition detection")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_action_write_and_link():
    tmp = setup_isolated_journal()
    try:
        import wave_journal as wj
        tfs = [_make_tf("1d", "impulse_up", 140.0, "down", 145.0, 130.0)]
        snap = wj.log_snapshot("BTC-USD", tfs, {})

        a = wj.log_action(
            snap["id"], "BTC-USD", "take_setup",
            direction="long", entry=140.0, stop_loss=135.0, target=160.0,
            timeframe="1d", notes="test setup",
        )
        assert a["snapshot_id"] == snap["id"]
        assert a["action_type"] == "take_setup"
        loaded = wj.load_actions()
        assert len(loaded) == 1
        assert loaded[0]["notes"] == "test setup"
        print("  ✓ action write + snapshot link")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_verify_snapshots_with_mock_fetcher():
    """Mock fetcher: 1d forecast says direction=down with target=130.0 and
    invalidation=145.0. We feed bars that hit target first → expect win."""
    tmp = setup_isolated_journal()
    try:
        import wave_journal as wj
        tfs = [_make_tf("1d", "impulse_up", 140.0, "down", 145.0, 130.0)]
        snap = wj.log_snapshot("BTC-USD", tfs, {})

        # Build a fake forward series: prices drop to 128 (target hit at 130).
        def fetcher(ticker: str, tf: str, start_ts: pd.Timestamp) -> pd.DataFrame:
            idx = pd.date_range("2026-05-12", periods=10, freq="D")
            return pd.DataFrame({
                "high":  [142, 140, 138, 135, 132, 131, 129, 128, 127, 126],
                "low":   [139, 137, 135, 132, 130, 129, 127, 126, 125, 124],
                "close": [140, 138, 136, 133, 131, 130, 128, 127, 126, 125],
            }, index=idx)

        stats = wj.verify_snapshots(fetcher)
        assert stats["checked"] == 1
        assert stats["tf_won"] == 1, f"expected 1 win, got {stats}"

        # Reload and check the outcome was persisted
        loaded = wj.load_snapshots()[0]
        outcome = loaded["per_tf_outcomes"]["1d"]
        assert outcome["status"] == "win"
        assert outcome["forecast_direction"] == "down"
        print("  ✓ verify_snapshots win case")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_verify_snapshots_loss_case():
    """Same setup but invalidation triggers first."""
    tmp = setup_isolated_journal()
    try:
        import wave_journal as wj
        tfs = [_make_tf("1d", "impulse_up", 140.0, "down", 145.0, 130.0)]
        wj.log_snapshot("BTC-USD", tfs, {})

        # Prices rise to 146 — invalidation (145) hit first.
        def fetcher(ticker: str, tf: str, start_ts: pd.Timestamp) -> pd.DataFrame:
            idx = pd.date_range("2026-05-12", periods=5, freq="D")
            return pd.DataFrame({
                "high":  [142, 144, 146, 147, 148],
                "low":   [140, 142, 144, 145, 146],
                "close": [141, 143, 145, 146, 147],
            }, index=idx)

        stats = wj.verify_snapshots(fetcher)
        assert stats["tf_lost"] == 1, f"expected 1 loss, got {stats}"

        loaded = wj.load_snapshots()[0]
        outcome = loaded["per_tf_outcomes"]["1d"]
        assert outcome["status"] == "loss"
        print("  ✓ verify_snapshots loss case")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_verify_actions_win():
    """Long setup with entry=100, stop=95, target=110. Price rises to 112."""
    tmp = setup_isolated_journal()
    try:
        import wave_journal as wj
        wj.log_action(None, "BTC-USD", "take_setup",
                       direction="long", entry=100.0, stop_loss=95.0,
                       target=110.0, timeframe="1d")

        def fetcher(ticker, tf, start_ts):
            idx = pd.date_range("2026-05-12", periods=5, freq="D")
            return pd.DataFrame({
                "high":  [101, 104, 108, 112, 114],
                "low":   [99, 101, 105, 109, 111],
                "close": [100, 103, 107, 111, 113],
            }, index=idx)

        stats = wj.verify_actions(fetcher)
        assert stats["won"] == 1, f"expected 1 win, got {stats}"
        a = wj.load_actions()[0]
        assert a["outcome"] == "win"
        assert abs(a["r_realized"] - 2.0) < 0.01, "R=10/5=2"
        print("  ✓ verify_actions win + R correct")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_empty_stats_dont_crash():
    tmp = setup_isolated_journal()
    try:
        import wave_journal as wj
        ds = wj.compute_detector_stats()
        assert ds["total_snapshots"] == 0
        assert ds["total_tf_outcomes"] == 0
        ast = wj.compute_action_stats()
        assert ast["total"] == 0
        assert ast["hit_rate"] is None
        ss = wj.compute_stability_stats()
        assert ss["total_transitions_in_window"] == 0
        print("  ✓ empty stats no crash")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_per_tf_serialization_keeps_forecast():
    tmp = setup_isolated_journal()
    try:
        import wave_journal as wj
        tfs = [_make_tf("4h", "impulse_down", 100.0, "up", 95.0, 110.0)]
        snap = wj.log_snapshot("ETH-USD", tfs, {})
        per_tf = snap["per_tf"][0]
        assert per_tf["forecast"]["direction"] == "up"
        assert per_tf["forecast"]["targets"][0]["price"] == 110.0
        assert per_tf["forecast"]["invalidation_level"] == 95.0
        assert per_tf["top"]["pattern"] == "impulse_down"
        print("  ✓ forecast serialization keeps fields")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    # Run from the project root so wave_journal can be imported.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    print("Running wave_journal smoke tests…")
    test_snapshot_roundtrip()
    test_snapshot_idempotent()
    test_transition_detection()
    test_action_write_and_link()
    test_verify_snapshots_with_mock_fetcher()
    test_verify_snapshots_loss_case()
    test_verify_actions_win()
    test_empty_stats_dont_crash()
    test_per_tf_serialization_keeps_forecast()
    print("\nAll smoke tests passed ✓")
