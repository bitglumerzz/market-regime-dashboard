"""Smoke tests for tv_annotate — PIL overlay of Claude annotations onto
a synthetic TV screenshot."""
from __future__ import annotations

import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _make_synthetic_tv_screenshot(path: str, w: int = 1600, h: int = 900) -> None:
    """Render a black image with a fake price-axis area for testing."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (w, h), (15, 23, 42))
    d = ImageDraw.Draw(im)
    # Draw rough chart area outline
    d.rectangle((60, 80, w - 90, h - 110), outline=(60, 80, 100), width=2)
    im.save(path, "PNG")


def _make_ohlc(rows: int = 30) -> pd.DataFrame:
    """Synthetic OHLC for coordinate-mapping tests."""
    idx = pd.date_range("2026-04-01", periods=rows, freq="4h")
    base = 70000.0
    return pd.DataFrame({
        "high":  [base + i * 100 + 50 for i in range(rows)],
        "low":   [base + i * 100 - 50 for i in range(rows)],
        "close": [base + i * 100      for i in range(rows)],
    }, index=idx)


def test_annotation_pipeline_basic():
    """Full pipeline: synthetic screenshot + 3 typed annotations → annotated PNG."""
    import tv_annotate
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "tv.png")
        dst = os.path.join(tmp, "annotated.png")
        _make_synthetic_tv_screenshot(src)
        ohlc = _make_ohlc(30)
        annotations = [
            {"type": "fib_level", "tf": "4h", "price": 72000,
             "label": "38.2%", "color": "cyan"},
            {"type": "zone", "tf": "4h",
             "price_low": 70500, "price_high": 71500,
             "label": "C-target", "color": "green"},
            {"type": "pivot", "tf": "4h", "price": 72800,
             "label": "Peak", "color": "yellow"},
        ]
        ok = tv_annotate.annotate_screenshot(
            src, annotations, ohlc, dst, target_tf="4h",
        )
        assert ok, "annotation should succeed"
        assert os.path.isfile(dst), "output file should exist"
        assert os.path.getsize(dst) > 0
    print("  ✓ annotation pipeline produces a PNG")


def test_tf_filter():
    """Annotations targeting other TFs should be skipped."""
    import tv_annotate
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "tv.png")
        dst_4h = os.path.join(tmp, "ann_4h.png")
        dst_1d = os.path.join(tmp, "ann_1d.png")
        _make_synthetic_tv_screenshot(src)
        ohlc = _make_ohlc(30)
        annotations = [
            {"type": "fib_level", "tf": "4h", "price": 72000, "label": "4h"},
            {"type": "fib_level", "tf": "1d", "price": 72500, "label": "1d"},
        ]
        ok_4h = tv_annotate.annotate_screenshot(
            src, annotations, ohlc, dst_4h, target_tf="4h",
        )
        ok_1d = tv_annotate.annotate_screenshot(
            src, annotations, ohlc, dst_1d, target_tf="1d",
        )
        assert ok_4h and ok_1d
    print("  ✓ tf filter — each TF gets its annotation")


def test_out_of_range_skipped():
    """Prices outside chart range should be silently skipped, not crash."""
    import tv_annotate
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "tv.png")
        dst = os.path.join(tmp, "ann.png")
        _make_synthetic_tv_screenshot(src)
        ohlc = _make_ohlc(30)
        annotations = [
            {"type": "fib_level", "tf": "4h", "price": 999999,
             "label": "way too high"},
            {"type": "fib_level", "tf": "4h", "price": 72000,
             "label": "valid"},
        ]
        ok = tv_annotate.annotate_screenshot(
            src, annotations, ohlc, dst, target_tf="4h",
        )
        assert ok, "should still succeed on the valid annotation"
    print("  ✓ out-of-range annotations skipped gracefully")


def test_missing_screenshot_returns_false():
    import tv_annotate
    ok = tv_annotate.annotate_screenshot(
        "/tmp/nonexistent_xyz_123.png", [{"type": "fib_level"}],
        _make_ohlc(10), "/tmp/dummy_out.png",
    )
    assert ok is False
    print("  ✓ missing screenshot → False, no crash")


def test_empty_annotations_returns_false():
    import tv_annotate
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "tv.png")
        _make_synthetic_tv_screenshot(src)
        ok = tv_annotate.annotate_screenshot(
            src, [], _make_ohlc(10), os.path.join(tmp, "out.png"),
        )
        assert ok is False
    print("  ✓ empty annotations → False")


def test_coordinate_mapping():
    """Verify price_to_y and date_to_x roundtrip sanity."""
    import tv_annotate
    ohlc = _make_ohlc(30)
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "tv.png")
        _make_synthetic_tv_screenshot(src, w=1600, h=900)
        frame = tv_annotate._build_frame(src, ohlc, "dark_default")
        assert frame is not None

        # Mid-price → mid-y
        mid_price = (frame.price_min + frame.price_max) / 2
        y = frame.price_to_y(mid_price)
        assert y is not None
        chart_mid_y = (frame.y_top + frame.y_bottom) / 2
        assert abs(y - chart_mid_y) < 5, \
            f"midprice should map near chart center: y={y}, mid={chart_mid_y}"

        # Out-of-range price → None
        assert frame.price_to_y(frame.price_min - 1000) is None
        assert frame.price_to_y(frame.price_max + 1000) is None

        # First date → near x_left
        x = frame.date_to_x(ohlc.index[0])
        assert x is not None
        assert abs(x - frame.x_left) < 5
        # Last date → near x_right
        x = frame.date_to_x(ohlc.index[-1])
        assert x is not None
        assert abs(x - frame.x_right) < 5
    print("  ✓ coordinate mapping accurate")


def test_arrow_renders():
    """Arrow needs from_price + to_price; should not crash without dates."""
    import tv_annotate
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "tv.png")
        dst = os.path.join(tmp, "ann.png")
        _make_synthetic_tv_screenshot(src)
        ohlc = _make_ohlc(30)
        annotations = [{
            "type": "arrow", "tf": "4h",
            "from_price": 70500, "to_price": 72500,
            "label": "wave 3 impulse", "color": "green",
        }]
        ok = tv_annotate.annotate_screenshot(
            src, annotations, ohlc, dst, target_tf="4h",
        )
        assert ok
    print("  ✓ arrow renders without dates")


def test_all_six_types_together():
    """All six annotation types in one screenshot."""
    import tv_annotate
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "tv.png")
        dst = os.path.join(tmp, "ann.png")
        _make_synthetic_tv_screenshot(src)
        ohlc = _make_ohlc(40)
        annotations = [
            {"type": "pivot", "tf": "4h", "price": 73500,
             "label": "Peak 2", "color": "yellow"},
            {"type": "fib_level", "tf": "4h", "price": 72500, "label": "38.2%"},
            {"type": "fib_level", "tf": "4h", "price": 71800, "label": "50%"},
            {"type": "zone", "tf": "4h",
             "price_low": 70500, "price_high": 71500,
             "label": "target zone", "color": "green"},
            {"type": "arrow", "tf": "4h",
             "from_price": 70200, "to_price": 73000,
             "label": "impulse"},
            {"type": "wave_label", "tf": "4h",
             "price": 73000, "label": "2", "color": "red"},
            {"type": "note", "tf": "4h", "price": 70000,
             "label": "пробой = инвалидация", "color": "warning"},
        ]
        ok = tv_annotate.annotate_screenshot(
            src, annotations, ohlc, dst, target_tf="4h",
        )
        assert ok
        assert os.path.getsize(dst) > 500  # non-trivial file
    print("  ✓ all 6 annotation types render together")


if __name__ == "__main__":
    print("Running tv_annotate smoke tests…")
    test_annotation_pipeline_basic()
    test_tf_filter()
    test_out_of_range_skipped()
    test_missing_screenshot_returns_false()
    test_empty_annotations_returns_false()
    test_coordinate_mapping()
    test_arrow_renders()
    test_all_six_types_together()
    print("\nAll tv_annotate smoke tests passed ✓")
