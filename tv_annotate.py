"""Annotate a TradingView screenshot with Claude's chart_annotations.

Takes the raw TV screenshot captured by tv_screenshot.py + the OHLC window
shown on the chart + Claude's structured chart_annotations list, and paints
horizontal lines, pivots, zones, arrows, wave labels and notes on top of
the screenshot via PIL.

Coordinate mapping
------------------
TV chart screenshots have a predictable layout. The dark default layout
that Anton uses places the price axis on the right with a known padding.
We need to map (date, price) → (x, y) pixels.

The mapping needs to know:
  * chart area: (left_px, top_px, right_px, bottom_px) inside the image
  * x-axis range: (date_min, date_max) visible on the chart
  * y-axis range: (price_min, price_max) visible on the chart

We can derive ranges from the OHLC slice the screenshot covers (Claude saw
the same data when producing annotations). chart area constants are
configurable through the ``tv_layout`` parameter.

Falls back gracefully when annotations land outside the visible chart area
(they're simply skipped). When the screenshot can't be opened the function
returns False without raising — calling code should treat annotation as
best-effort.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:                                # pragma: no cover
    _PIL_AVAILABLE = False
    Image = ImageDraw = ImageFont = None  # type: ignore


# --------------------------------------------------------------------------
# TV layout descriptors — chart-area paddings inside the screenshot, by layout
# --------------------------------------------------------------------------
@dataclass
class TVLayout:
    """Pixel paddings of the chart-drawing area inside the screenshot.

    Anton's TV screenshots are 1600x900 dark default — chart area covers
    roughly the inner box with these margins. Tweak from UI if your TV
    layout differs.
    """
    left: int = 60       # space for left-side toolbar (if present)
    top: int = 80        # symbol bar + status row
    right: int = 90      # price axis on right
    bottom: int = 110    # time axis + indicator panels + footer


LAYOUTS: dict[str, TVLayout] = {
    "dark_default":   TVLayout(left=60, top=80, right=90, bottom=110),
    "wide_dark":      TVLayout(left=20, top=70, right=90, bottom=130),
    "no_toolbar":     TVLayout(left=10, top=70, right=90, bottom=100),
}


# --------------------------------------------------------------------------
# Color palette
# --------------------------------------------------------------------------
COLOR_RGBA: dict[str, tuple[int, int, int, int]] = {
    "red":     (240, 80, 80, 255),
    "green":   (90, 190, 100, 255),
    "blue":    (68, 147, 248, 255),
    "cyan":    (0, 200, 240, 255),
    "yellow":  (210, 153, 34, 255),
    "gray":    (180, 180, 180, 255),
    "white":   (240, 240, 240, 255),
    "info":    (0, 200, 240, 255),
    "warning": (210, 153, 34, 255),
    "danger":  (240, 80, 80, 255),
}


def _rgba(name: str, alpha: int | None = None) -> tuple[int, int, int, int]:
    c = COLOR_RGBA.get(name.lower(), COLOR_RGBA["cyan"])
    if alpha is None:
        return c
    return (c[0], c[1], c[2], alpha)


# --------------------------------------------------------------------------
# Coordinate mapping
# --------------------------------------------------------------------------
@dataclass
class ChartFrame:
    """Maps (date, price) ↔ pixel (x, y) for one TV screenshot."""
    width: int
    height: int
    layout: TVLayout
    date_min: pd.Timestamp
    date_max: pd.Timestamp
    price_min: float
    price_max: float

    @property
    def x_left(self) -> int:
        return self.layout.left

    @property
    def x_right(self) -> int:
        return self.width - self.layout.right

    @property
    def y_top(self) -> int:
        return self.layout.top

    @property
    def y_bottom(self) -> int:
        return self.height - self.layout.bottom

    def price_to_y(self, price: float) -> int | None:
        """Convert a price to y-pixel; None if out of range."""
        if self.price_max <= self.price_min:
            return None
        ratio = (price - self.price_min) / (self.price_max - self.price_min)
        if not 0.0 <= ratio <= 1.0:
            return None
        # Higher price → smaller y (top of chart)
        return int(self.y_bottom - ratio * (self.y_bottom - self.y_top))

    def date_to_x(self, date: pd.Timestamp) -> int | None:
        """Convert a date to x-pixel; None if out of range."""
        if self.date_max <= self.date_min:
            return None
        span = (self.date_max - self.date_min).total_seconds()
        if span <= 0:
            return None
        offset = (date - self.date_min).total_seconds()
        ratio = offset / span
        if not 0.0 <= ratio <= 1.0:
            return None
        return int(self.x_left + ratio * (self.x_right - self.x_left))


def _build_frame(image_path: str, ohlc: pd.DataFrame | pd.Series,
                  layout_name: str = "dark_default") -> ChartFrame | None:
    """Build a ChartFrame from screenshot + the OHLC window it covers."""
    if not _PIL_AVAILABLE or not os.path.isfile(image_path) or len(ohlc) == 0:
        return None
    try:
        with Image.open(image_path) as im:
            w, h = im.size
    except Exception:
        return None

    if isinstance(ohlc, pd.Series):
        prices = ohlc
        price_min = float(prices.min())
        price_max = float(prices.max())
    else:
        # DataFrame — use high/low if present, else close
        if "high" in ohlc.columns and "low" in ohlc.columns:
            price_min = float(ohlc["low"].min())
            price_max = float(ohlc["high"].max())
        else:
            close_col = "close" if "close" in ohlc.columns else ohlc.columns[0]
            price_min = float(ohlc[close_col].min())
            price_max = float(ohlc[close_col].max())

    # Add 2% breathing room — TV charts usually do the same
    price_pad = (price_max - price_min) * 0.02
    price_min -= price_pad
    price_max += price_pad

    layout = LAYOUTS.get(layout_name, LAYOUTS["dark_default"])
    return ChartFrame(
        width=w, height=h, layout=layout,
        date_min=pd.Timestamp(ohlc.index[0]),
        date_max=pd.Timestamp(ohlc.index[-1]),
        price_min=price_min, price_max=price_max,
    )


# --------------------------------------------------------------------------
# Drawing primitives
# --------------------------------------------------------------------------
def _get_font(size: int) -> Any:
    """Load a font; fall back to default if system font unavailable."""
    if not _PIL_AVAILABLE:
        return None
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _draw_label_box(draw: Any, xy: tuple[int, int], text: str,
                     color: tuple[int, int, int, int], font: Any,
                     anchor: str = "lt") -> None:
    """Draw a text label with a background box."""
    if not text:
        return
    x, y = xy
    try:
        bbox = draw.textbbox((x, y), text, font=font, anchor=anchor)
    except (AttributeError, TypeError):
        # Older PIL versions don't support textbbox
        try:
            tw, th = draw.textsize(text, font=font)
        except Exception:
            tw, th = len(text) * 6, 14
        bbox = (x, y, x + tw, y + th)
    pad = 3
    bg_box = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    draw.rectangle(bg_box, fill=(11, 14, 19, 220), outline=color, width=1)
    draw.text((x, y), text, fill=color, font=font, anchor=anchor)


def _parse_date(d: Any) -> pd.Timestamp | None:
    if d is None:
        return None
    try:
        return pd.Timestamp(d)
    except Exception:
        return None


# --------------------------------------------------------------------------
# Annotation renderers per-type
# --------------------------------------------------------------------------
def _draw_pivot(draw: Any, frame: ChartFrame, ann: dict, font: Any) -> None:
    price = ann.get("price")
    if not isinstance(price, (int, float)):
        return
    y = frame.price_to_y(float(price))
    if y is None:
        return
    d = _parse_date(ann.get("date"))
    x = frame.date_to_x(d) if d else frame.x_right - 20
    if x is None:
        x = frame.x_right - 20
    color = _rgba(str(ann.get("color", "yellow")))
    # Diamond marker
    s = 7
    draw.polygon(
        [(x, y - s), (x + s, y), (x, y + s), (x - s, y)],
        fill=color, outline=(11, 14, 19, 255),
    )
    label = str(ann.get("label", ""))[:30]
    if label:
        _draw_label_box(draw, (x + s + 4, y - 6), label, color, font,
                        anchor="lt")


def _draw_fib_level(draw: Any, frame: ChartFrame, ann: dict, font: Any) -> None:
    price = ann.get("price")
    if not isinstance(price, (int, float)):
        return
    y = frame.price_to_y(float(price))
    if y is None:
        return
    color = _rgba(str(ann.get("color", "cyan")))
    # Dotted line — PIL doesn't have built-in dots, simulate with short segments
    for x_start in range(frame.x_left, frame.x_right, 12):
        x_end = min(x_start + 6, frame.x_right)
        draw.line([(x_start, y), (x_end, y)], fill=color, width=1)
    # Right-edge label
    label_text = f"{ann.get('label', '')}  {price:.2f}".strip()
    _draw_label_box(draw, (frame.x_right + 4, y - 8),
                     label_text, color, font, anchor="lt")


def _draw_zone(draw: Any, frame: ChartFrame, ann: dict, font: Any) -> None:
    lo = ann.get("price_low")
    hi = ann.get("price_high")
    if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float))):
        return
    y0 = frame.price_to_y(float(min(lo, hi)))
    y1 = frame.price_to_y(float(max(lo, hi)))
    if y0 is None or y1 is None:
        return
    color = _rgba(str(ann.get("color", "cyan")))
    fill = (color[0], color[1], color[2], 50)
    # Translucent rect — overlay layer
    overlay = Image.new("RGBA", (frame.x_right - frame.x_left,
                                  abs(y0 - y1)), fill)
    # We can't directly paste with alpha onto draw — need to handle by caller
    # via the parent annotate_screenshot. For inline simplicity, draw a
    # bordered rect with semi-transparent overlay added separately.
    draw.rectangle(
        (frame.x_left, min(y0, y1), frame.x_right, max(y0, y1)),
        fill=None, outline=color, width=1,
    )
    label = str(ann.get("label", ""))[:30]
    if label:
        _draw_label_box(draw, (frame.x_right + 4, (y0 + y1) // 2 - 8),
                        label, color, font, anchor="lt")


def _draw_arrow(draw: Any, frame: ChartFrame, ann: dict, font: Any) -> None:
    fp = ann.get("from_price")
    tp = ann.get("to_price")
    if not (isinstance(fp, (int, float)) and isinstance(tp, (int, float))):
        return
    y0 = frame.price_to_y(float(fp))
    y1 = frame.price_to_y(float(tp))
    if y0 is None or y1 is None:
        return
    fd = _parse_date(ann.get("from_date"))
    td = _parse_date(ann.get("to_date"))
    x0 = frame.date_to_x(fd) if fd else frame.x_left + 30
    x1 = frame.date_to_x(td) if td else frame.x_right - 30
    if x0 is None: x0 = frame.x_left + 30
    if x1 is None: x1 = frame.x_right - 30
    color = _rgba(str(ann.get("color", "green")))
    # Line
    draw.line([(x0, y0), (x1, y1)], fill=color, width=2)
    # Arrowhead at (x1, y1)
    import math
    dx = x1 - x0
    dy = y1 - y0
    angle = math.atan2(dy, dx)
    head_len = 12
    head_angle = math.radians(25)
    ax = x1 - head_len * math.cos(angle - head_angle)
    ay = y1 - head_len * math.sin(angle - head_angle)
    bx = x1 - head_len * math.cos(angle + head_angle)
    by = y1 - head_len * math.sin(angle + head_angle)
    draw.polygon([(x1, y1), (ax, ay), (bx, by)], fill=color)
    label = str(ann.get("label", ""))[:30]
    if label:
        mid_x = (x0 + x1) // 2
        mid_y = (y0 + y1) // 2 - 12
        _draw_label_box(draw, (mid_x, mid_y), label, color, font,
                        anchor="mm")


def _draw_wave_label(draw: Any, frame: ChartFrame, ann: dict, font: Any) -> None:
    price = ann.get("price")
    if not isinstance(price, (int, float)):
        return
    y = frame.price_to_y(float(price))
    if y is None:
        return
    d = _parse_date(ann.get("date"))
    x = frame.date_to_x(d) if d else frame.x_right - 30
    if x is None:
        x = frame.x_right - 30
    color = _rgba(str(ann.get("color", "yellow")))
    label = str(ann.get("label", "?"))[:10]
    # Big bold square label like classic Elliott markings
    big_font = _get_font(16)
    try:
        bbox = draw.textbbox((x, y), label, font=big_font, anchor="mm")
    except (AttributeError, TypeError):
        bbox = (x - 12, y - 10, x + 12, y + 10)
    pad = 4
    box = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    draw.rectangle(box, fill=color, outline=(11, 14, 19, 255), width=1)
    draw.text((x, y), label, fill=(11, 14, 19, 255), font=big_font,
              anchor="mm")


def _draw_note(draw: Any, frame: ChartFrame, ann: dict, font: Any) -> None:
    price = ann.get("price")
    if not isinstance(price, (int, float)):
        return
    y = frame.price_to_y(float(price))
    if y is None:
        return
    color = _rgba(str(ann.get("color", "warning")))
    # Dash-dot line
    pattern = [(0, 6), (8, 4), (14, 2)]  # (offset, length) tuples for each segment in 16px cycle
    for x_start in range(frame.x_left, frame.x_right, 16):
        for offset, length in pattern:
            x0 = x_start + offset
            x1 = min(x0 + length, frame.x_right)
            if x0 < frame.x_right:
                draw.line([(x0, y), (x1, y)], fill=color, width=1)
    label = str(ann.get("label", ""))[:40]
    if label:
        _draw_label_box(draw, (frame.x_right + 4, y - 8),
                        f"📝 {label}" if not label.startswith("📝") else label,
                        color, font, anchor="lt")


_RENDERERS: dict[str, Any] = {
    "pivot":      _draw_pivot,
    "fib_level":  _draw_fib_level,
    "zone":       _draw_zone,
    "arrow":      _draw_arrow,
    "wave_label": _draw_wave_label,
    "note":       _draw_note,
}


# --------------------------------------------------------------------------
# Top-level entrypoint
# --------------------------------------------------------------------------
def annotate_screenshot(
    screenshot_path: str,
    chart_annotations: list[dict],
    ohlc_window: pd.DataFrame | pd.Series,
    output_path: str,
    *,
    layout_name: str = "dark_default",
    target_tf: str = "1d",
    add_legend: bool = True,
) -> bool:
    """Paint Claude's chart_annotations on top of a TV screenshot.

    Parameters
    ----------
    screenshot_path : str
        Raw TV chart screenshot (PNG).
    chart_annotations : list[dict]
        Each dict has a ``type`` field and type-specific coordinate fields.
        See wave_ai.build_holistic_prompt for the schema.
    ohlc_window : DataFrame | Series
        OHLC data (or close-price series) covering the same date range
        shown on the screenshot. Used for coordinate mapping.
    output_path : str
        Where to save the annotated PNG. Overwritten if exists.
    layout_name : str
        TV layout descriptor for chart-area padding.
    target_tf : str
        Only annotations whose ``tf`` matches this TF (or "all") will be
        drawn — e.g. 4h annotations skipped on a 1d screenshot.
    add_legend : bool
        Add a small "📌 CLAUDE annotations" badge in the bottom-left.

    Returns
    -------
    bool — True if at least one annotation was painted and file saved.
    """
    if not _PIL_AVAILABLE:
        return False
    if not os.path.isfile(screenshot_path):
        return False
    if not chart_annotations:
        return False

    frame = _build_frame(screenshot_path, ohlc_window, layout_name)
    if frame is None:
        return False

    try:
        base = Image.open(screenshot_path).convert("RGBA")
    except Exception:
        return False

    # Translucent zones live on a separate layer so we can alpha-blend
    zone_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    zone_draw = ImageDraw.Draw(zone_layer)

    # All other primitives go on the main drawing
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _get_font(11)

    n_drawn = 0
    for ann in chart_annotations:
        if not isinstance(ann, dict):
            continue
        a_tf = str(ann.get("tf", "all")).lower()
        if a_tf not in ("all", target_tf.lower()):
            continue
        a_type = str(ann.get("type", "")).lower()

        # Zone needs filled translucent layer separately
        if a_type == "zone":
            lo = ann.get("price_low")
            hi = ann.get("price_high")
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                y0 = frame.price_to_y(float(min(lo, hi)))
                y1 = frame.price_to_y(float(max(lo, hi)))
                if y0 is not None and y1 is not None:
                    color = _rgba(str(ann.get("color", "cyan")))
                    fill = (color[0], color[1], color[2], 50)
                    zone_draw.rectangle(
                        (frame.x_left, min(y0, y1),
                         frame.x_right, max(y0, y1)),
                        fill=fill,
                    )
            # also draw border + label on overlay
            renderer = _RENDERERS.get("zone")
            if renderer is not None:
                renderer(draw, frame, ann, font)
                n_drawn += 1
            continue

        renderer = _RENDERERS.get(a_type)
        if renderer is None:
            continue
        try:
            renderer(draw, frame, ann, font)
            n_drawn += 1
        except Exception:
            continue

    if n_drawn == 0:
        return False

    # Compose: base + zone layer (alpha) + overlay
    result = Image.alpha_composite(base, zone_layer)
    result = Image.alpha_composite(result, overlay)

    if add_legend:
        legend_draw = ImageDraw.Draw(result)
        big_font = _get_font(13)
        legend_text = f"📌 CLAUDE · {target_tf} · {n_drawn} annotations"
        _draw_label_box(
            legend_draw,
            (frame.x_left + 8, frame.y_bottom - 24),
            legend_text,
            _rgba("cyan"), big_font, anchor="lt",
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    try:
        # Convert to RGB for PNG; alpha not needed for final file
        result.convert("RGB").save(output_path, "PNG", optimize=True)
    except Exception:
        return False
    return True


# --------------------------------------------------------------------------
# Convenience wrapper for the holistic pipeline
# --------------------------------------------------------------------------
def annotate_for_holistic(
    prediction_id: str,
    chart_annotations: list[dict],
    ohlc_by_tf: dict[str, pd.DataFrame | pd.Series],
    screenshots_dir: str = "predictions/screenshots",
    annotated_dir: str = "predictions/screenshots_annotated",
    layout_name: str = "dark_default",
) -> dict[str, str]:
    """For each TF that has both a screenshot AND annotations: produce
    annotated_dir/{prediction_id}_{tf}_annotated.png.

    Returns ``{tf: path}`` for every successfully annotated file. Errors
    are swallowed to keep the holistic call from failing because of post-
    processing.
    """
    out: dict[str, str] = {}
    if not chart_annotations:
        return out
    os.makedirs(annotated_dir, exist_ok=True)
    for tf, ohlc in (ohlc_by_tf or {}).items():
        if ohlc is None or len(ohlc) == 0:
            continue
        # Match the filename convention used by prediction_log.log_holistic:
        # tv_screenshot.capture writes "{pid}_{tf}" (no extension), let it
        # be the prefix and accept .png or .jpg
        for ext in (".png", ".jpg", ".jpeg"):
            src = os.path.join(screenshots_dir, f"{prediction_id}_{tf}{ext}")
            if os.path.isfile(src):
                break
        else:
            continue
        dst = os.path.join(annotated_dir,
                            f"{prediction_id}_{tf}_annotated.png")
        ok = annotate_screenshot(
            src, chart_annotations, ohlc, dst,
            layout_name=layout_name, target_tf=tf,
        )
        if ok:
            out[tf] = dst
    return out
