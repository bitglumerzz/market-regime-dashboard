"""Design system: dark theme, regime colors, HTML component helpers.

All visual identity lives here. Other modules import REGIME_COLORS and the
helper functions; they should never hard-code hex values.
"""
from __future__ import annotations

import colorsys
from typing import Any

import streamlit as st


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
ACCENT_CYAN: str = "#00D4FF"
BG_PRIMARY: str = "#0E1117"
BG_CARD: str = "#161B22"
BG_CARD_ALT: str = "#1C2230"
TEXT_PRIMARY: str = "#E6EDF3"
TEXT_MUTED: str = "#8B949E"
BORDER_SUBTLE: str = "#21262D"

# Canonical regime labels mapped to hex colors.
# Additional / generated regime names (e.g. "Vol Regime 2") use HSL
# interpolation between Low and High via `get_regime_color`.
REGIME_COLORS: dict[str, str] = {
    "Low Vol": "#00C9A7",          # teal
    "Medium-Low Vol": "#7FD78A",   # green-teal
    "Medium Vol": "#FFB347",       # amber
    "Medium-High Vol": "#FF8E5C",  # orange
    "High Vol": "#FF6B6B",         # red
    "Uncertain": "#808080",        # grey
}

# Threshold lines, opacity defaults — pulled out of magic-number territory.
REGIME_BAND_OPACITY: float = 0.13
CONFIDENCE_THRESHOLD: float = 0.75
CONFIDENCE_FILL_OPACITY: float = 0.30


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------
def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    return "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))


def _interpolate_hsl(low_hex: str, high_hex: str, t: float) -> str:
    """Interpolate two hex colors in HSL space (smoother than RGB lerp)."""
    r1, g1, b1 = _hex_to_rgb(low_hex)
    r2, g2, b2 = _hex_to_rgb(high_hex)
    h1, l1, s1 = colorsys.rgb_to_hls(r1, g1, b1)
    h2, l2, s2 = colorsys.rgb_to_hls(r2, g2, b2)
    h = h1 + (h2 - h1) * t
    l = l1 + (l2 - l1) * t
    s = s1 + (s2 - s1) * t
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return _rgb_to_hex(r, g, b)


def get_regime_color(label: str, fallback_index: int = 0, total: int = 1) -> str:
    """Return a hex color for any regime label.

    Known labels in REGIME_COLORS resolve directly. Generated labels
    ("Vol Regime N") are interpolated from Low Vol → High Vol so that the
    color order on the chart still tracks volatility ordering.
    """
    if label in REGIME_COLORS:
        return REGIME_COLORS[label]
    t = fallback_index / max(total - 1, 1)
    return _interpolate_hsl(REGIME_COLORS["Low Vol"], REGIME_COLORS["High Vol"], t)


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert hex → rgba() string for Plotly fills."""
    r, g, b = _hex_to_rgb(hex_color)
    return f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {alpha:.3f})"


# ---------------------------------------------------------------------------
# Theme / page config
# ---------------------------------------------------------------------------
def apply_theme() -> None:
    """Set Streamlit page config and inject dark theme CSS.

    Must be called once, at the very top of app.py, before any other st.* call.
    """
    st.set_page_config(
        page_title="Market Regime Detection",
        page_icon="◉",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    css = f"""
    <style>
    /* App background + global text */
    .stApp {{
        background-color: {BG_PRIMARY};
        color: {TEXT_PRIMARY};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {BG_CARD};
        border-right: 1px solid {BORDER_SUBTLE};
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {TEXT_PRIMARY};
        font-weight: 600;
        letter-spacing: -0.01em;
    }}

    /* Regime pill badge */
    .regime-badge {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 13px;
        letter-spacing: 0.02em;
        color: #0B0E13;
    }}
    .regime-badge .conf {{
        opacity: 0.75;
        font-weight: 500;
        font-size: 11px;
    }}

    /* Metric card */
    .metric-card {{
        background: {BG_CARD};
        border: 1px solid {BORDER_SUBTLE};
        border-left: 4px solid {ACCENT_CYAN};
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 8px;
    }}
    .metric-card .title {{
        color: {TEXT_MUTED};
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }}
    .metric-card .value {{
        color: {TEXT_PRIMARY};
        font-size: 22px;
        font-weight: 600;
        line-height: 1.1;
    }}
    .metric-card .subtitle {{
        color: {TEXT_MUTED};
        font-size: 12px;
        margin-top: 4px;
    }}

    /* Section header */
    .section-header {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 20px 0 10px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid {BORDER_SUBTLE};
    }}
    .section-header .dot {{
        width: 8px;
        height: 8px;
        background: {ACCENT_CYAN};
        border-radius: 50%;
        box-shadow: 0 0 8px {ACCENT_CYAN};
    }}
    .section-header .text {{
        color: {TEXT_PRIMARY};
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}

    /* Top bar large value */
    .top-value {{
        color: {ACCENT_CYAN};
        font-size: 28px;
        font-weight: 700;
        line-height: 1;
    }}
    .top-label {{
        color: {TEXT_MUTED};
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }}
    .top-ticker {{
        color: {TEXT_PRIMARY};
        font-size: 30px;
        font-weight: 700;
        letter-spacing: -0.01em;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Component helpers — all return HTML strings (no st.* side effects unless noted)
# ---------------------------------------------------------------------------
def regime_badge(label: str, confidence: float) -> str:
    """Return HTML string for a colored pill badge with a soft glow.

    Use with: st.markdown(regime_badge(...), unsafe_allow_html=True).
    """
    color = REGIME_COLORS.get(label, REGIME_COLORS["Uncertain"])
    glow = hex_to_rgba(color, 0.55)
    conf_pct = f"{confidence:.0%}"
    return (
        f'<span class="regime-badge" '
        f'style="background:{color}; box-shadow: 0 0 14px {glow};">'
        f"{label}<span class=\"conf\">· {conf_pct}</span>"
        f"</span>"
    )


def metric_card(title: str, value: str, subtitle: str = "", border_color: str | None = None) -> str:
    """Return HTML string for a dark metric card with a colored left border."""
    color = border_color or ACCENT_CYAN
    subtitle_html = f'<div class="subtitle">{subtitle}</div>' if subtitle else ""
    return (
        f'<div class="metric-card" style="border-left-color:{color};">'
        f'<div class="title">{title}</div>'
        f'<div class="value">{value}</div>'
        f"{subtitle_html}"
        f"</div>"
    )


def section_header(text: str) -> None:
    """Render a styled section divider directly to the page."""
    st.markdown(
        f'<div class="section-header"><span class="dot"></span>'
        f'<span class="text">{text}</span></div>',
        unsafe_allow_html=True,
    )


def get_plotly_layout() -> dict[str, Any]:
    """Return a base Plotly layout dict matching the dark theme.

    Charts should merge this and only override what they need
    (titles, height, shapes, etc.).
    """
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": TEXT_PRIMARY, "family": "Inter, system-ui, sans-serif", "size": 12},
        "xaxis": {
            "showgrid": False,
            "zeroline": False,
            "showline": True,
            "linecolor": BORDER_SUBTLE,
            "tickcolor": BORDER_SUBTLE,
            "color": TEXT_MUTED,
        },
        "yaxis": {
            "showgrid": False,
            "zeroline": False,
            "showline": True,
            "linecolor": BORDER_SUBTLE,
            "tickcolor": BORDER_SUBTLE,
            "color": TEXT_MUTED,
        },
        "margin": {"l": 50, "r": 30, "t": 30, "b": 40},
        "hovermode": "x unified",
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": BORDER_SUBTLE,
            "borderwidth": 1,
            "font": {"color": TEXT_PRIMARY},
            "orientation": "h",
            "y": -0.15,
        },
    }
