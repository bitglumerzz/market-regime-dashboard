"""Correlation Break Detector — monitor pairs of normally-correlated assets.

The page renders, for each pair:
- a status card with current correlation, z-score, and break flag
- a chart showing short- vs long-window rolling correlation, with
  break regions shaded
- a list of historical break episodes (start, end, duration)

If any pair is currently broken, a banner at the top calls it out — that's
the signal most institutional teams react to.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from correlation import (
    BASELINE_WINDOW_DEFAULT,
    BREAK_THRESHOLD_DEFAULT,
    DEFAULT_PAIRS,
    LONG_WINDOW_DEFAULT,
    SHORT_WINDOW_DEFAULT,
    analyze_pair,
    break_episodes,
    parse_pair_list,
)
from design_system import (
    ACCENT_CYAN,
    BORDER_SUBTLE,
    REGIME_COLORS,
    TEXT_MUTED,
    TEXT_PRIMARY,
    apply_theme,
    get_plotly_layout,
    hex_to_rgba,
    metric_card,
    section_header,
)


apply_theme()

st.markdown(
    '<div style="display:flex; align-items:baseline; gap:14px;">'
    '<div style="font-size:24px; font-weight:700; letter-spacing:-0.02em;">'
    '◉ Correlation Break Detector</div>'
    '<div style="color:#8B949E; font-size:13px;">Catch when normally-coupled '
    'pairs decouple — often a leading macro/sector signal</div>'
    "</div>",
    unsafe_allow_html=True,
)
st.write("")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Pair configuration")

default_text = "\n".join(f"{a} / {b}" for a, b in DEFAULT_PAIRS)
pairs_text = st.sidebar.text_area(
    "Pairs to monitor",
    value=default_text,
    height=140,
    help=(
        "One pair per line, two tickers separated by `/`. "
        "Example: `SPY / QQQ`. Comma-separated also works."
    ),
)

today = dt.date.today()
start_default = today - dt.timedelta(days=365 * 3)
start_date = st.sidebar.date_input("Start Date", value=start_default,
                                    key="corr_start")
end_date = st.sidebar.date_input("End Date", value=today, key="corr_end")

st.sidebar.markdown("---")
st.sidebar.markdown("### Detection parameters")

short_window = st.sidebar.number_input(
    "Short window (bars)",
    min_value=5, max_value=120, value=SHORT_WINDOW_DEFAULT, step=1,
    help="Fast moving correlation. Smaller = more responsive but noisier.",
)
long_window = st.sidebar.number_input(
    "Long window (bars)",
    min_value=20, max_value=252, value=LONG_WINDOW_DEFAULT, step=5,
    help="Reference / baseline correlation.",
)
baseline_window = BASELINE_WINDOW_DEFAULT  # kept for API compat, no longer surfaced
break_z = st.sidebar.slider(
    "Break threshold (short − long)",
    min_value=-0.80, max_value=-0.05, value=BREAK_THRESHOLD_DEFAULT, step=0.05,
    help=(
        "A break triggers when (short_corr − long_corr) falls below this "
        "absolute value. −0.30 means short correlation is 0.30 lower than "
        "long correlation. More negative = stricter."
    ),
)

run_btn = st.sidebar.button("Run Detection", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Parse and validate pairs
# ---------------------------------------------------------------------------
pairs = parse_pair_list(pairs_text)
if not pairs:
    st.warning("Add at least one valid pair (format: `SPY / QQQ`).")
    st.stop()


# ---------------------------------------------------------------------------
# Run analysis (cached by config)
# ---------------------------------------------------------------------------
cache_key = (
    tuple(pairs),
    start_date.isoformat(), end_date.isoformat(),
    int(short_window), int(long_window), int(baseline_window), float(break_z),
)
if run_btn or st.session_state.get("corr_cache_key") != cache_key:
    progress = st.progress(0.0, text="Fetching pair data…")
    results = []
    for i, (a, b) in enumerate(pairs):
        progress.progress(i / len(pairs), text=f"Analyzing {a} / {b}…")
        res = analyze_pair(
            a, b, start_date.isoformat(), end_date.isoformat(),
            short_window=int(short_window),
            long_window=int(long_window),
            baseline_window=int(baseline_window),
            break_zscore=float(break_z),
        )
        results.append(res)
    progress.empty()
    st.session_state["corr_results"] = results
    st.session_state["corr_cache_key"] = cache_key

results = st.session_state.get("corr_results", [])
if not results:
    st.info("Click **Run Detection** in the sidebar.")
    st.stop()


# ---------------------------------------------------------------------------
# Top-level alert banner
# ---------------------------------------------------------------------------
active_breaks = [r for r in results if r.ok and r.current_break]
if active_breaks:
    names = ", ".join(r.label for r in active_breaks)
    st.error(
        f"⚠️ **{len(active_breaks)} pair(s) currently broken**: {names}\n\n"
        "Short-term correlation is well below the historical baseline. "
        "Often a leading indicator of macro or sector rotation."
    )
elif any(r.ok for r in results):
    st.success("✓ No active correlation breaks — all monitored pairs are within historical norms.")


# ---------------------------------------------------------------------------
# Status cards — one per pair
# ---------------------------------------------------------------------------
section_header("Current Status")

n_cards_per_row = 3
for i in range(0, len(results), n_cards_per_row):
    row = results[i:i + n_cards_per_row]
    cols = st.columns(len(row))
    for col, r in zip(cols, row):
        with col:
            if not r.ok:
                st.markdown(
                    metric_card(
                        title=f"{r.ticker_a} / {r.ticker_b}",
                        value="—",
                        subtitle=f"❌ {r.error[:60]}",
                        border_color=REGIME_COLORS["Uncertain"],
                    ),
                    unsafe_allow_html=True,
                )
                continue
            # Color: red if broken, cyan otherwise.
            color = REGIME_COLORS["High Vol"] if r.current_break else ACCENT_CYAN
            status = "🔴 BROKEN" if r.current_break else "✓ Normal"
            subtitle = (
                f"Δ={r.current_z:+.2f} · long {r.current_long_corr:+.2f} · "
                f"{r.n_break_events} historical breaks"
            )
            st.markdown(
                metric_card(
                    title=f"{r.label} — {status}",
                    value=f"ρ {r.current_short_corr:+.2f}",
                    subtitle=subtitle,
                    border_color=color,
                ),
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Per-pair detail charts
# ---------------------------------------------------------------------------
section_header("Per-Pair Detail")

for r in results:
    if not r.ok:
        continue

    st.markdown(
        f'<div style="font-size:16px; font-weight:600; margin-top:18px;">'
        f'{r.label}</div>',
        unsafe_allow_html=True,
    )

    fig = go.Figure()

    # Short window — bright cyan.
    fig.add_trace(go.Scatter(
        x=r.short_corr.index, y=r.short_corr.values,
        mode="lines", name=f"{int(short_window)}d corr",
        line=dict(color=ACCENT_CYAN, width=1.8),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>short=%{y:.2f}<extra></extra>",
    ))
    # Long window — muted dashed.
    fig.add_trace(go.Scatter(
        x=r.long_corr.index, y=r.long_corr.values,
        mode="lines", name=f"{int(long_window)}d corr",
        line=dict(color=TEXT_MUTED, width=1.4, dash="dot"),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>long=%{y:.2f}<extra></extra>",
    ))
    # Zero reference.
    fig.add_hline(y=0, line_dash="solid", line_color=BORDER_SUBTLE)

    # Shade the break regions.
    breaks = break_episodes(r.in_break, r.short_corr.index)
    shapes = []
    for ep in breaks:
        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=ep["start"], x1=ep["end"], y0=0, y1=1,
            fillcolor=hex_to_rgba(REGIME_COLORS["High Vol"], 0.18),
            line=dict(width=0), layer="below",
        ))

    layout = get_plotly_layout()
    layout.update({
        "height": 280,
        "shapes": shapes,
        "yaxis": {
            **layout["yaxis"],
            "title": "Rolling correlation", "range": [-1.05, 1.05],
        },
        "xaxis": {**layout["xaxis"], "title": ""},
        "margin": {"l": 50, "r": 30, "t": 10, "b": 40},
    })
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

    # Episodes table.
    if breaks:
        with st.expander(f"Historical break episodes ({len(breaks)})"):
            ep_df = pd.DataFrame([
                {
                    "Start": ep["start"].date().isoformat(),
                    "End":   ep["end"].date().isoformat(),
                    "Bars":  ep["duration_bars"],
                }
                for ep in breaks
            ])
            st.dataframe(ep_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Alert log — recent breaks across all pairs, sorted by date
# ---------------------------------------------------------------------------
section_header("Recent Break Events")

alerts: list[dict] = []
for r in results:
    if not r.ok:
        continue
    for ep in break_episodes(r.in_break, r.short_corr.index):
        alerts.append({
            "Pair": r.label,
            "Start": ep["start"].date().isoformat(),
            "End":   ep["end"].date().isoformat(),
            "Duration (bars)": ep["duration_bars"],
        })

if alerts:
    alert_df = pd.DataFrame(alerts).sort_values("Start", ascending=False)
    st.dataframe(alert_df, use_container_width=True, hide_index=True)
else:
    st.info("No historical breaks in the loaded window.")

st.caption(
    f"Detection: short={int(short_window)}d, long={int(long_window)}d, "
    f"threshold (short − long) < {break_z:.2f}. "
    "A correlation break often precedes regime changes or sector rotations. "
    "Use the live status to size positions or hedge before the move propagates."
)
