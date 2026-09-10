"""Portfolio Risk Dashboard — see your book through several lenses.

Inputs:
  - Paste a CSV (`ticker,qty,cost_basis` — last col optional)
  - Or fill the editable table in the UI

Outputs:
  - Total market value, cost value, P&L
  - Allocation bar
  - Correlation heatmap (catch positions that are really the same trade)
  - Per-position current regime (Low/Med/High Vol)
  - Stress test against historical crisis windows
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
from portfolio import (
    Position,
    analyze_portfolio,
    correlation_matrix,
    parse_positions_csv,
    portfolio_total_value,
    stress_test_portfolio,
)


apply_theme()

st.markdown(
    '<div style="display:flex; align-items:baseline; gap:14px;">'
    '<div style="font-size:24px; font-weight:700; letter-spacing:-0.02em;">'
    '◉ Portfolio Risk Dashboard</div>'
    '<div style="color:#8B949E; font-size:13px;">Allocation, correlations, '
    'regimes, and stress tests for your current book</div>'
    "</div>",
    unsafe_allow_html=True,
)
st.write("")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Configuration")

today = dt.date.today()
start_default = today - dt.timedelta(days=365 * 3)
start_date = st.sidebar.date_input("Start date", value=start_default, key="port_start")
end_date = st.sidebar.date_input("End date", value=today, key="port_end")

st.sidebar.markdown("---")
st.sidebar.markdown("### Positions")

mode = st.sidebar.radio(
    "Input method",
    options=["Editable table", "Paste CSV"],
    horizontal=True,
)

DEFAULT_POSITIONS = pd.DataFrame([
    {"ticker": "SPY", "qty": 100, "cost_basis": 480.0},
    {"ticker": "QQQ", "qty": 50,  "cost_basis": 420.0},
    {"ticker": "GLD", "qty": 30,  "cost_basis": 200.0},
    {"ticker": "TLT", "qty": 100, "cost_basis": 95.0},
    {"ticker": "BTC-USD", "qty": 0.5, "cost_basis": 60000.0},
])

if mode == "Editable table":
    positions_df = st.sidebar.data_editor(
        DEFAULT_POSITIONS, num_rows="dynamic", use_container_width=True,
        key="positions_editor",
    )
    csv_text = ""
else:
    csv_text = st.sidebar.text_area(
        "CSV (ticker,qty,cost_basis)",
        value=DEFAULT_POSITIONS.to_csv(index=False),
        height=180,
    )
    positions_df = None

run_btn = st.sidebar.button(
    "Run Analysis", type="primary", use_container_width=True, key="port_run"
)


# ---------------------------------------------------------------------------
# Build positions list
# ---------------------------------------------------------------------------
positions: list[Position] = []
try:
    if mode == "Editable table" and positions_df is not None:
        df = positions_df.dropna(subset=["ticker", "qty"])
        for _, row in df.iterrows():
            cb = row.get("cost_basis")
            try:
                cb_val = float(cb) if pd.notna(cb) else None
            except (TypeError, ValueError):
                cb_val = None
            positions.append(Position(
                ticker=str(row["ticker"]).strip().upper(),
                qty=float(row["qty"]),
                cost_basis=cb_val,
            ))
    else:
        positions = parse_positions_csv(csv_text) if csv_text.strip() else []
except Exception as exc:
    st.error(f"Could not parse positions: {exc}")
    st.stop()

if not positions:
    st.warning("Add at least one position in the sidebar.")
    st.stop()


# ---------------------------------------------------------------------------
# Run analysis (cached)
# ---------------------------------------------------------------------------
cache_key = (
    tuple((p.ticker, p.qty, p.cost_basis) for p in positions),
    start_date.isoformat(), end_date.isoformat(),
)
if run_btn or st.session_state.get("port_cache_key") != cache_key:
    progress = st.progress(0.0, text="Fetching prices…")

    def _cb(p: float, msg: str) -> None:
        progress.progress(min(max(p, 0.0), 1.0), text=msg)

    analyses = analyze_portfolio(
        positions, start_date.isoformat(), end_date.isoformat(),
        progress_cb=_cb,
    )
    progress.empty()
    st.session_state["port_analyses"] = analyses
    st.session_state["port_cache_key"] = cache_key

analyses = st.session_state.get("port_analyses", [])
if not analyses:
    st.info("Click **Run Analysis** in the sidebar.")
    st.stop()


# ---------------------------------------------------------------------------
# Top totals
# ---------------------------------------------------------------------------
totals = portfolio_total_value(analyses)

t_cols = st.columns(4)
top_metrics = [
    ("Market Value", f"${totals['market_value']:,.0f}", "Current sum of all positions"),
    ("Cost Basis",   (f"${totals['cost_value']:,.0f}" if totals['cost_value']
                       else "—"), "Total invested"),
    ("P&L",          (f"${totals['pnl_abs']:+,.0f}" if totals['cost_value']
                       else "—"), "Market value − Cost basis"),
    ("P&L %",        (f"{totals['pnl_pct']:+.1f}%" if totals['cost_value']
                       else "—"), ""),
]
for col, (title, value, subtitle) in zip(t_cols, top_metrics):
    with col:
        color = (
            REGIME_COLORS["Low Vol"] if title == "P&L %" and totals['pnl_pct'] > 0 else
            REGIME_COLORS["High Vol"] if title == "P&L %" and totals['pnl_pct'] < 0 else
            ACCENT_CYAN
        )
        st.markdown(
            metric_card(title=title, value=value, subtitle=subtitle, border_color=color),
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Per-position table + regime badges
# ---------------------------------------------------------------------------
section_header("Positions & Current Regimes")

rows = []
for a in analyses:
    regime_color = REGIME_COLORS.get(a.regime, TEXT_MUTED)
    rows.append({
        "Ticker": a.ticker,
        "Qty": a.qty,
        "Last Price": (f"${a.last_price:,.2f}" if a.ok else "—"),
        "Market Value": (f"${a.market_value:,.0f}" if a.ok else "—"),
        "Weight %": (f"{a.market_value / totals['market_value'] * 100:.1f}%"
                       if a.ok and totals['market_value'] > 0 else "—"),
        "P&L %": (f"{a.pnl_pct * 100:+.1f}%" if a.pnl_pct is not None else "—"),
        "Regime": a.regime,
        "Confidence": (f"{a.regime_confidence:.0%}" if a.regime_confidence else "—"),
        "Status": ("✓" if a.ok else f"❌ {a.error[:40]}"),
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Allocation bar
# ---------------------------------------------------------------------------
section_header("Allocation by Market Value")

ok = [a for a in analyses if a.ok and a.market_value > 0]
if ok and totals['market_value'] > 0:
    sorted_ok = sorted(ok, key=lambda a: a.market_value, reverse=True)
    labels = [a.ticker for a in sorted_ok]
    values = [a.market_value for a in sorted_ok]
    colors = [REGIME_COLORS.get(a.regime, ACCENT_CYAN) for a in sorted_ok]

    fig_alloc = go.Figure()
    fig_alloc.add_trace(go.Bar(
        x=labels, y=values,
        marker=dict(color=colors),
        text=[f"${v:,.0f}" for v in values], textposition="outside",
        hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>",
    ))
    layout = get_plotly_layout()
    layout.update({
        "height": 260,
        "yaxis": {**layout["yaxis"], "title": "Market value ($)"},
        "xaxis": {**layout["xaxis"], "title": ""},
        "showlegend": False,
    })
    fig_alloc.update_layout(**layout)
    st.plotly_chart(fig_alloc, use_container_width=True)


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------
section_header("Correlation Matrix")

corr = correlation_matrix(analyses)
if corr.empty:
    st.info("Need ≥ 2 positions with sufficient history to compute correlations.")
else:
    tickers = list(corr.columns)
    z = corr.values
    text = [[f"{v:+.2f}" if np.isfinite(v) else "—" for v in row] for row in z]
    fig_corr = go.Figure(data=go.Heatmap(
        z=z, x=tickers, y=tickers,
        colorscale=[
            [0.0, REGIME_COLORS["High Vol"]],
            [0.5, "rgba(14, 17, 23, 1)"],
            [1.0, ACCENT_CYAN],
        ],
        zmin=-1, zmax=1, zmid=0,
        text=text, texttemplate="%{text}",
        textfont={"color": TEXT_PRIMARY, "size": 12},
        colorbar=dict(tickfont=dict(color=TEXT_MUTED),
                      outlinecolor=BORDER_SUBTLE, bgcolor="rgba(0,0,0,0)"),
        hovertemplate="ρ(%{y}, %{x}) = %{z:.2f}<extra></extra>",
    ))
    layout_corr = get_plotly_layout()
    layout_corr.update({
        "height": min(80 + 35 * len(tickers), 500),
        "yaxis": {**layout_corr["yaxis"], "title": "", "autorange": "reversed"},
        "xaxis": {**layout_corr["xaxis"], "title": "", "side": "bottom"},
        "showlegend": False,
    })
    fig_corr.update_layout(**layout_corr)
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption(
        "Pairs with ρ > 0.7 are essentially the same trade — concentration risk "
        "even if the tickers look different. Pairs with ρ < −0.5 are natural hedges."
    )


# ---------------------------------------------------------------------------
# Stress test
# ---------------------------------------------------------------------------
section_header("Stress Tests — Historical Crisis Windows")

stress_df = stress_test_portfolio(analyses)
if stress_df.empty:
    st.info("No crisis windows have enough overlapping price history in your "
             "loaded date range. Widen the start date to 2008-09-01 to "
             "capture all 7 crises.")
else:
    # Format for display.
    disp = stress_df.copy()
    disp["Portfolio Return"] = disp["Portfolio Return"].map(lambda x: f"{x:+.1%}")
    disp["Portfolio Max DD"] = disp["Portfolio Max DD"].map(lambda x: f"{x:.1%}")
    disp["Worst Single DD"]  = disp["Worst Single DD"].map(lambda x: f"{x:.1%}")
    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.caption(
        "Each row asks: 'if you held today's allocation through this crisis, "
        "what would the portfolio have done?' Weights are current market-value "
        "weights, returns are weighted-summed daily log returns through the window."
    )
