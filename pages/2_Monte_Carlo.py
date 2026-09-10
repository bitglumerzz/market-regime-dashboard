"""Monte Carlo Simulator — separate page in the regime dashboard.

Inputs (in order of preference):
1. Re-use the last backtest run in session_state (most common path).
2. Allow CSV upload for arbitrary trade-return series.

Outputs:
- Fan chart: all 1000 bootstrap equity paths overlaid + percentile band.
- KPI cards: median terminal, P5, probability of loss, worst-5% drawdown.
- Histograms: terminal value, max drawdown, Sharpe distributions.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from design_system import (
    ACCENT_CYAN,
    TEXT_MUTED,
    TEXT_PRIMARY,
    apply_theme,
    get_plotly_layout,
    hex_to_rgba,
    metric_card,
    section_header,
)
from monte_carlo import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_N_SIMULATIONS,
    PERCENTILES,
    run_monte_carlo,
)


apply_theme()

st.markdown(
    '<div style="display:flex; align-items:baseline; gap:14px;">'
    '<div style="font-size:24px; font-weight:700; letter-spacing:-0.02em;">'
    '◉ Monte Carlo Simulator</div>'
    '<div style="color:#8B949E; font-size:13px;">Bootstrap a backtest 1000× '
    'to see the distribution of outcomes</div>'
    "</div>",
    unsafe_allow_html=True,
)
st.write("")

# ---------------------------------------------------------------------------
# Input selection
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Monte Carlo configuration")

source_choice = st.sidebar.radio(
    "Returns source",
    options=["Last backtest", "Upload CSV"],
    index=0,
)

n_sims = st.sidebar.number_input(
    "Number of simulations",
    min_value=100, max_value=10_000,
    value=DEFAULT_N_SIMULATIONS, step=100,
)
mode_choice = st.sidebar.radio(
    "Resampling mode",
    options=["block", "iid"],
    index=0,
    help=(
        "block = preserve serial dependence (volatility clustering); "
        "iid = sample bars independently. Use 'block' for daily-bar returns."
    ),
)
block_size = st.sidebar.number_input(
    "Block size (bars)",
    min_value=1, max_value=50, value=DEFAULT_BLOCK_SIZE, step=1,
    disabled=(mode_choice != "block"),
)
seed_in = st.sidebar.number_input(
    "RNG seed",
    min_value=0, max_value=10_000, value=0, step=1,
    help="0 for reproducible runs.",
)

threshold_in = st.sidebar.slider(
    "Loss-probability threshold",
    min_value=0.5, max_value=1.0, value=0.9, step=0.05,
    help=(
        "Reports P(terminal equity < threshold × starting capital). "
        "E.g. 0.9 = probability you end down >10%."
    ),
)


# ---------------------------------------------------------------------------
# Acquire return series
# ---------------------------------------------------------------------------
returns_series: pd.Series | None = None
label: str = ""

if source_choice == "Last backtest":
    if "result" not in st.session_state:
        st.warning(
            "No backtest in session yet. Switch to the **Regime Dashboard** page, "
            "run analysis, then come back here."
        )
        st.stop()
    result = st.session_state["result"]
    df_full: pd.DataFrame = result["data"]

    # Re-derive the strategy's lagged returns the same way the main page does.
    # We can't pickle BacktestResult cleanly across pages, so reconstruct.
    from backtest import run_backtest

    # Pull long_regimes / cost / sizing from session_state if we stored them;
    # otherwise use a sensible default.
    long_regimes = st.session_state.get(
        "long_regimes", ["Low Vol", "Medium-Low Vol", "Medium Vol"]
    )
    cost_bps = st.session_state.get("cost_bps", 5.0)
    slippage_bps = st.session_state.get("slippage_bps", 1.0)
    conf_gate = st.session_state.get("confidence_gate", 0.0)
    sizing = st.session_state.get("sizing", "binary")

    strat, _ = run_backtest(
        df_full,
        long_regimes=long_regimes,
        cost_bps=cost_bps, slippage_bps=slippage_bps,
        confidence_threshold=conf_gate, sizing=sizing,
    )
    returns_series = strat.daily_return.copy()
    label = (
        f"{st.session_state.get('ticker', 'Strategy')} · "
        f"{len(returns_series)} bars · "
        f"Sharpe {strat.sharpe:.2f} · DD {strat.max_drawdown:.1%}"
    )

elif source_choice == "Upload CSV":
    uploaded = st.sidebar.file_uploader(
        "CSV of log returns (1 column) or full strategy output",
        type=["csv"],
    )
    if uploaded is None:
        st.info(
            "Upload a CSV file. Either a single column of per-bar log returns, "
            "or a CSV with a 'strat_log_return' / 'log_return' column."
        )
        st.stop()
    df_csv = pd.read_csv(uploaded)
    cand_cols = [
        c for c in df_csv.columns
        if c.lower() in ("strat_log_return", "log_return", "return", "returns")
    ]
    if cand_cols:
        returns_series = df_csv[cand_cols[0]].astype(float)
        col_used = cand_cols[0]
    elif df_csv.shape[1] == 1:
        returns_series = df_csv.iloc[:, 0].astype(float)
        col_used = df_csv.columns[0]
    else:
        st.error(
            f"Could not find a returns column in {list(df_csv.columns)}. "
            "Add a column named 'log_return' or 'strat_log_return'."
        )
        st.stop()
    returns_series = returns_series.dropna()
    label = f"Uploaded · {len(returns_series)} bars · column '{col_used}'"

if returns_series is None or len(returns_series) < 30:
    st.error("Need at least 30 valid return bars to bootstrap.")
    st.stop()


# ---------------------------------------------------------------------------
# Header strip
# ---------------------------------------------------------------------------
st.markdown(
    f'<div style="color:#8B949E; font-size:13px;">Source: {label}</div>',
    unsafe_allow_html=True,
)
st.write("")

run_btn = st.sidebar.button("Run Simulation", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Simulate (cached by config)
# ---------------------------------------------------------------------------
cache_key = (
    label, int(n_sims), mode_choice, int(block_size), int(seed_in),
    float(returns_series.sum()), int(len(returns_series)),
)
if run_btn or st.session_state.get("mc_cache_key") != cache_key:
    with st.spinner(f"Running {n_sims} simulations…"):
        mc_result = run_monte_carlo(
            returns_series,
            n_simulations=int(n_sims),
            mode=mode_choice,  # type: ignore[arg-type]
            block_size=int(block_size),
            seed=int(seed_in),
        )
    st.session_state["mc_result"] = mc_result
    st.session_state["mc_cache_key"] = cache_key

mc_result = st.session_state.get("mc_result")
if mc_result is None:
    st.stop()


# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
section_header("Distribution of Outcomes")

q = mc_result.quantiles()
prob_loss = mc_result.probability_of_loss()
prob_below = mc_result.probability_below(float(threshold_in))

k_cols = st.columns(5)
kpis = [
    ("Median Terminal", f"×{q['final_pct'][50]:.3f}", "Median final equity"),
    ("P5 Terminal",     f"×{q['final_pct'][5]:.3f}",  "Worst 5% paths end here"),
    ("P95 Terminal",    f"×{q['final_pct'][95]:.3f}", "Best 5% paths end here"),
    ("Probability of Loss",
     f"{prob_loss:.1%}", "P(final < starting capital)"),
    (f"P(end < ×{threshold_in:.2f})",
     f"{prob_below:.1%}", "Tail-loss probability"),
]
for col, (title, value, subtitle) in zip(k_cols, kpis):
    with col:
        st.markdown(
            metric_card(title=title, value=value, subtitle=subtitle,
                        border_color=ACCENT_CYAN),
            unsafe_allow_html=True,
        )

k2 = st.columns(5)
kpis2 = [
    ("Median Sharpe", f"{q['sharpe_pct'][50]:+.2f}", "Across all paths"),
    ("P5 Sharpe",     f"{q['sharpe_pct'][5]:+.2f}",  "Worst 5%"),
    ("Median Max DD", f"{q['dd_pct'][50]:.1%}",      "Across all paths"),
    ("P5 Max DD",     f"{q['dd_pct'][5]:.1%}",       "Worst 5% drawdown"),
    ("Simulations",   f"{mc_result.n_simulations:,}",
     f"{mc_result.mode}, block={mc_result.block_size if mc_result.mode=='block' else '-'}"),
]
for col, (title, value, subtitle) in zip(k2, kpis2):
    with col:
        st.markdown(
            metric_card(title=title, value=value, subtitle=subtitle,
                        border_color=ACCENT_CYAN),
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Fan chart
# ---------------------------------------------------------------------------
section_header("Equity Fan Chart (all simulations)")

n_show = min(300, mc_result.n_simulations)  # plot at most 300 thin lines
fig_fan = go.Figure()

# Thin individual paths in the background.
step = max(1, mc_result.n_simulations // n_show)
for i in range(0, mc_result.n_simulations, step):
    fig_fan.add_trace(go.Scatter(
        x=np.arange(mc_result.n_bars + 1),
        y=mc_result.equity_paths[i],
        mode="lines",
        line=dict(color=hex_to_rgba(ACCENT_CYAN, 0.05), width=0.6),
        hoverinfo="skip",
        showlegend=False,
    ))

# Percentile envelope on top.
bands = mc_result.equity_band((5, 25, 50, 75, 95))
x = np.arange(mc_result.n_bars + 1)
fig_fan.add_trace(go.Scatter(
    x=x, y=bands[0], mode="lines",
    line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
))
fig_fan.add_trace(go.Scatter(
    x=x, y=bands[4], mode="lines",
    line=dict(color="rgba(0,0,0,0)"),
    fill="tonexty", fillcolor=hex_to_rgba(ACCENT_CYAN, 0.12),
    name="P5–P95", showlegend=True, hoverinfo="skip",
))
fig_fan.add_trace(go.Scatter(
    x=x, y=bands[1], mode="lines",
    line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
))
fig_fan.add_trace(go.Scatter(
    x=x, y=bands[3], mode="lines",
    line=dict(color="rgba(0,0,0,0)"),
    fill="tonexty", fillcolor=hex_to_rgba(ACCENT_CYAN, 0.22),
    name="P25–P75", showlegend=True, hoverinfo="skip",
))
fig_fan.add_trace(go.Scatter(
    x=x, y=bands[2], mode="lines",
    line=dict(color=ACCENT_CYAN, width=2.2),
    name="Median", hovertemplate="bar %{x}<br>median=%{y:.3f}<extra></extra>",
))
fig_fan.add_hline(
    y=1.0, line_dash="dash", line_color=TEXT_MUTED,
    annotation_text="break-even", annotation_position="bottom right",
    annotation_font_color=TEXT_MUTED,
)
layout_fan = get_plotly_layout()
layout_fan.update({
    "height": 420,
    "xaxis": {**layout_fan["xaxis"], "title": "Bar"},
    "yaxis": {**layout_fan["yaxis"], "title": "Equity (×1.0 start)"},
})
fig_fan.update_layout(**layout_fan)
st.plotly_chart(fig_fan, use_container_width=True)


# ---------------------------------------------------------------------------
# Histograms (3 wide)
# ---------------------------------------------------------------------------
section_header("Outcome Histograms")

def _hist(values: np.ndarray, title: str, fmt: str, x_title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=values, nbinsx=40,
        marker=dict(color=hex_to_rgba(ACCENT_CYAN, 0.6),
                    line=dict(color=ACCENT_CYAN, width=0.5)),
        hovertemplate=f"{x_title}: %{{x:{fmt}}}<br>count: %{{y}}<extra></extra>",
    ))
    median = float(np.median(values))
    fig.add_vline(
        x=median, line_dash="dash", line_color=TEXT_PRIMARY,
        annotation_text=f"median {format(median, fmt)}",
        annotation_position="top left", annotation_font_color=TEXT_PRIMARY,
    )
    layout = get_plotly_layout()
    layout.update({
        "height": 260,
        "title": dict(text=title, font=dict(color=TEXT_PRIMARY, size=13),
                      x=0.02, xanchor="left"),
        "xaxis": {**layout["xaxis"], "title": x_title},
        "yaxis": {**layout["yaxis"], "title": "Count"},
        "showlegend": False,
    })
    fig.update_layout(**layout)
    return fig

h_cols = st.columns(3)
with h_cols[0]:
    st.plotly_chart(
        _hist(mc_result.final_values, "Final Equity Distribution",
              ".3f", "Final equity (×1.0 start)"),
        use_container_width=True,
    )
with h_cols[1]:
    st.plotly_chart(
        _hist(mc_result.drawdowns * 100, "Max Drawdown Distribution",
              ".1f", "Max drawdown (%)"),
        use_container_width=True,
    )
with h_cols[2]:
    st.plotly_chart(
        _hist(mc_result.sharpe_ratios, "Sharpe Distribution",
              ".2f", "Annualized Sharpe"),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Methodology note
# ---------------------------------------------------------------------------
st.caption(
    f"{mc_result.n_simulations:,} simulations · resampling mode: **{mc_result.mode}**"
    + (f" (block size {mc_result.block_size})" if mc_result.mode == "block" else "")
    + ". "
    "If the median fan is comfortably above 1.0 and the P5 envelope stays above "
    "your loss tolerance, the strategy survives most realistic re-orderings of "
    "its trades. If a wide gap opens between P5 and P95 — outcomes are heavily "
    "regime-dependent and the headline backtest is a small sample."
)
