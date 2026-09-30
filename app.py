"""Streamlit entry point — Market Regime Detection Dashboard.

Run: `streamlit run app.py`

Pipeline:
    yfinance / Stooq OHLCV  →  features  →  StandardScaler  →  GaussianHMM (BIC/AIC)
                                                          ↓
                                                 forward_filter (causal!)
                                                          ↓
                                                 label + stability filter
                                                          ↓
                                             charts + backtest + CSV export
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import StandardScaler

from backtest import run_backtest
from data_loader import load_data
from multi_asset import DEFAULT_TICKERS, run_multi_asset
from sensitivity import (
    DEFAULT_CONFIDENCE_GRID,
    DEFAULT_N_GRID,
    cells_to_matrix,
    robustness_score,
    run_sensitivity_grid,
)
from stress_tests import CRISIS_WINDOWS, run_stress_tests
from design_system import (
    ACCENT_CYAN,
    BG_CARD,
    BORDER_SUBTLE,
    CONFIDENCE_FILL_OPACITY,
    CONFIDENCE_THRESHOLD,
    REGIME_BAND_OPACITY,
    REGIME_COLORS,
    TEXT_MUTED,
    TEXT_PRIMARY,
    apply_theme,
    get_plotly_layout,
    get_regime_color,
    hex_to_rgba,
    metric_card,
    regime_badge,
    section_header,
)
from feature_engineering import (
    HMM_FEATURE_COLS,
    LOG_RETURN_COL,
    REALIZED_VOL_COL,
    VOLUME_RATIO_COL,
    compute_features,
    feature_matrix,
)
from hmm_model import (
    DEFAULT_MIN_TRAIN_SIZE,
    DEFAULT_REFIT_PERIOD,
    forward_filter,
    train_best_hmm,
    verify_no_lookahead,
    walk_forward_classify,
)
from regime_labeler import (
    UNCERTAIN_LABEL,
    apply_stability_filter,
    label_regimes,
)


# ---------------------------------------------------------------------------
# Page config + theme (must be the very first Streamlit call)
# ---------------------------------------------------------------------------
apply_theme()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ANNUALIZATION_FACTOR: int = 252
MIN_ROWS_AFTER_FEATURES: int = 60
MAIN_CHART_HEIGHT: int = 550
CONFIDENCE_CHART_HEIGHT: int = 200
EQUITY_CHART_HEIGHT: int = 320
TRANSITION_HEATMAP_HEIGHT: int = 360
DEFAULT_LOOKBACK_YEARS: int = 3
SYNTHETIC_BAR_COUNT: int = 250

# Analysis modes. The values are stored in session results, so keep them
# stable; the sidebar shows the longer, explicit labels.
MODE_IN_SAMPLE: str = "In-sample"
MODE_WALK_FORWARD: str = "Walk-forward (OOS)"
MODE_LABELS: dict[str, str] = {
    MODE_IN_SAMPLE: "In-sample (full-history fit, not OOS)",
    MODE_WALK_FORWARD: "Walk-forward (OOS)",
}

# Canonical regime label menu (all the labels label_regimes() can produce).
ALL_REGIME_LABELS: tuple[str, ...] = (
    "Low Vol",
    "Medium-Low Vol",
    "Medium Vol",
    "Medium-High Vol",
    "High Vol",
    UNCERTAIN_LABEL,
)
DEFAULT_LONG_REGIMES: tuple[str, ...] = ("Low Vol", "Medium-Low Vol", "Medium Vol")


# ---------------------------------------------------------------------------
# Startup verification — runs once, proves the forward filter is causal
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _startup_verification() -> tuple[bool, str]:
    rng = np.random.default_rng(0)
    half = SYNTHETIC_BAR_COUNT // 2
    a = rng.normal(0.0, 0.4, size=(half, 3))
    b = rng.normal(1.5, 0.4, size=(SYNTHETIC_BAR_COUNT - half, 3))
    synthetic = np.vstack([a, b])
    try:
        model, _ = train_best_hmm(synthetic, n_range=(2, 3))
        verify_no_lookahead(model, synthetic)
        return True, "Forward-algorithm posteriors are causal (verified on synthetic data)."
    except AssertionError as exc:
        return False, f"LOOK-AHEAD BIAS DETECTED: {exc}"
    except Exception as exc:
        return False, f"Verification failed to run: {exc}"


_verif_ok, _verif_msg = _startup_verification()
if _verif_ok:
    st.sidebar.success(f"✓ No look-ahead bias detected\n\n{_verif_msg}")
else:
    st.sidebar.error(_verif_msg)
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Configuration")

ticker = st.sidebar.text_input("Ticker", value="SPY").strip().upper()

today = dt.date.today()
start_default = today - dt.timedelta(days=365 * DEFAULT_LOOKBACK_YEARS)
start_date = st.sidebar.date_input("Start Date", value=start_default)
end_date = st.sidebar.date_input("End Date", value=today)

regime_choice = st.sidebar.selectbox(
    "Regimes Override",
    options=["Auto", 3, 4, 5, 6, 7],
    index=0,
    help="Auto = let the information criterion choose.",
)

criterion_choice = st.sidebar.selectbox(
    "Auto criterion",
    options=["BIC", "AIC"],
    index=0,
    help=(
        "BIC penalizes complexity more strongly (favors fewer regimes). "
        "AIC is more permissive (often picks one or two more)."
    ),
)

mode_choice = st.sidebar.radio(
    "Mode",
    options=[MODE_IN_SAMPLE, MODE_WALK_FORWARD],
    index=0,
    format_func=MODE_LABELS.get,
    help=(
        "In-sample: scaler and HMM fitted on ALL data (future bars included), "
        "then forward-filtered — descriptive only, not an out-of-sample result. "
        "Walk-forward: scaler and HMM refitted on past bars only at each refit "
        "point, then applied to the next bars. This is the honest "
        "out-of-sample mode."
    ),
)
if mode_choice == MODE_WALK_FORWARD:
    min_train_size = st.sidebar.number_input(
        "Walk-forward min train size",
        min_value=100, max_value=2000, value=DEFAULT_MIN_TRAIN_SIZE, step=10,
        help="Bars used for the initial training window before any OOS prediction.",
    )
    refit_period = st.sidebar.number_input(
        "Walk-forward refit period",
        min_value=1, max_value=250, value=DEFAULT_REFIT_PERIOD, step=1,
        help="Refit the HMM every N bars. Smaller = more responsive, slower.",
    )
else:
    min_train_size = DEFAULT_MIN_TRAIN_SIZE
    refit_period = DEFAULT_REFIT_PERIOD

st.sidebar.markdown("---")
st.sidebar.markdown("### Backtest")
long_regimes_selected = st.sidebar.multiselect(
    "Long when regime is in",
    options=list(ALL_REGIME_LABELS),
    default=list(DEFAULT_LONG_REGIMES),
    help=(
        "On any other regime the strategy goes to cash. "
        "Position is lagged by one bar to avoid look-ahead."
    ),
)
sizing_choice = st.sidebar.radio(
    "Position sizing",
    options=["binary", "confidence"],
    index=0,
    horizontal=True,
    help=(
        "binary = 0 or 1; confidence = position proportional to posterior "
        "confidence (in [0, 1])."
    ),
)
confidence_gate = st.sidebar.slider(
    "Confidence gate",
    min_value=0.0, max_value=0.95, value=0.0, step=0.05,
    help="Force cash when lagged confidence is below this threshold.",
)

# Warn against the over-trading trap discovered during testing.
if sizing_choice == "confidence" and confidence_gate >= 0.5:
    st.sidebar.warning(
        "⚠️ **Over-trading trap**\n\n"
        "`sizing=confidence` + `gate ≥ 50%` causes massive over-trading: "
        "every small fluctuation in posterior confidence becomes a trade, "
        "which can eat 10–20% of returns in commissions.\n\n"
        "Safer combos: `binary` with any gate, or `confidence` with `gate=0%`."
    )
cost_bps_in = st.sidebar.number_input(
    "Commission, bps", min_value=0.0, max_value=100.0, value=5.0, step=1.0,
    help="Round-trip commission per leg, in basis points (1 bps = 0.01%).",
)
slippage_bps_in = st.sidebar.number_input(
    "Slippage, bps", min_value=0.0, max_value=100.0, value=1.0, step=1.0,
    help="Estimated slippage per leg, in basis points.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Extra analysis")
do_stress_tests = st.sidebar.checkbox(
    "Run stress tests on crisis windows", value=True,
    help="Backtest restricted to historical crisis windows (COVID, 2022, etc).",
)
do_multi_asset = st.sidebar.checkbox(
    "Compare across multiple assets", value=False,
    help="Run the full pipeline on SPY/QQQ/GLD/TLT/BTC-USD. Slow: 1-2 min.",
)
if do_multi_asset:
    multi_tickers_str = st.sidebar.text_area(
        "Comparison tickers (comma-separated)",
        value=", ".join(DEFAULT_TICKERS),
        height=70,
    )
do_sensitivity = st.sidebar.checkbox(
    "Run sensitivity analysis", value=False,
    help=(
        "Sweep n_components × confidence_threshold and compute Sharpe/return for "
        "each combo. Shows whether the strategy is robust or fragile to "
        "parameter choice. Slow: ~30s. In-sample only."
    ),
)

run = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _contiguous_blocks(labels: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    blocks: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    if labels.empty:
        return blocks
    start_idx = labels.index[0]
    prev_label = labels.iloc[0]
    prev_idx = labels.index[0]
    for ts, lab in labels.items():
        if lab != prev_label:
            blocks.append((start_idx, prev_idx, prev_label))
            start_idx = ts
            prev_label = lab
        prev_idx = ts
    blocks.append((start_idx, prev_idx, prev_label))
    return blocks


def _stability_status(labels: pd.Series, lookback: int = 20) -> str:
    if labels.empty:
        return "—"
    tail = labels.tail(lookback)
    if (tail == UNCERTAIN_LABEL).iloc[-1]:
        return "Uncertain"
    flips = int((tail.shift() != tail).sum() - 1)
    if flips <= 1:
        return "Stable"
    if flips <= 3:
        return "Transitioning"
    return "Uncertain"


def _resolve_n_components(choice: Any) -> tuple[int, int]:
    if choice == "Auto":
        return (3, 7)
    n = int(choice)
    return (n, n)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    ticker: str,
    start: str,
    end: str,
    n_range: tuple[int, int],
    criterion: str,
    mode: str = MODE_IN_SAMPLE,
    min_train: int = DEFAULT_MIN_TRAIN_SIZE,
    refit_every: int = DEFAULT_REFIT_PERIOD,
    progress_cb=None,
) -> dict[str, Any]:
    raw = load_data(ticker, start, end)
    feats = compute_features(raw)
    if len(feats) < MIN_ROWS_AFTER_FEATURES:
        raise ValueError(
            f"Only {len(feats)} feature rows after NaN drop "
            f"(minimum {MIN_ROWS_AFTER_FEATURES}). Widen the date range."
        )

    X = feature_matrix(feats)
    # Full-sample scaling: mean/variance include every bar, future ones too.
    # Fine for the in-sample mode (and its sensitivity sweep), never for OOS.
    Xs = StandardScaler().fit_transform(X)

    if mode == MODE_WALK_FORWARD:
        if len(X) < min_train + refit_every:
            raise ValueError(
                f"Need at least {min_train + refit_every} bars for walk-forward "
                f"with min_train={min_train}, refit_period={refit_every}; "
                f"only have {len(X)}. Widen the date range or shrink the parameters."
            )
        # Pick n_components from the initial training window only — never
        # peeking at future data. The scaler is fitted on that window too.
        init_scaler = StandardScaler().fit(X[:min_train])
        init_model, best_n = train_best_hmm(
            init_scaler.transform(X[:min_train]),
            n_range=n_range, criterion=criterion.lower(),
        )
        # Now do the OOS classification on RAW features: walk_forward_classify
        # refits the scaler on X[:r] at every refit point.
        posteriors_oos, state_orderings, models = walk_forward_classify(
            X,
            n_components=best_n,
            min_train_size=min_train,
            refit_period=refit_every,
            progress_callback=progress_cb,
        )
        # Map sorted columns to canonical labels (col j = j-th vol-sorted regime).
        # We need a state_to_label-style dict. Build canonical labels by n.
        canonical = {0: label_regimes(init_model, best_n)[
            int(np.argsort(init_model.means_[:, 0])[0])
        ]}
        # The cleanest way: re-derive labels from sorted order each refit.
        # All refits share the same n_components, so the label set is the same.
        # Use the initial model just to get the canonical label list:
        init_state_to_label = label_regimes(init_model, best_n)
        init_order = np.argsort(init_model.means_[:, 0])
        col_to_label = [init_state_to_label[int(s)] for s in init_order]

        # Trim the dataset to the OOS region.
        feats_oos = feats.iloc[min_train:].copy()
        # OOS regime per bar = argmax over sorted columns
        col_ids = np.argmax(posteriors_oos, axis=1)
        confidence = np.max(posteriors_oos, axis=1)
        raw_labels = pd.Series(
            [col_to_label[int(c)] for c in col_ids],
            index=feats_oos.index, name="regime_raw",
        )
        smooth_labels = apply_stability_filter(raw_labels)
        smooth_labels.name = "regime"

        out_df = feats_oos.copy()
        out_df["regime"] = smooth_labels
        out_df["confidence"] = confidence

        return {
            "raw_ohlcv": raw,
            "data": out_df,
            "feats_full": feats,         # full features (incl. pre-OOS bars)
            "Xs": Xs,                    # full-sample scaling — only for the in-sample sensitivity sweep
            "model": models[-1],       # most recent fitted model (for transmat heatmap)
            "best_n": best_n,
            "state_to_label": label_regimes(models[-1], best_n),
            "posteriors": posteriors_oos,
            "source": raw.attrs.get("source", "unknown"),
            "mode": MODE_WALK_FORWARD,
            "wf_n_refits": len(models),
            "wf_min_train": min_train,
            "wf_refit_period": refit_every,
        }

    # --- In-sample mode (original behaviour) ---
    # Scaler and HMM parameters are fitted on the whole history, so past
    # posteriors use information from later bars. Only the filtering step is
    # causal; this is a descriptive view, not an out-of-sample result.
    model, best_n = train_best_hmm(Xs, n_range=n_range, criterion=criterion.lower())
    verify_no_lookahead(model, Xs)

    posteriors = forward_filter(model, Xs)
    raw_state_ids = np.argmax(posteriors, axis=1)
    confidence = np.max(posteriors, axis=1)

    state_to_label = label_regimes(model, best_n)
    raw_labels = pd.Series(
        [state_to_label[int(s)] for s in raw_state_ids],
        index=feats.index, name="regime_raw",
    )
    smooth_labels = apply_stability_filter(raw_labels)
    smooth_labels.name = "regime"

    out_df = feats.copy()
    out_df["regime"] = smooth_labels
    out_df["confidence"] = confidence

    return {
        "raw_ohlcv": raw,
        "data": out_df,
        "feats_full": feats,
        "Xs": Xs,
        "model": model,
        "best_n": best_n,
        "state_to_label": state_to_label,
        "posteriors": posteriors,
        "source": raw.attrs.get("source", "unknown"),
        "mode": MODE_IN_SAMPLE,
    }


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div style="display:flex; align-items:baseline; gap:14px;">'
    '<div style="font-size:24px; font-weight:700; letter-spacing:-0.02em;">◉ Market Regime Detection</div>'
    f'<div style="color:#8B949E; font-size:13px;">Forward-algorithm HMM · No look-ahead bias</div>'
    "</div>",
    unsafe_allow_html=True,
)
st.write("")

if not run and "result" not in st.session_state:
    st.info(
        "Configure the ticker and date range in the sidebar, then click "
        "**Run Analysis**. The startup check has already confirmed the "
        "forward filter is causal."
    )
    st.stop()

if run:
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()
    n_range = _resolve_n_components(regime_choice)

    spinner_text = (
        f"Walk-forward fitting {ticker}…" if mode_choice == MODE_WALK_FORWARD
        else f"Analyzing {ticker}…"
    )
    progress_bar = (
        st.sidebar.progress(0.0, text="Walk-forward refits")
        if mode_choice == MODE_WALK_FORWARD else None
    )

    def _progress(p: float) -> None:
        if progress_bar is not None:
            progress_bar.progress(min(max(p, 0.0), 1.0), text=f"Refit {p:.0%}")

    with st.spinner(spinner_text):
        try:
            result = run_pipeline(
                ticker, start_date.isoformat(), end_date.isoformat(),
                n_range, criterion_choice,
                mode=mode_choice,
                min_train=int(min_train_size),
                refit_every=int(refit_period),
                progress_cb=_progress,
            )
            st.session_state["result"] = result
            st.session_state["ticker"] = ticker
            # Cache backtest params so the Monte Carlo page can reproduce them.
            st.session_state["long_regimes"] = list(long_regimes_selected)
            st.session_state["cost_bps"] = float(cost_bps_in)
            st.session_state["slippage_bps"] = float(slippage_bps_in)
            st.session_state["confidence_gate"] = float(confidence_gate)
            st.session_state["sizing"] = sizing_choice
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
            st.stop()
    if progress_bar is not None:
        progress_bar.empty()

result = st.session_state["result"]
ticker = st.session_state["ticker"]
df_full: pd.DataFrame = result["data"]
best_n: int = result["best_n"]
model = result["model"]
state_to_label: dict[int, str] = result["state_to_label"]


# ---------------------------------------------------------------------------
# Top bar
# ---------------------------------------------------------------------------
current_label = str(df_full["regime"].iloc[-1])
current_conf = float(df_full["confidence"].iloc[-1])
stab = _stability_status(df_full["regime"])

c1, c2, c3, c4, c5 = st.columns([1.2, 1.4, 1.0, 1.2, 1.2])

with c1:
    st.markdown(
        f'<div class="top-label">Ticker</div>'
        f'<div class="top-ticker">{ticker}</div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown('<div class="top-label">Current Regime</div>', unsafe_allow_html=True)
    st.markdown(regime_badge(current_label, current_conf), unsafe_allow_html=True)
with c3:
    st.markdown(
        f'<div class="top-label">Confidence</div>'
        f'<div class="top-value">{current_conf:.1%}</div>',
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f'<div class="top-label">Stability</div>'
        f'<div style="font-size:18px; font-weight:600;">{stab}</div>',
        unsafe_allow_html=True,
    )
with c5:
    st.markdown(
        f'<div class="top-label">Regimes Detected</div>'
        f'<div style="font-size:18px; font-weight:600;">{best_n}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main price + regime bands chart
# ---------------------------------------------------------------------------
section_header("Price & Regime History")

unique_labels = list(dict.fromkeys(df_full["regime"].tolist()))
label_to_color = {lab: get_regime_color(lab, i, len(unique_labels))
                  for i, lab in enumerate(unique_labels)}

fig_main = go.Figure()
fig_main.add_trace(
    go.Scatter(
        x=df_full.index, y=df_full["Close"], mode="lines", name="Close",
        line=dict(color="#FFFFFF", width=1.6),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Close: $%{y:.2f}<extra></extra>",
    )
)

blocks = _contiguous_blocks(df_full["regime"])
shapes: list[dict[str, Any]] = []
for start_ts, end_ts, lab in blocks:
    shapes.append(dict(
        type="rect", xref="x", yref="paper",
        x0=start_ts, x1=end_ts, y0=0, y1=1,
        fillcolor=hex_to_rgba(label_to_color[lab], REGIME_BAND_OPACITY),
        line=dict(width=0), layer="below",
    ))

for lab, color in label_to_color.items():
    fig_main.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=12, color=color, symbol="square"),
        name=lab, hoverinfo="skip", showlegend=True,
    ))

layout = get_plotly_layout()
layout.update({
    "height": MAIN_CHART_HEIGHT, "shapes": shapes,
    "yaxis": {**layout["yaxis"], "title": "Price"},
    "xaxis": {**layout["xaxis"], "title": ""},
})
fig_main.update_layout(**layout)
st.plotly_chart(fig_main, use_container_width=True)


# ---------------------------------------------------------------------------
# Regime statistics
# ---------------------------------------------------------------------------
section_header("Regime Statistics")
stats_rows: list[dict[str, Any]] = []
total_bars = len(df_full)
for lab in unique_labels:
    mask = df_full["regime"] == lab
    n_bars = int(mask.sum())
    if n_bars == 0:
        continue
    sub = df_full.loc[mask]
    stats_rows.append({
        "label": lab,
        "ann_return": float(sub[LOG_RETURN_COL].mean() * ANNUALIZATION_FACTOR),
        "mean_vol": float(sub[REALIZED_VOL_COL].mean()),
        "mean_volume_ratio": float(sub[VOLUME_RATIO_COL].mean()),
        "pct_time": n_bars / total_bars,
    })

if stats_rows:
    cols = st.columns(len(stats_rows))
    for col, row in zip(cols, stats_rows):
        with col:
            st.markdown(
                metric_card(
                    title=row["label"],
                    value=f"σ {row['mean_vol']:.3f}",
                    subtitle=(
                        f"Ann. return {row['ann_return']:+.1%} · "
                        f"Vol×{row['mean_volume_ratio']:.2f} · "
                        f"{row['pct_time']:.1%} of time"
                    ),
                    border_color=label_to_color[row["label"]],
                ),
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Transition matrix heatmap
# ---------------------------------------------------------------------------
section_header("Transition Matrix")

# model.transmat_ is in raw-state space (state ids 0..n-1). Reorder it so
# rows/cols follow ascending volatility — that's the order our labels live in.
sorted_states = np.argsort(model.means_[:, 0])
trans_sorted = model.transmat_[sorted_states][:, sorted_states]
sorted_label_list = [state_to_label[int(s)] for s in sorted_states]

# Plotly text labels: percentages.
text_matrix = [[f"{v:.0%}" for v in row] for row in trans_sorted]

fig_trans = go.Figure(
    data=go.Heatmap(
        z=trans_sorted,
        x=sorted_label_list,
        y=sorted_label_list,
        colorscale=[
            [0.0, "rgba(14, 17, 23, 1)"],
            [0.5, hex_to_rgba(ACCENT_CYAN, 0.45)],
            [1.0, ACCENT_CYAN],
        ],
        text=text_matrix,
        texttemplate="%{text}",
        textfont={"color": TEXT_PRIMARY, "size": 13},
        zmin=0, zmax=1,
        colorbar=dict(
            tickformat=".0%",
            title=dict(text="P(j | i)", font=dict(color=TEXT_MUTED)),
            tickfont=dict(color=TEXT_MUTED),
            bgcolor="rgba(0,0,0,0)",
            outlinecolor=BORDER_SUBTLE,
        ),
        hovertemplate="From <b>%{y}</b><br>To <b>%{x}</b><br>P = %{z:.1%}<extra></extra>",
    )
)
layout_t = get_plotly_layout()
layout_t.update({
    "height": TRANSITION_HEATMAP_HEIGHT,
    "yaxis": {**layout_t["yaxis"], "title": "From", "autorange": "reversed"},
    "xaxis": {**layout_t["xaxis"], "title": "To", "side": "bottom"},
    "showlegend": False,
})
fig_trans.update_layout(**layout_t)
st.plotly_chart(fig_trans, use_container_width=True)
st.caption(
    "Each cell is P(regime tomorrow = column | regime today = row). "
    "Diagonal values close to 1 mean a regime is 'sticky' — once entered, "
    "it tends to persist."
)


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
section_header("Backtest — Regime-Gated Long/Cash vs Buy & Hold")

if not long_regimes_selected:
    st.warning("Select at least one regime in the sidebar to run a backtest.")
else:
    strat, bh = run_backtest(
        df_full,
        long_regimes=long_regimes_selected,
        cost_bps=float(cost_bps_in),
        slippage_bps=float(slippage_bps_in),
        confidence_threshold=float(confidence_gate),
        sizing=sizing_choice,  # type: ignore[arg-type]
    )

    # KPI cards — strategy vs B&H side by side. Two rows of 5 cards each.
    row1 = st.columns(5)
    row1_metrics = [
        ("Total Return", f"{strat.total_return:+.1%}", f"B&H: {bh.total_return:+.1%}"),
        ("Ann. Return",  f"{strat.ann_return:+.1%}",   f"B&H: {bh.ann_return:+.1%}"),
        ("Ann. Vol",     f"{strat.ann_vol:.1%}",       f"B&H: {bh.ann_vol:.1%}"),
        ("Sharpe",       f"{strat.sharpe:.2f}",        f"B&H: {bh.sharpe:.2f}"),
        ("Max Drawdown", f"{strat.max_drawdown:.1%}",  f"B&H: {bh.max_drawdown:.1%}"),
    ]
    for col, (title, value, subtitle) in zip(row1, row1_metrics):
        with col:
            st.markdown(
                metric_card(title=title, value=value, subtitle=subtitle, border_color=ACCENT_CYAN),
                unsafe_allow_html=True,
            )

    row2 = st.columns(5)
    row2_metrics = [
        ("Sortino",         f"{strat.sortino:.2f}",        f"B&H: {bh.sortino:.2f}"),
        ("Calmar",          f"{strat.calmar:.2f}",         f"B&H: {bh.calmar:.2f}"),
        ("Win Rate",        f"{strat.win_rate:.1%}",       f"In-position bars"),
        ("Round-Trips",     f"{strat.n_round_trips}",      f"{strat.n_trades} pos. changes"),
        ("Cost Drag",       f"−{strat.total_cost:.2%}",    f"Cumulative paid"),
    ]
    for col, (title, value, subtitle) in zip(row2, row2_metrics):
        with col:
            st.markdown(
                metric_card(title=title, value=value, subtitle=subtitle, border_color=ACCENT_CYAN),
                unsafe_allow_html=True,
            )

    # Equity curve overlay.
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=bh.equity.index, y=bh.equity.values, mode="lines", name="Buy & Hold",
        line=dict(color=TEXT_MUTED, width=1.5, dash="dot"),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>B&H equity: %{y:.3f}<extra></extra>",
    ))
    fig_eq.add_trace(go.Scatter(
        x=strat.equity.index, y=strat.equity.values, mode="lines",
        name="Regime Strategy",
        line=dict(color=ACCENT_CYAN, width=1.8),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Strategy equity: %{y:.3f}<extra></extra>",
    ))
    layout_eq = get_plotly_layout()
    layout_eq.update({
        "height": EQUITY_CHART_HEIGHT,
        "yaxis": {**layout_eq["yaxis"], "title": "Equity (×1.0 start)"},
        "xaxis": {**layout_eq["xaxis"], "title": ""},
    })
    fig_eq.update_layout(**layout_eq)
    st.plotly_chart(fig_eq, use_container_width=True)

    st.caption(
        f"Strategy is long when regime ∈ {{{', '.join(long_regimes_selected)}}} "
        f"and lagged confidence ≥ {confidence_gate:.0%}. "
        f"Sizing: {sizing_choice}. "
        f"Costs: {cost_bps_in:.0f} bps commission + {slippage_bps_in:.0f} bps slippage per leg. "
        f"Position lagged by 1 bar (no look-ahead). "
        f"{strat.n_round_trips} round-trips · {strat.time_in_market:.0%} time in market."
    )


# ---------------------------------------------------------------------------
# Stress tests
# ---------------------------------------------------------------------------
if do_stress_tests and long_regimes_selected:
    section_header("Stress Tests — Historical Crisis Windows")
    stress_rows = run_stress_tests(
        df_full,
        long_regimes=long_regimes_selected,
        cost_bps=float(cost_bps_in),
        slippage_bps=float(slippage_bps_in),
        confidence_threshold=float(confidence_gate),
        sizing=sizing_choice,  # type: ignore[arg-type]
    )

    if not stress_rows:
        st.info(
            "No crisis windows overlap the loaded date range. "
            f"Try widening the start date — earliest window starts "
            f"{CRISIS_WINDOWS[0].start}."
        )
    else:
        # Compact comparison table.
        table_df = pd.DataFrame([
            {
                "Crisis": r["window"].label,
                "Period": f"{r['start']} → {r['end']}",
                "Bars": r["bars"],
                "Strat Total": r["strat_total"],
                "B&H Total": r["bh_total"],
                "Outperf.": r["outperformance"],
                "Strat DD": r["strat_dd"],
                "B&H DD": r["bh_dd"],
                "DD Saved": r["dd_saved"],
                "Time in Mkt": r["time_in_market"],
                "Trades": r["n_trades"],
            }
            for r in stress_rows
        ])
        # Format percentages.
        for col in ("Strat Total", "B&H Total", "Outperf.",
                    "Strat DD", "B&H DD", "DD Saved", "Time in Mkt"):
            table_df[col] = table_df[col].map(lambda x: f"{x:+.1%}" if "DD" not in col and "Time" not in col else f"{x:.1%}")
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        st.caption(
            "Outperformance = strategy total return − B&H total return (positive = strategy won). "
            "DD Saved = strategy max DD − B&H max DD (positive = strategy had smaller drawdown). "
            "Windows with <20 bars in the loaded dataset are skipped."
        )


# ---------------------------------------------------------------------------
# Multi-asset comparison
# ---------------------------------------------------------------------------
if do_multi_asset and long_regimes_selected:
    section_header("Multi-Asset Comparison")

    tickers_for_comparison = [
        t.strip().upper() for t in multi_tickers_str.split(",") if t.strip()
    ]
    cache_key = (
        tuple(tickers_for_comparison),
        start_date.isoformat(), end_date.isoformat(),
        _resolve_n_components(regime_choice),
        criterion_choice.lower(),
        tuple(sorted(long_regimes_selected)),
        float(cost_bps_in), float(slippage_bps_in),
        float(confidence_gate), sizing_choice,
    )
    if st.session_state.get("multi_cache_key") != cache_key:
        ma_progress = st.progress(0.0, text="Running multi-asset comparison…")

        def _ma_progress(p: float, msg: str) -> None:
            ma_progress.progress(min(max(p, 0.0), 1.0), text=msg)

        multi_results = run_multi_asset(
            tickers_for_comparison,
            start_date.isoformat(), end_date.isoformat(),
            n_range=_resolve_n_components(regime_choice),
            criterion=criterion_choice.lower(),
            long_regimes=long_regimes_selected,
            cost_bps=float(cost_bps_in),
            slippage_bps=float(slippage_bps_in),
            confidence_threshold=float(confidence_gate),
            sizing=sizing_choice,
            progress_cb=_ma_progress,
        )
        ma_progress.empty()
        st.session_state["multi_results"] = multi_results
        st.session_state["multi_cache_key"] = cache_key

    multi_results = st.session_state.get("multi_results", [])

    if not multi_results:
        st.warning("No tickers were processed.")
    else:
        # Comparison table.
        rows = []
        for r in multi_results:
            if not r.ok:
                rows.append({
                    "Ticker": r.ticker, "Regimes": "—",
                    "Strat Ret": "—", "B&H Ret": "—", "Outperf.": "—",
                    "Strat Sharpe": "—", "B&H Sharpe": "—",
                    "Strat DD": "—", "B&H DD": "—",
                    "Status": f"❌ {r.error[:40]}",
                })
                continue
            rows.append({
                "Ticker": r.ticker,
                "Regimes": r.best_n,
                "Strat Ret": f"{r.strat.total_return:+.1%}",
                "B&H Ret":  f"{r.bh.total_return:+.1%}",
                "Outperf.": f"{r.strat.total_return - r.bh.total_return:+.1%}",
                "Strat Sharpe": f"{r.strat.sharpe:.2f}",
                "B&H Sharpe":   f"{r.bh.sharpe:.2f}",
                "Strat DD": f"{r.strat.max_drawdown:.1%}",
                "B&H DD":   f"{r.bh.max_drawdown:.1%}",
                "Status": f"✓ {r.source}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Mini equity overlays, one chart per successful asset.
        ok_results = [r for r in multi_results if r.ok]
        if ok_results:
            cols_per_row = 2
            for i in range(0, len(ok_results), cols_per_row):
                row_results = ok_results[i:i + cols_per_row]
                cols = st.columns(len(row_results))
                for col, r in zip(cols, row_results):
                    with col:
                        fig_mini = go.Figure()
                        fig_mini.add_trace(go.Scatter(
                            x=r.bh.equity.index, y=r.bh.equity.values,
                            mode="lines", name=f"{r.ticker} B&H",
                            line=dict(color=TEXT_MUTED, width=1.2, dash="dot"),
                            showlegend=False,
                        ))
                        fig_mini.add_trace(go.Scatter(
                            x=r.strat.equity.index, y=r.strat.equity.values,
                            mode="lines", name=f"{r.ticker} Strat",
                            line=dict(color=ACCENT_CYAN, width=1.5),
                            showlegend=False,
                        ))
                        layout_mini = get_plotly_layout()
                        layout_mini.update({
                            "height": 220,
                            "title": dict(
                                text=f"<b>{r.ticker}</b>  ·  Strat {r.strat.sharpe:.2f}  vs  "
                                     f"B&H {r.bh.sharpe:.2f}",
                                font=dict(color=TEXT_PRIMARY, size=13),
                                x=0.02, xanchor="left",
                            ),
                            "margin": {"l": 40, "r": 20, "t": 35, "b": 30},
                        })
                        fig_mini.update_layout(**layout_mini)
                        st.plotly_chart(fig_mini, use_container_width=True)

        st.caption(
            "Multi-asset runs use the same parameters as the main run, but always "
            "in-sample (walk-forward is too slow for this view). "
            "Use this to spot where the regime framework helps — typically equity "
            "indices and gold; less reliably on crypto or single names."
        )


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------
if do_sensitivity and long_regimes_selected:
    section_header("Sensitivity Analysis — Robustness Across Parameters")

    feats_full = result["feats_full"]
    Xs = result["Xs"]
    sens_cache_key = (
        ticker, start_date.isoformat(), end_date.isoformat(),
        tuple(sorted(long_regimes_selected)),
        float(cost_bps_in), float(slippage_bps_in),
        sizing_choice, DEFAULT_N_GRID, DEFAULT_CONFIDENCE_GRID,
    )
    if st.session_state.get("sens_cache_key") != sens_cache_key:
        sens_progress = st.progress(0.0, text="Sweeping parameter grid…")

        def _sens_progress(p: float, msg: str) -> None:
            sens_progress.progress(min(max(p, 0.0), 1.0), text=msg)

        cells = run_sensitivity_grid(
            feats_full, Xs,
            long_regimes=long_regimes_selected,
            n_grid=DEFAULT_N_GRID,
            confidence_grid=DEFAULT_CONFIDENCE_GRID,
            cost_bps=float(cost_bps_in),
            slippage_bps=float(slippage_bps_in),
            sizing=sizing_choice,
            progress_cb=_sens_progress,
        )
        sens_progress.empty()
        st.session_state["sens_cells"] = cells
        st.session_state["sens_cache_key"] = sens_cache_key

    cells = st.session_state.get("sens_cells", [])
    if not cells:
        st.warning("Sensitivity sweep produced no results.")
    else:
        # Robustness score on top.
        rob = robustness_score(cells, metric="sharpe")
        finite_sharpes = [c.sharpe for c in cells if np.isfinite(c.sharpe)]
        if finite_sharpes:
            median_sharpe = float(np.median(finite_sharpes))
            iqr_sharpe = float(np.percentile(finite_sharpes, 75) - np.percentile(finite_sharpes, 25))
        else:
            median_sharpe = iqr_sharpe = 0.0

        rs_cols = st.columns(3)
        for col, (title, value, subtitle) in zip(rs_cols, [
            ("Median Sharpe (grid)", f"{median_sharpe:.2f}", f"Across {len(cells)} cells"),
            ("Sharpe IQR (grid)",    f"{iqr_sharpe:.2f}",    "Lower = more stable"),
            ("Robustness Score",     f"{rob:.2f}",           "Median / (1 + IQR)"),
        ]):
            with col:
                st.markdown(
                    metric_card(title=title, value=value, subtitle=subtitle,
                                border_color=ACCENT_CYAN),
                    unsafe_allow_html=True,
                )

        n_grid = DEFAULT_N_GRID
        ct_grid = DEFAULT_CONFIDENCE_GRID

        def _heatmap(matrix: np.ndarray, *, title: str, fmt: str, zmid: float | None = None) -> go.Figure:
            text = [[(fmt.format(v) if np.isfinite(v) else "—") for v in row] for row in matrix]
            colorscale = [
                [0.0, "#FF6B6B"],
                [0.5, "rgba(14, 17, 23, 1)"],
                [1.0, ACCENT_CYAN],
            ]
            kwargs = dict(
                z=matrix,
                x=[f"{ct:.0%}" for ct in ct_grid],
                y=[f"n={n}" for n in n_grid],
                colorscale=colorscale,
                text=text, texttemplate="%{text}",
                textfont={"color": TEXT_PRIMARY, "size": 13},
                hovertemplate="n=%{y}<br>conf≥%{x}<br>value=%{z:.3f}<extra></extra>",
                colorbar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    tickfont=dict(color=TEXT_MUTED),
                    outlinecolor=BORDER_SUBTLE,
                ),
            )
            if zmid is not None:
                kwargs["zmid"] = zmid
            fig = go.Figure(data=go.Heatmap(**kwargs))
            layout = get_plotly_layout()
            layout.update({
                "height": 320,
                "title": dict(text=title, font=dict(color=TEXT_PRIMARY, size=13),
                              x=0.02, xanchor="left"),
                "xaxis": {**layout["xaxis"], "title": "Confidence threshold"},
                "yaxis": {**layout["yaxis"], "title": "n_components", "autorange": "reversed"},
                "showlegend": False,
            })
            fig.update_layout(**layout)
            return fig

        sharpe_mat = cells_to_matrix(cells, "sharpe", n_grid, ct_grid)
        ret_mat = cells_to_matrix(cells, "total_return", n_grid, ct_grid)

        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.plotly_chart(
                _heatmap(sharpe_mat, title="Sharpe Ratio", fmt="{:.2f}", zmid=0.0),
                use_container_width=True,
            )
        with chart_cols[1]:
            st.plotly_chart(
                _heatmap(ret_mat * 100.0, title="Total Return (%)", fmt="{:.1f}", zmid=0.0),
                use_container_width=True,
            )

        st.caption(
            "If the heatmaps are largely one color, the strategy is robust — Sharpe and "
            "returns are stable across parameter neighborhoods. If one bright cell sits "
            "in a sea of red, that's overfitting: pick a different cell and performance "
            "collapses. Sensitivity is run in-sample for speed; walk-forward would "
            "multiply runtime by ~20×."
        )


# ---------------------------------------------------------------------------
# Confidence timeline
# ---------------------------------------------------------------------------
section_header("Posterior Confidence")
fig_conf = go.Figure()
fig_conf.add_trace(go.Scatter(
    x=df_full.index, y=df_full["confidence"], mode="lines", name="Confidence",
    line=dict(color=ACCENT_CYAN, width=1.2),
    fill="tozeroy", fillcolor=hex_to_rgba(ACCENT_CYAN, CONFIDENCE_FILL_OPACITY),
    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Confidence: %{y:.1%}<extra></extra>",
))
fig_conf.add_hline(
    y=CONFIDENCE_THRESHOLD, line_dash="dash", line_color="#8B949E",
    annotation_text=f"{CONFIDENCE_THRESHOLD:.0%} threshold",
    annotation_position="top right", annotation_font_color="#8B949E",
)
layout_conf = get_plotly_layout()
layout_conf.update({
    "height": CONFIDENCE_CHART_HEIGHT,
    "yaxis": {**layout_conf["yaxis"], "range": [0, 1.02], "tickformat": ".0%", "title": ""},
    "showlegend": False,
})
fig_conf.update_layout(**layout_conf)
st.plotly_chart(fig_conf, use_container_width=True)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
section_header("Export")

export_df = df_full[["Open", "High", "Low", "Close", "Volume",
                     LOG_RETURN_COL, REALIZED_VOL_COL, "regime", "confidence"]].copy()
if long_regimes_selected:
    # Re-derive the lagged position so it matches what the backtest used.
    signal = df_full["regime"].isin(long_regimes_selected).astype(float)
    export_df["position"] = signal.shift(1).fillna(0.0)

csv_bytes = export_df.to_csv(index=True).encode("utf-8")
filename = (
    f"{ticker}_regimes_{df_full.index.min().date()}_{df_full.index.max().date()}.csv"
)
st.download_button(
    label="📥 Download regime signals as CSV",
    data=csv_bytes,
    file_name=filename,
    mime="text/csv",
    use_container_width=False,
)
st.caption(
    "Columns: OHLCV · log_return · realized_vol · regime · confidence · "
    "position (lagged by 1 bar, only present if backtest regimes were selected)."
)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
mode_info = result.get("mode", MODE_IN_SAMPLE)
extra = ""
if mode_info == MODE_WALK_FORWARD:
    extra = (
        f" · WF: {result.get('wf_n_refits', '?')} refits, "
        f"min_train={result.get('wf_min_train', '?')}, "
        f"refit_every={result.get('wf_refit_period', '?')}"
    )
st.caption(
    f"Bars analyzed: {len(df_full):,} · "
    f"Period: {df_full.index.min().date()} → {df_full.index.max().date()} · "
    f"Data source: {result.get('source', 'unknown')} · "
    f"Criterion: {criterion_choice} · "
    f"Mode: {MODE_LABELS.get(mode_info, mode_info)}{extra} · "
    "Posteriors are causal (forward-algorithm only)."
)
