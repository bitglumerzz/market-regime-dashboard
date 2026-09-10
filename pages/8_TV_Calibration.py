"""TradingView Calibration — tune the Elliott detector against TV ground truth.

Workflow:
1. User uses Claude Code (with tradesdontlie/tradingview-mcp) to harvest
   wave labels from a community Elliott Wave Pine indicator on their charts,
   saving them as CSVs under `tv_ground_truth/<TICKER>_<TF>.csv`.
   (See `tv_ground_truth/README.md` for the exact prompt.)

2. Open this page. It lists the loaded files, runs a threshold sweep for
   each, and shows per-file F1 + per-TF aggregate.

3. Click "Save calibration" to write `elliott_calibration.json`. Next time
   any page launches, multi_tf.py picks up these thresholds automatically.

4. Re-run the Elliott Waves page — the swings now match TV's labeling
   far more closely.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from calibrate_from_tv import (
    CALIBRATION_FILE,
    DEFAULT_MATCH_TOLERANCE,
    DEFAULT_THRESHOLD_GRID,
    calibrate_all,
    calibrate_one,
    load_all_ground_truth,
    load_calibration,
    load_ground_truth,
    save_calibration,
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
from zigzag import detect_zigzag


GT_DIR = Path("tv_ground_truth")


apply_theme()

st.markdown(
    '<div style="display:flex; align-items:baseline; gap:14px;">'
    '<div style="font-size:24px; font-weight:700; letter-spacing:-0.02em;">'
    '◉ TradingView Calibration</div>'
    '<div style="color:#8B949E; font-size:13px;">Tune our ZigZag thresholds '
    'against TradingView Pine-indicator labels</div>'
    "</div>",
    unsafe_allow_html=True,
)
st.write("")


# ---------------------------------------------------------------------------
# Current calibration status
# ---------------------------------------------------------------------------
section_header("Current Calibration")

current = load_calibration()
if current:
    cols = st.columns(len(current))
    for col, (tf, thr) in zip(cols, current.items()):
        with col:
            st.markdown(
                metric_card(
                    title=f"{tf} threshold",
                    value=f"{thr * 100:.2f}%",
                    subtitle="from elliott_calibration.json",
                    border_color=ACCENT_CYAN,
                ),
                unsafe_allow_html=True,
            )
    st.caption(
        f"These values were loaded from `{CALIBRATION_FILE}` at app startup "
        "and are now used as defaults across the Elliott Waves page."
    )
else:
    st.info(
        "No calibration on disk yet. Built-in defaults are in use "
        "(1d=12%, 4h=6%, 15m=2.5%). Drop ground-truth CSVs in "
        "`tv_ground_truth/` and run the sweep below."
    )


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------
section_header("Ground Truth Files")

uploaded = st.file_uploader(
    "Upload TV ground-truth CSV(s) — header: date,close,wave_label",
    type="csv", accept_multiple_files=True,
)
if uploaded:
    GT_DIR.mkdir(parents=True, exist_ok=True)
    for f in uploaded:
        target = GT_DIR / f.name
        target.write_bytes(f.read())
    st.success(f"Saved {len(uploaded)} file(s) to `tv_ground_truth/`.")

# List what's currently on disk.
gts = load_all_ground_truth(GT_DIR)
if not gts:
    st.warning(
        "No ground-truth files found in `tv_ground_truth/`. "
        "Read `tv_ground_truth/README.md` for the Claude Code prompt to "
        "harvest labels from your TradingView charts."
    )
    st.stop()

gt_df = pd.DataFrame([
    {
        "File": f"{g.ticker}_{g.timeframe}.csv",
        "Ticker": g.ticker,
        "Timeframe": g.timeframe,
        "Bars": len(g.prices),
        "Labels": g.n_labels,
        "First date": g.prices.index.min().date().isoformat(),
        "Last date":  g.prices.index.max().date().isoformat(),
    }
    for g in gts
])
st.dataframe(gt_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Run sweep
# ---------------------------------------------------------------------------
section_header("Calibration Sweep")

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    tolerance = st.number_input(
        "Match tolerance (bars)",
        min_value=0, max_value=10, value=DEFAULT_MATCH_TOLERANCE,
        help="How many bars apart can our swing be from a TV label and still 'match'?",
    )
with c2:
    grid_min = st.number_input("Grid min %", min_value=0.5, max_value=10.0,
                                 value=1.0, step=0.5)
with c3:
    grid_max = st.number_input("Grid max %", min_value=5.0, max_value=30.0,
                                 value=22.0, step=1.0)
grid_step_n = st.slider("Grid step count", min_value=8, max_value=40, value=14)

# Build the threshold grid.
import numpy as np
threshold_grid = tuple(
    round(x / 100.0, 4)
    for x in np.linspace(grid_min, grid_max, grid_step_n)
)

run_btn = st.button("Run sweep", type="primary")

if run_btn or st.session_state.get("calib_results") is None:
    per_file = []
    progress = st.progress(0.0, text="Calibrating…")
    for i, g in enumerate(gts):
        progress.progress(i / max(1, len(gts)),
                          text=f"Sweeping {g.ticker}_{g.timeframe}…")
        per_file.append(
            calibrate_one(g, grid=threshold_grid, tolerance_bars=int(tolerance))
        )
    progress.empty()
    aggregated = calibrate_all(gts, grid=threshold_grid,
                                tolerance_bars=int(tolerance))
    st.session_state["calib_per_file"] = per_file
    st.session_state["calib_aggregated"] = aggregated
    st.session_state["calib_results"] = True

per_file = st.session_state.get("calib_per_file", [])
aggregated = st.session_state.get("calib_aggregated", {})


# ---------------------------------------------------------------------------
# Per-file results
# ---------------------------------------------------------------------------
section_header("Per-File Results")

rows = []
for r in per_file:
    rows.append({
        "Ticker": r.ticker,
        "TF": r.timeframe,
        "Best threshold (%)": f"{r.best_threshold_pct * 100:.2f}",
        "Best F1": f"{r.best_f1:.2f}",
        "Grid size": len(r.grid_results),
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# F1-vs-threshold curve for each file.
if per_file:
    fig = go.Figure()
    for r in per_file:
        x = [g["threshold_pct"] * 100 for g in r.grid_results]
        y = [g["f1"] for g in r.grid_results]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers",
            name=f"{r.ticker} {r.timeframe}",
            hovertemplate=("<b>%{fullData.name}</b><br>"
                           "threshold %{x:.2f}%<br>F1 %{y:.2f}<extra></extra>"),
        ))
    layout = get_plotly_layout()
    layout.update({
        "height": 380,
        "xaxis": {**layout["xaxis"], "title": "ZigZag threshold (%)"},
        "yaxis": {**layout["yaxis"], "title": "F1 vs TV labels",
                  "range": [0, 1]},
        "showlegend": True,
    })
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Each line = one ground-truth file. Peaks show the threshold that "
        "best agrees with TradingView's Pine-indicator labeling for that "
        "ticker/timeframe."
    )


# ---------------------------------------------------------------------------
# Per-TF aggregate
# ---------------------------------------------------------------------------
section_header("Aggregate per Timeframe (F1-weighted)")

if aggregated:
    cols = st.columns(len(aggregated))
    for col, (tf, r) in zip(cols, aggregated.items()):
        with col:
            color = (
                REGIME_COLORS["Low Vol"] if r.best_f1 >= 0.6 else
                REGIME_COLORS["High Vol"] if r.best_f1 <= 0.3 else
                ACCENT_CYAN
            )
            st.markdown(
                metric_card(
                    title=f"{tf} aggregate",
                    value=f"{r.best_threshold_pct * 100:.2f}%",
                    subtitle=f"avg F1 {r.best_f1:.2f}",
                    border_color=color,
                ),
                unsafe_allow_html=True,
            )
else:
    st.info("Run the sweep above to get aggregate values.")


# ---------------------------------------------------------------------------
# Save & apply
# ---------------------------------------------------------------------------
section_header("Save & Apply")

if not aggregated:
    st.info("Nothing to save yet.")
else:
    if st.button("💾 Save calibration", type="primary"):
        save_calibration(aggregated, per_file=per_file, path=CALIBRATION_FILE)
        st.success(
            f"Saved to `{CALIBRATION_FILE}`. **Restart the container** "
            "(`docker compose restart`) so `multi_tf.py` re-imports the file and "
            "picks up the new thresholds. After that, every other page will "
            "use these calibrated values as defaults."
        )

    # Side-by-side comparison: built-in vs calibrated.
    cmp_df = pd.DataFrame([
        {
            "Timeframe": tf,
            "Built-in (%)": f"{({'1d': 12.0, '4h': 6.0, '15m': 2.5}.get(tf, '—'))}",
            "Calibrated (%)": f"{aggregated[tf].best_threshold_pct * 100:.2f}",
            "Avg F1 vs TV": f"{aggregated[tf].best_f1:.2f}",
        }
        for tf in aggregated
    ])
    st.markdown("**Comparison: built-in defaults vs TV-calibrated**")
    st.dataframe(cmp_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
with st.expander("How do I produce these CSVs?"):
    readme_path = GT_DIR / "README.md"
    if readme_path.exists():
        st.markdown(readme_path.read_text())
    else:
        st.markdown(
            "See `tv_ground_truth/README.md` in the project for the "
            "exact Claude Code prompt to give to `tradesdontlie/tradingview-mcp`."
        )
