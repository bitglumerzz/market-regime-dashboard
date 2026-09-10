"""Sentiment Analysis — morning brief on a watchlist of tickers.

For each ticker:
  - Fetch recent headlines via yfinance.news (free, no API key)
  - Score each headline with VADER (compound, [-1, +1])
  - Aggregate into a per-ticker sentiment label
  - Show top positive / negative headlines

Manual paste mode is provided as a fallback when yfinance.news has no
data for a symbol (often the case for non-US tickers).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from design_system import (
    ACCENT_CYAN,
    REGIME_COLORS,
    TEXT_MUTED,
    TEXT_PRIMARY,
    apply_theme,
    get_plotly_layout,
    hex_to_rgba,
    metric_card,
    section_header,
)
from sentiment import (
    NEGATIVE_THRESHOLD,
    POSITIVE_THRESHOLD,
    analyze_pasted,
    analyze_ticker,
    get_analyzer,
)


apply_theme()

st.markdown(
    '<div style="display:flex; align-items:baseline; gap:14px;">'
    '<div style="font-size:24px; font-weight:700; letter-spacing:-0.02em;">'
    '◉ Sentiment Analysis</div>'
    '<div style="color:#8B949E; font-size:13px;">VADER-scored news '
    'sentiment per ticker — your morning brief</div>'
    "</div>",
    unsafe_allow_html=True,
)
st.write("")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Watchlist")

default_tickers = "SPY, QQQ, AAPL, NVDA, TSLA, BTC-USD"
tickers_text = st.sidebar.text_area(
    "Tickers (comma-separated)",
    value=default_tickers,
    height=80,
    help="One ticker per entry. Works best on US stocks/ETFs.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Manual paste fallback")
st.sidebar.caption(
    "When yfinance has no news for a ticker, paste headlines here "
    "(one per line). Will be scored as ticker 'MANUAL'."
)
paste_ticker = st.sidebar.text_input("Paste-as-ticker", value="MANUAL")
pasted = st.sidebar.text_area(
    "Headlines (one per line)",
    value="", height=100, placeholder="Apple's Q4 earnings beat expectations\n…",
)

run_btn = st.sidebar.button(
    "Run Sentiment Scan", type="primary", use_container_width=True
)


# ---------------------------------------------------------------------------
# Parse tickers and run
# ---------------------------------------------------------------------------
tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
if not tickers and not pasted.strip():
    st.warning("Add at least one ticker or paste headlines.")
    st.stop()

cache_key = (tuple(tickers), pasted.strip(), paste_ticker.strip().upper())
if run_btn or st.session_state.get("sent_cache_key") != cache_key:
    try:
        analyzer = get_analyzer()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    progress = st.progress(0.0, text="Fetching news…")
    results = []
    for i, tk in enumerate(tickers):
        progress.progress(i / max(1, len(tickers)), text=f"{tk}…")
        results.append(analyze_ticker(tk, analyzer=analyzer))
    if pasted.strip():
        results.append(analyze_pasted(paste_ticker, pasted, analyzer=analyzer))
    progress.empty()
    st.session_state["sent_results"] = results
    st.session_state["sent_cache_key"] = cache_key

results = st.session_state.get("sent_results", [])
if not results:
    st.info("Click **Run Sentiment Scan** in the sidebar.")
    st.stop()


# ---------------------------------------------------------------------------
# Aggregate bar chart at the top
# ---------------------------------------------------------------------------
section_header("Aggregate Sentiment")

good = [r for r in results if r.ok]
if good:
    sorted_results = sorted(good, key=lambda r: r.mean_score, reverse=True)
    labels = [r.ticker for r in sorted_results]
    scores = [r.mean_score for r in sorted_results]
    colors = [
        REGIME_COLORS["Low Vol"] if s >= POSITIVE_THRESHOLD else
        REGIME_COLORS["High Vol"] if s <= NEGATIVE_THRESHOLD else
        TEXT_MUTED
        for s in scores
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=scores,
        marker=dict(color=colors),
        text=[f"{s:+.2f}" for s in scores], textposition="outside",
        hovertemplate="<b>%{x}</b><br>mean compound: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=POSITIVE_THRESHOLD, line_dash="dot",
                   line_color=hex_to_rgba(REGIME_COLORS["Low Vol"], 0.5),
                   annotation_text="bullish ≥ 0.05",
                   annotation_position="right",
                   annotation_font_color=REGIME_COLORS["Low Vol"])
    fig.add_hline(y=NEGATIVE_THRESHOLD, line_dash="dot",
                   line_color=hex_to_rgba(REGIME_COLORS["High Vol"], 0.5),
                   annotation_text="bearish ≤ −0.05",
                   annotation_position="right",
                   annotation_font_color=REGIME_COLORS["High Vol"])

    layout = get_plotly_layout()
    layout.update({
        "height": 280,
        "yaxis": {**layout["yaxis"], "title": "Mean compound score",
                  "range": [-1.0, 1.0]},
        "xaxis": {**layout["xaxis"], "title": ""},
        "showlegend": False,
    })
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Per-ticker cards
# ---------------------------------------------------------------------------
section_header("Per-Ticker Detail")

n_cards_per_row = 3
for i in range(0, len(results), n_cards_per_row):
    row = results[i:i + n_cards_per_row]
    cols = st.columns(len(row))
    for col, r in zip(cols, row):
        with col:
            if not r.ok:
                st.markdown(
                    metric_card(
                        title=f"{r.ticker} — no data",
                        value="—",
                        subtitle=r.error[:80] if r.error else "No headlines.",
                        border_color=REGIME_COLORS["Uncertain"],
                    ),
                    unsafe_allow_html=True,
                )
                continue
            color = (
                REGIME_COLORS["Low Vol"] if r.label == "bullish" else
                REGIME_COLORS["High Vol"] if r.label == "bearish" else
                TEXT_MUTED
            )
            emoji = "🟢" if r.label == "bullish" else "🔴" if r.label == "bearish" else "⚪️"
            st.markdown(
                metric_card(
                    title=f"{r.ticker} — {emoji} {r.label}",
                    value=f"{r.mean_score:+.2f}",
                    subtitle=(
                        f"{r.n} headlines · {r.n_pos}+ / {r.n_neu}⚪️ / {r.n_neg}−"
                    ),
                    border_color=color,
                ),
                unsafe_allow_html=True,
            )

# Detailed expanders per ticker.
for r in results:
    if not r.ok:
        continue
    with st.expander(f"{r.ticker} — top headlines ({r.n})"):
        df = pd.DataFrame([
            {
                "Score": f"{h.score:+.2f}",
                "Label": h.label,
                "Title": h.title,
                "Publisher": h.publisher,
                "When": (h.datetime.strftime("%Y-%m-%d %H:%M")
                          if h.datetime else "—"),
            }
            for h in sorted(r.headlines, key=lambda h: h.score, reverse=True)
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

st.caption(
    "Sentiment scoring: VADER compound, range [-1, +1]. "
    "Aggregate label = mean of per-headline scores. Thresholds: "
    "≥ +0.05 bullish, ≤ −0.05 bearish. "
    "News source: yfinance.Ticker.news (free, no API key — Yahoo's schema "
    "is unstable so some tickers may return no items)."
)
