"""Trading Bot — paper-trading dashboard for the regime strategy.

This page is a CONTROL PANEL on top of the bot skeleton, not an
autonomous bot running 24/7 in this container. To make it live, you'd:
  1. Replace `MockBroker` with a real adapter (Alpaca paper is recommended)
  2. Schedule the bot's `.step()` call on a timer or scheduled task
  3. Persist `Bot.log` and `Bot.safety` to disk so state survives restarts

What this page IS:
  - A safe place to see exactly what the bot would do given current signals
  - A control panel for circuit-breaker limits and strategy parameters
  - A visualization of the decision log
"""
from __future__ import annotations

import datetime as dt
from typing import Any

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
    metric_card,
    section_header,
)
from trading_bot import (
    DEFAULT_CONFIDENCE_GATE,
    DEFAULT_DAILY_LOSS_LIMIT,
    DEFAULT_LONG_REGIMES,
    DEFAULT_MAX_DRAWDOWN_LIMIT,
    Bot,
    MockBroker,
)


apply_theme()

st.markdown(
    '<div style="display:flex; align-items:baseline; gap:14px;">'
    '<div style="font-size:24px; font-weight:700; letter-spacing:-0.02em;">'
    '◉ Trading Bot</div>'
    '<div style="color:#8B949E; font-size:13px;">Paper trading skeleton — '
    'connect a real broker when you trust the strategy</div>'
    "</div>",
    unsafe_allow_html=True,
)
st.write("")

st.error(
    "🛑 **PAPER TRADING ONLY.** This bot uses MockBroker — orders never reach "
    "any exchange. Treat the numbers below as 'what the bot WOULD have done'. "
    "Connecting a real broker is a deliberate engineering step (Alpaca adapter, "
    "key storage, scheduled execution) and your responsibility, not the app's."
)


# ---------------------------------------------------------------------------
# Sidebar — strategy + safety params
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Strategy")
ticker_for_bot = st.sidebar.text_input(
    "Ticker", value="SPY", key="bot_ticker"
).strip().upper()
sizing = st.sidebar.radio(
    "Sizing", ["binary", "confidence"], index=0, horizontal=True
)
confidence_gate = st.sidebar.slider(
    "Confidence gate", min_value=0.0, max_value=0.95,
    value=DEFAULT_CONFIDENCE_GATE, step=0.05,
)
long_regimes = st.sidebar.multiselect(
    "Long regimes",
    options=["Low Vol", "Medium-Low Vol", "Medium Vol",
             "Medium-High Vol", "High Vol", "Uncertain"],
    default=list(DEFAULT_LONG_REGIMES),
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Safety")
daily_loss = st.sidebar.slider(
    "Daily-loss circuit breaker (%)",
    min_value=0.5, max_value=10.0, value=DEFAULT_DAILY_LOSS_LIMIT * 100.0,
    step=0.5,
) / 100.0
max_dd = st.sidebar.slider(
    "Max drawdown circuit breaker (%)",
    min_value=2.0, max_value=30.0, value=DEFAULT_MAX_DRAWDOWN_LIMIT * 100.0,
    step=1.0,
) / 100.0

st.sidebar.markdown("---")
st.sidebar.markdown("### Broker")
starting_equity = st.sidebar.number_input(
    "Paper-trading starting equity ($)",
    min_value=1_000, max_value=10_000_000,
    value=100_000, step=1_000,
)
broker_choice = st.sidebar.selectbox(
    "Broker adapter",
    options=["MockBroker (in-memory)", "Alpaca paper (stub — not yet wired)"],
    index=0,
)
if broker_choice.startswith("Alpaca"):
    st.sidebar.warning(
        "Alpaca adapter is a stub. To enable, implement AlpacaBroker in "
        "trading_bot.py with the same interface as MockBroker, and store keys "
        "in .streamlit/secrets.toml (NOT in source)."
    )


# ---------------------------------------------------------------------------
# Initialize / reset bot
# ---------------------------------------------------------------------------
def _new_bot() -> Bot:
    return Bot(
        ticker=ticker_for_bot,
        broker=MockBroker(starting_equity=starting_equity),
        long_regimes=tuple(long_regimes),
        confidence_gate=confidence_gate,
        sizing=sizing,
        daily_loss_limit=daily_loss,
        max_drawdown_limit=max_dd,
    )


col_init, col_reset = st.sidebar.columns(2)
with col_init:
    if st.button("Initialize", use_container_width=True):
        st.session_state["bot"] = _new_bot()
with col_reset:
    if st.button("Reset", use_container_width=True):
        st.session_state.pop("bot", None)
        st.session_state.pop("bot_signals", None)

bot: Bot | None = st.session_state.get("bot")
if bot is None:
    st.info("Click **Initialize** in the sidebar to create a bot.")
    st.stop()


# ---------------------------------------------------------------------------
# Feed signals — either manual or replay from regime-dashboard result
# ---------------------------------------------------------------------------
section_header("Signal Source")
src = st.radio(
    "Where do signals come from?",
    options=["Manual one-shot", "Replay last regime-dashboard run"],
    horizontal=True,
)

if src == "Manual one-shot":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        regime_in = st.selectbox("Current regime",
                                   options=long_regimes + ["High Vol", "Uncertain"])
    with c2:
        conf_in = st.slider("Confidence", 0.0, 1.0, 0.85, 0.01)
    with c3:
        price_in = st.number_input("Price hint ($)", min_value=0.01,
                                     value=500.0, step=1.0)
    with c4:
        if st.button("Submit signal", type="primary", use_container_width=True):
            d = bot.step(regime_in, conf_in, price_in)
            st.session_state["bot"] = bot
elif src == "Replay last regime-dashboard run":
    result = st.session_state.get("result")
    if result is None:
        st.warning(
            "No regime-dashboard run found in session. Open the main page, "
            "run analysis, then return here."
        )
    else:
        df_full = result["data"]
        n_bars = len(df_full)
        st.caption(
            f"Loaded {n_bars} bars from the regime-dashboard session "
            f"({result.get('mode', 'In-sample')} mode). "
            f"Replay will step the bot through every bar in sequence."
        )
        if st.button("Replay all bars", type="primary",
                     use_container_width=False):
            progress = st.progress(0.0, text="Replaying signals…")
            for i, (ts, row) in enumerate(df_full.iterrows()):
                if i % 20 == 0:
                    progress.progress(i / n_bars, text=f"Bar {i}/{n_bars}")
                bot.step(
                    regime=str(row["regime"]),
                    confidence=float(row["confidence"]),
                    price=float(row["Close"]),
                    today_iso=pd.Timestamp(ts).date().isoformat(),
                )
            progress.empty()
            st.session_state["bot"] = bot
            st.success(f"Replayed {n_bars} bars. Equity: ${bot.broker.get_equity():,.0f}")


# ---------------------------------------------------------------------------
# Current state — top cards
# ---------------------------------------------------------------------------
section_header("Bot Status")

current_eq = bot.broker.get_equity()
qty = bot.broker.get_position(bot.ticker)
last_decision = bot.log[-1] if bot.log else None

status_color = REGIME_COLORS["High Vol"] if (bot.safety and bot.safety.any_tripped) else REGIME_COLORS["Low Vol"]
status_text = (
    "🛑 HALTED — " + ", ".join(k for k, v in (bot.safety.status() if bot.safety else {}).items() if v)
    if bot.safety and bot.safety.any_tripped
    else "✓ Operating"
)

s_cols = st.columns(5)
status_metrics = [
    ("Status", status_text, f"Broker: {bot.broker.name()}"),
    ("Equity (paper)", f"${current_eq:,.0f}",
     f"Start: ${bot.safety.starting_equity:,.0f}" if bot.safety else ""),
    ("Position", f"{qty:.3f} {bot.ticker}",
     f"Last decision: {last_decision.rationale[:30]}" if last_decision else "—"),
    ("Daily P&L",
     f"{(current_eq / bot.safety.daily_anchor_equity - 1) * 100:+.2f}%"
     if bot.safety and bot.safety.daily_anchor_equity > 0 else "—",
     f"Limit: −{daily_loss * 100:.1f}%"),
    ("Drawdown",
     f"{(current_eq / bot.safety.peak_equity - 1) * 100:+.2f}%"
     if bot.safety and bot.safety.peak_equity > 0 else "—",
     f"Limit: −{max_dd * 100:.1f}%"),
]
for col, (title, value, sub) in zip(s_cols, status_metrics):
    with col:
        st.markdown(
            metric_card(title=title, value=value, subtitle=sub, border_color=status_color),
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Circuit-breaker lights
# ---------------------------------------------------------------------------
section_header("Circuit Breakers")

cb_cols = st.columns(3)
breakers = bot.safety.status() if bot.safety else {"daily": False, "drawdown": False, "manual": False}
for col, (name, tripped) in zip(cb_cols, breakers.items()):
    with col:
        emoji = "🛑" if tripped else "🟢"
        st.markdown(
            metric_card(
                title=name.upper(),
                value=f"{emoji} {'TRIPPED' if tripped else 'armed'}",
                subtitle=("position forced to cash"
                           if tripped else "watching"),
                border_color=(REGIME_COLORS["High Vol"] if tripped
                               else REGIME_COLORS["Low Vol"]),
            ),
            unsafe_allow_html=True,
        )

m1, m2 = st.columns(2)
with m1:
    if st.button("🛑 Manual halt"):
        bot.trip_manual(True)
        st.session_state["bot"] = bot
with m2:
    if st.button("🔄 Clear manual halt"):
        bot.trip_manual(False)
        st.session_state["bot"] = bot


# ---------------------------------------------------------------------------
# Decision log
# ---------------------------------------------------------------------------
section_header(f"Decision Log ({len(bot.log)})")

if not bot.log:
    st.info("No decisions yet — submit a signal above.")
else:
    df_log = pd.DataFrame([
        {
            "Time": d.timestamp,
            "Regime": d.regime,
            "Confidence": f"{d.confidence:.0%}",
            "Target": f"{d.target_fraction:.0%}",
            "Side": d.order.get("side", ""),
            "ΔQty": f"{d.order.get('delta_qty', 0):+.3f}",
            "Price": f"${d.order.get('price', 0):.2f}",
            "Equity After": f"${d.equity_after:,.0f}",
            "Rationale": d.rationale,
        }
        for d in reversed(bot.log[-200:])
    ])
    st.dataframe(df_log, use_container_width=True, hide_index=True)

    # Equity curve from the log.
    eq_series = pd.Series(
        [d.equity_after for d in bot.log],
        index=pd.to_datetime([d.timestamp for d in bot.log], errors="coerce"),
    )
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=eq_series.index, y=eq_series.values,
        mode="lines", name="Paper equity",
        line=dict(color=ACCENT_CYAN, width=1.8),
        hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>$%{y:,.0f}<extra></extra>",
    ))
    fig_eq.add_hline(
        y=bot.safety.starting_equity if bot.safety else starting_equity,
        line_dash="dash", line_color=TEXT_MUTED,
        annotation_text="start", annotation_position="bottom right",
    )
    layout_eq = get_plotly_layout()
    layout_eq.update({
        "height": 320,
        "yaxis": {**layout_eq["yaxis"], "title": "Paper equity ($)"},
        "xaxis": {**layout_eq["xaxis"], "title": ""},
        "showlegend": False,
    })
    fig_eq.update_layout(**layout_eq)
    st.plotly_chart(fig_eq, use_container_width=True)


# ---------------------------------------------------------------------------
# Wiring guide
# ---------------------------------------------------------------------------
with st.expander("How to connect a real broker (Alpaca paper)"):
    st.markdown(
        """
1. **Install the adapter SDK** — add `alpaca-py>=0.20.0` to requirements.txt.

2. **Implement `AlpacaBroker`** in `trading_bot.py` with the same methods
   as `MockBroker`:
   `name`, `is_live`, `get_equity`, `get_position`, `submit_target_position`.

3. **Store API keys** — put `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` in
   `.streamlit/secrets.toml` (already in `.dockerignore`). **Never commit
   keys to source.**

4. **Wire the choice** — extend the broker-choice dropdown in this page
   to instantiate `AlpacaBroker()` when selected, and pass it to `Bot(broker=...)`.

5. **Schedule `.step()` execution** — Streamlit reruns the page on each
   interaction. For a 24/7 bot you need an external scheduler (cron,
   APScheduler, or a separate worker container) that imports `Bot` and
   calls `.step()` on a fixed cadence. Persist `bot.log` and `bot.safety`
   to a file or DB between runs.
"""
    )
