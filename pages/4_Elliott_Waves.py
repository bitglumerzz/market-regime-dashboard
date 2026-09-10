"""Elliott Waves multi-timeframe analyzer (top-down).

Reads 1d / 4h / 15m. For each timeframe:
  - MINOR ZigZag (thin gray) — visual context, fine swing structure
  - MAJOR ZigZag (bold cyan) — drives Elliott classification
  - Labeled wave structure (0..5 / 0/A/B/C) on the winning hypothesis
  - Fibonacci retracement lines drawn ONLY across the labeled region

A top-down synthesis panel at the top reads all 3 TFs and produces the
trading-setup verdict (LONG / SHORT / WAIT).
"""
from __future__ import annotations

import os

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
from elliott import WaveCandidate, classify, multi_label_swings
from forecast import WaveForecast
from i18n import _RU, t, translate_verdict
from multi_tf import (
    DEFAULT_TF_CONFIGS,
    TFAnalysis,
    analyze_ticker,
    top_down_synthesis,
)
from wave_ai import (
    ask_claude, ask_claude_holistic, availability_message, is_available,
    ClaudeSpendLimitError,
)
from wave_validator import validate_detector
from zigzag import Swing
from prediction_log import log_per_tf, log_holistic
from wave_journal import (
    log_snapshot, log_action, detect_and_log_transitions,
)
from holistic_store import (
    save_last_holistic, load_last_holistic, age_minutes,
)
from holistic_history import (
    save_holistic_entry, load_history, verify_history,
)
from holistic_report import build_mcp_style_report
from translation_helpers import translate_detector_text, translate_pattern_slug
from tv_mcp_analysis_store import (
    save_manual_analysis as save_mcp_analysis,
    load_last_analysis as load_mcp_analysis,
    age_minutes as mcp_age_minutes,
)


apply_theme()

st.markdown(
    '<div style="display:flex; align-items:baseline; gap:14px;">'
    '<div style="font-size:24px; font-weight:700; letter-spacing:-0.02em;">'
    '◉ Elliott Waves — Top-Down Multi-TF</div>'
    '<div style="color:#8B949E; font-size:13px;">1d → 4h → 15m. Major ZigZag '
    'drives classification, minor ZigZag adds context.</div>'
    "</div>",
    unsafe_allow_html=True,
)
st.write("")

st.info(
    "**How to read:** start from the **1d** card (macro bias), then check "
    "the **4h** (sub-structure inside the 1d wave), then **15m** (entry "
    "timing). The verdict at the top combines all three. R0 = net direction "
    "sanity (most important rule). Bigger moves score higher than micro-noise."
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Elliott Waves configuration")

ticker = st.sidebar.text_input("Ticker", value="ETH-USD",
                                 key="elliott_ticker").strip().upper()

data_source = st.sidebar.selectbox(
    "Data source",
    options=["auto", "ccxt", "yfinance"],
    index=0,
    help=(
        "auto = CCXT for crypto (better 4h/15m), yfinance otherwise. "
        "ccxt = force Binance (native 4h, years of 15m). "
        "yfinance = force yfinance (limited intraday history)."
    ),
)

include_5m = st.sidebar.checkbox(
    "Include 5m timeframe", value=True,
    help="Adds a 4th timeframe (5m) for live entry timing. "
          "Available history is short (~5 days) — meant for now-decision only.",
)

show_claude_overlay = st.sidebar.checkbox(
    "Показывать уровни Claude на графиках", value=True,
    help="Когда есть holistic мнение Claude — рисует entry/stop/target "
          "horizontal levels поверх каждого TF графика + зоны риска/прибыли + "
          "структурную позицию в углу.",
)

make_tv_annotations = st.sidebar.checkbox(
    "📌 Аннотировать TV-скриншоты по ответу Claude", value=True,
    help="После holistic-вызова берёт chart_annotations от Claude (pivot/fib/"
          "zone/arrow/wave_label/note) и наносит их PIL'ом на TV-скриншоты, "
          "захваченные через tv_screenshot. Сохраняет в "
          "predictions/screenshots_annotated/{pid}_{tf}_annotated.png.",
)

# Phase 32 — MCP-style markdown report vs compact widgets
holistic_view_mode = st.sidebar.radio(
    "📋 Формат holistic-ответа",
    options=["MCP-style report", "Compact widgets"],
    index=0,
    help="MCP-style — единый markdown-отчёт с эмодзи-заголовками секций "
         "(как у TradingView MCP в чате). Compact — текущие виджеты с метрик-"
         "картами, dataframe'ами и expander'ами.",
)

tv_layout_name = st.sidebar.selectbox(
    "TV layout (для координатной калибровки)",
    options=["dark_default", "wide_dark", "no_toolbar"],
    index=0,
    help="Если TV-скриншот делается с не-стандартным layout (другие padding'и "
          "слева/справа), переключи. Подбирается один раз под твоё TV окно.",
)

st.sidebar.markdown("**MAJOR thresholds** (drive classification)")
# Pull calibrated defaults if available.
_d = {cfg["name"]: cfg for cfg in DEFAULT_TF_CONFIGS}
maj_1d = st.sidebar.slider(
    "1d major (%)", min_value=1.0, max_value=25.0,
    value=float(_d.get("1d", {}).get("major_threshold_pct", 0.12)) * 100,
    step=0.25,
) / 100.0
maj_4h = st.sidebar.slider(
    "4h major (%)", min_value=0.5, max_value=15.0,
    value=float(_d.get("4h", {}).get("major_threshold_pct", 0.06)) * 100,
    step=0.25,
) / 100.0
maj_15m = st.sidebar.slider(
    "15m major (%)", min_value=0.2, max_value=6.0,
    value=float(_d.get("15m", {}).get("major_threshold_pct", 0.025)) * 100,
    step=0.1,
) / 100.0
if include_5m:
    maj_5m = st.sidebar.slider(
        "5m major (%)", min_value=0.1, max_value=3.0,
        value=float(_d.get("5m", {}).get("major_threshold_pct", 0.012)) * 100,
        step=0.05,
    ) / 100.0
else:
    maj_5m = float(_d.get("5m", {}).get("major_threshold_pct", 0.012))

st.sidebar.markdown("**MINOR thresholds** (visual context)")
min_1d = st.sidebar.slider(
    "1d minor (%)", min_value=0.5, max_value=10.0,
    value=float(_d.get("1d", {}).get("minor_threshold_pct", 0.05)) * 100,
    step=0.25,
) / 100.0
min_4h = st.sidebar.slider(
    "4h minor (%)", min_value=0.2, max_value=5.0,
    value=float(_d.get("4h", {}).get("minor_threshold_pct", 0.025)) * 100,
    step=0.1,
) / 100.0
min_15m = st.sidebar.slider(
    "15m minor (%)", min_value=0.1, max_value=3.0,
    value=float(_d.get("15m", {}).get("minor_threshold_pct", 0.010)) * 100,
    step=0.05,
) / 100.0
if include_5m:
    min_5m = st.sidebar.slider(
        "5m minor (%)", min_value=0.05, max_value=2.0,
        value=float(_d.get("5m", {}).get("minor_threshold_pct", 0.005)) * 100,
        step=0.05,
    ) / 100.0
else:
    min_5m = float(_d.get("5m", {}).get("minor_threshold_pct", 0.005))

run_btn = st.sidebar.button("Run Analysis", type="primary",
                              use_container_width=True, key="elliott_run")


# ---------------------------------------------------------------------------
# Run analysis (cached)
# ---------------------------------------------------------------------------
tf_configs = [
    {**DEFAULT_TF_CONFIGS[0], "major_threshold_pct": maj_1d,
     "minor_threshold_pct": min_1d},
    {**DEFAULT_TF_CONFIGS[1], "major_threshold_pct": maj_4h,
     "minor_threshold_pct": min_4h},
    {**DEFAULT_TF_CONFIGS[2], "major_threshold_pct": maj_15m,
     "minor_threshold_pct": min_15m},
]
if include_5m and len(DEFAULT_TF_CONFIGS) >= 4:
    tf_configs.append({
        **DEFAULT_TF_CONFIGS[3],
        "major_threshold_pct": maj_5m,
        "minor_threshold_pct": min_5m,
    })

cache_key = (ticker, maj_1d, maj_4h, maj_15m, maj_5m,
              min_1d, min_4h, min_15m, min_5m,
              include_5m, data_source)

# Phase 21+ — STRICT LAZY analysis. analyze_ticker is heavy (15-30s, hits
# multiple exchanges via CCXT, computes thousands of swings). Auto-running it
# on every page entry burns time and rate-limits the upstream APIs. Run ONLY
# when the user explicitly clicks "Run Analysis" — that's the contract.
#
# This means: on first page entry (or after container rebuild that wipes
# session_state) the user sees "Click Run Analysis" and has to press the
# button. That's intentional. After the first click, results persist in
# session_state for the rest of the session — page navigation back to
# Elliott Waves restores them without a re-run.
_cached_key = st.session_state.get("elliott_cache_key")

if run_btn:
    with st.spinner(f"Analyzing {ticker} across 3 timeframes…"):
        try:
            results = analyze_ticker(
                ticker, tf_configs=tf_configs, source=data_source
            )
            st.session_state["elliott_results"] = results
            st.session_state["elliott_cache_key"] = cache_key
        except Exception as exc:
            st.error(f"Analysis failed: {type(exc).__name__}: {exc}")
            st.stop()

results: list[TFAnalysis] = st.session_state.get("elliott_results", [])
if not results:
    st.info(
        "📊 Нажми **Run Analysis** в sidebar чтобы запустить анализ. "
        "Анализ занимает 15-30 секунд (CCXT берёт данные с нескольких "
        "бирж + расчёт волновой структуры). После первого клика результат "
        "сохранится в session_state до конца сессии — переход на другие "
        "вкладки и обратно не запустит повторный анализ."
    )
    st.stop()

# Surface a "stale" badge if anything changed since the last analysis:
# - ticker (different symbol)
# - slider thresholds (different swings)
# User can choose to click Run Analysis to refresh, but we don't force it.
if _cached_key is not None and _cached_key != cache_key:
    if _cached_key[0] != ticker:
        st.error(
            f"🚨 **TICKER MISMATCH** — в sidebar выбран `{ticker}`, но "
            f"графики и анализ показывают данные для `{_cached_key[0]}`!\n\n"
            f"❌ Holistic Claude **заблокирован** до тех пор пока ты не "
            f"нажмёшь **Run Analysis** в sidebar — иначе в Track Record "
            f"запишется ложный прогноз с данными одного тикера и именем "
            f"другого.\n\n"
            f"✅ Нажми **Run Analysis** чтобы загрузить актуальные данные "
            f"для {ticker}."
        )
    else:
        st.caption(
            "⚠️ Slider'ы изменены с последнего анализа — графики показывают "
            "старые swings. Нажми **Run Analysis** в sidebar чтобы обновить."
        )


# ---------------------------------------------------------------------------
# Phase 20 — auto-load persisted holistic opinion from disk for this ticker.
# This survives Streamlit page navigation and browser reloads. Without this,
# the holistic Claude answer disappears every time the user navigates away.
# ---------------------------------------------------------------------------
_current_full = st.session_state.get("last_holistic_full")
if not _current_full or _current_full.get("ticker") != ticker:
    _persisted = load_last_holistic(ticker)
    if _persisted and isinstance(_persisted.get("opinion"), dict):
        _age = age_minutes(_persisted)
        st.session_state["last_holistic_full"] = _persisted["opinion"]
        if _persisted.get("pid"):
            st.session_state["last_holistic_pid"] = _persisted["pid"]
        if _age is not None:
            _age_str = (f"{int(_age)} мин назад" if _age < 60
                        else f"{int(_age / 60)} ч {int(_age % 60)} мин назад")
            st.info(
                f"🕒 Восстановлен сохранённый holistic-ответ Claude для "
                f"{ticker} от {_age_str}. Чтобы обновить — нажми "
                f"«🧠 Получить holistic мнение Claude» ниже."
            )


# ---------------------------------------------------------------------------
# Auto-log a snapshot of this analysis. Idempotent within the same minute —
# if the per-TF fingerprint is identical, no extra row is written.
# Transitions (top-pattern flips between snapshots) are detected and logged
# too, so the «Detector Stability» track-record tab has data.
# ---------------------------------------------------------------------------
try:
    synth_for_log = top_down_synthesis(results)
    _snapshot = log_snapshot(ticker, results, synth_for_log,
                              source="elliott_page")
    st.session_state["wave_snapshot_id"] = _snapshot["id"]
    st.session_state["wave_snapshot_logged_at"] = _snapshot["logged_at"]
    _trans = detect_and_log_transitions(_snapshot)
    if _trans:
        st.session_state["wave_transitions_recent"] = _trans
except Exception as _journal_exc:
    # Never let journaling break the analysis UI.
    st.session_state["wave_journal_error"] = (
        f"{type(_journal_exc).__name__}: {_journal_exc}"
    )


# ---------------------------------------------------------------------------
# Top-down synthesis
# ---------------------------------------------------------------------------
section_header(t("section.synthesis"))

synth = top_down_synthesis(results)
# Translate the verdict text from English markers to Russian.
synth["setup"] = translate_verdict(synth["setup"])

color_map = {
    "long":  REGIME_COLORS["Low Vol"],
    "short": REGIME_COLORS["High Vol"],
    "wait":  TEXT_MUTED,
}
verdict_color = color_map.get(synth["color"], TEXT_MUTED)

verdict_emoji = {"long": "🟢", "short": "🔴", "wait": "⚪️"}[synth["color"]]
st.markdown(
    f'<div style="padding:14px 18px; border-left: 4px solid {verdict_color}; '
    f'background:#161B22; border-radius: 8px;">'
    f'<div style="font-size:11px; color:#8B949E; text-transform:uppercase; '
    f'letter-spacing:.08em; margin-bottom:6px;">{t("card.trading_setup")}</div>'
    f'<div style="font-size:22px; font-weight:700; color:{verdict_color};">'
    f'{verdict_emoji} {synth["setup"]}</div>'
    f'</div>',
    unsafe_allow_html=True,
)
st.write("")

# Show 4 cards when 5m is included, otherwise 3.
synth_metrics = [
    (t("card.macro"), synth["macro_view"], t("card.macro.sub")),
    (t("card.meso"),  synth["meso_view"],  t("card.meso.sub")),
    (t("card.micro"), synth["micro_view"], t("card.micro.sub")),
]
if synth.get("entry_view") is not None:
    synth_metrics.append(
        (t("card.entry"), synth["entry_view"], t("card.entry.sub"))
    )

cols = st.columns(len(synth_metrics))
for col, (title, value, sub) in zip(cols, synth_metrics):
    with col:
        st.markdown(
            metric_card(title=title, value=value, subtitle=sub,
                        border_color=verdict_color),
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Action journal — record what you actually did with this analysis
# ---------------------------------------------------------------------------
section_header("📒 Журнал действий")

_snap_id = st.session_state.get("wave_snapshot_id")
_snap_at = st.session_state.get("wave_snapshot_logged_at", "")
_jrn_err = st.session_state.get("wave_journal_error")
if _jrn_err:
    st.warning(f"Журнал недоступен: {_jrn_err}")

st.caption(
    f"Текущий snapshot: `{_snap_id or '—'}`  ·  записан: "
    f"{_snap_at[:16] if _snap_at else '—'}  ·  каждое действие связывается "
    f"с этим snapshot'ом и позже оценивается по реальной цене на странице "
    f"Track Record."
)

# Pre-fill entry/stop/target from holistic Claude opinion if it was just made;
# otherwise from the 1d forecast's first target / invalidation as a reasonable
# starting guess. The user can override before pressing «Взял setup».
_holistic_cached = st.session_state.get("last_holistic_for_action")

def _default_levels() -> tuple[float | None, float | None, float | None, str, str]:
    """Returns (entry, stop, target, direction, tf) — best-guess defaults."""
    # 1) From most recent holistic Claude opinion if any
    if _holistic_cached is not None and _holistic_cached.get("ticker") == ticker:
        h = _holistic_cached
        bias = h.get("bias") or ""
        direction = ("long" if bias == "long"
                      else "short" if bias == "short" else "")
        return (h.get("entry"), h.get("stop_loss"), h.get("target"),
                direction, "all")
    # 2) From the 1d forecast targets / invalidation
    for r in results:
        if r.ok and r.name == "1d" and r.forecast and r.forecast.is_actionable:
            fc = r.forecast
            entry = float(r.prices.iloc[-1]) if len(r.prices) else None
            target = fc.targets[0].price if fc.targets else None
            stop = fc.invalidation_level
            direction = ("long" if fc.direction == "up"
                          else "short" if fc.direction == "down" else "")
            return (entry, stop, target, direction, "1d")
    return (None, None, None, "", "")


_def_entry, _def_stop, _def_target, _def_dir, _def_tf = _default_levels()

with st.expander("✏️  Параметры сделки (для «Взял setup»)", expanded=False):
    af1, af2, af3, af4 = st.columns(4)
    with af1:
        action_direction = st.selectbox(
            "Направление", options=["long", "short"],
            index=0 if _def_dir != "short" else 1,
            key="action_direction",
        )
    with af2:
        action_tf = st.selectbox(
            "TF на котором играю",
            options=["1d", "4h", "15m", "5m", "all"],
            index=(["1d", "4h", "15m", "5m", "all"].index(_def_tf)
                   if _def_tf in ["1d", "4h", "15m", "5m", "all"] else 0),
            key="action_tf",
        )
    with af3:
        entry_input = st.number_input(
            "Entry", value=float(_def_entry or 0.0), step=0.01,
            format="%.4f", key="action_entry",
        )
    with af4:
        stop_input = st.number_input(
            "Stop", value=float(_def_stop or 0.0), step=0.01,
            format="%.4f", key="action_stop",
        )
    target_input = st.number_input(
        "Target", value=float(_def_target or 0.0), step=0.01,
        format="%.4f", key="action_target",
    )
    notes_input = st.text_area(
        "Заметка (опционально)", value="", height=70, key="action_notes",
        placeholder="например: жду подтверждения 5m, сократил размер из-за новостей…",
    )

action_btn_cols = st.columns(4)
with action_btn_cols[0]:
    if st.button("✅ Взял сетап", type="primary", use_container_width=True,
                  key="act_take"):
        try:
            log_action(
                _snap_id, ticker, "take_setup",
                direction=action_direction,
                entry=entry_input or None,
                stop_loss=stop_input or None,
                target=target_input or None,
                timeframe=action_tf,
                price_at_action=entry_input or None,
                notes=notes_input,
            )
            st.success("Записано в журнал действий ✓")
        except Exception as exc:
            st.error(f"Не удалось записать: {type(exc).__name__}: {exc}")
with action_btn_cols[1]:
    if st.button("⏭️ Пропустил", use_container_width=True, key="act_skip"):
        try:
            log_action(_snap_id, ticker, "skip",
                        direction=action_direction,
                        timeframe=action_tf, notes=notes_input)
            st.success("Записано: «пропустил» ✓")
        except Exception as exc:
            st.error(f"Ошибка: {type(exc).__name__}: {exc}")
with action_btn_cols[2]:
    if st.button("❌ Закрыл по стопу", use_container_width=True, key="act_stop"):
        try:
            log_action(_snap_id, ticker, "stop_hit",
                        direction=action_direction,
                        entry=entry_input or None,
                        stop_loss=stop_input or None,
                        timeframe=action_tf,
                        price_at_action=stop_input or None,
                        notes=notes_input)
            st.success("Записано: «стоп» ✓")
        except Exception as exc:
            st.error(f"Ошибка: {type(exc).__name__}: {exc}")
with action_btn_cols[3]:
    if st.button("💰 Закрыл в плюс", use_container_width=True, key="act_tp"):
        try:
            log_action(_snap_id, ticker, "take_profit",
                        direction=action_direction,
                        entry=entry_input or None,
                        target=target_input or None,
                        timeframe=action_tf,
                        price_at_action=target_input or None,
                        notes=notes_input)
            st.success("Записано: «тейк» ✓")
        except Exception as exc:
            st.error(f"Ошибка: {type(exc).__name__}: {exc}")

# Show recent transitions for context (only when there's something new).
_recent_trans = st.session_state.get("wave_transitions_recent") or []
if _recent_trans:
    st.markdown("**🔁 Детектор переключился по сравнению с прошлым запуском:**")
    for tr in _recent_trans:
        # Phase 37 — переводим english patterns/positions на русский
        from_slug = translate_pattern_slug(tr['from_pattern'])
        to_slug   = translate_pattern_slug(tr['to_pattern'])
        from_pos  = translate_detector_text(tr.get('from_position', ''))
        to_pos    = translate_detector_text(tr.get('to_position', ''))
        st.markdown(
            f"- **{tr['timeframe']}**: `{from_slug}` "
            f"({from_pos}) → `{to_slug}` "
            f"({to_pos})"
        )


# ---------------------------------------------------------------------------
# Per-TF detail charts
# ---------------------------------------------------------------------------
def _plot_tf(r: TFAnalysis) -> go.Figure:
    fig = go.Figure()
    # Price line.
    fig.add_trace(go.Scatter(
        x=r.prices.index, y=r.prices.values,
        mode="lines", name="Close",
        line=dict(color="#FFFFFF", width=1.4),
        hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>%{y:.4f}<extra></extra>",
    ))

    # MINOR ZigZag — thin, gray, fades to the background.
    if r.minor_swings:
        zz_x = [s.index for s in r.minor_swings]
        zz_y = [s.price for s in r.minor_swings]
        fig.add_trace(go.Scatter(
            x=zz_x, y=zz_y,
            mode="lines+markers", name=f"minor ZZ ({r.minor_threshold_pct:.1%})",
            line=dict(color=hex_to_rgba(TEXT_MUTED, 0.45), width=0.9),
            marker=dict(size=3, color=hex_to_rgba(TEXT_MUTED, 0.6)),
            hoverinfo="skip", showlegend=False,
        ))

    # MAJOR ZigZag — bold cyan, the structure that drives classification.
    if r.major_swings:
        zz_x = [s.index for s in r.major_swings]
        zz_y = [s.price for s in r.major_swings]
        fig.add_trace(go.Scatter(
            x=zz_x, y=zz_y,
            mode="lines+markers", name=f"major ZZ ({r.major_threshold_pct:.1%})",
            line=dict(color=ACCENT_CYAN, width=2.0),
            marker=dict(size=8, color=ACCENT_CYAN, symbol="circle"),
            hovertemplate=("<b>%{x|%Y-%m-%d %H:%M}</b><br>major swing %{y:.4f}"
                           "<extra></extra>"),
            showlegend=False,
        ))

    # Forecast — projection lines and zones extending past the last swing.
    fc = r.forecast
    if (fc is not None and fc.is_actionable
            and r.top is not None and r.top.swings_used):
        last_swing = r.top.swings_used[-1]
        # Project forecast levels into the future from the last bar.
        # Use forecast duration to place vertical end-line, default to ~20% of chart.
        last_ts = r.prices.index[-1]
        # Convert expected duration in bars to a timedelta using median bar gap.
        if len(r.prices.index) >= 2:
            gaps = (pd.Series(r.prices.index).diff().dt.total_seconds()
                    .dropna())
            median_gap = float(gaps.median()) if len(gaps) else 86400.0
        else:
            median_gap = 86400.0  # fallback 1 day
        duration_lo, duration_hi = (
            fc.expected_duration_bars
            if fc.expected_duration_bars is not None
            else (5, 25)
        )
        forecast_end_ts = (last_ts +
                           pd.Timedelta(seconds=median_gap * duration_hi))

        # Soft band rectangle covering the forecast price range.
        target_prices = [t_.price for t_ in fc.targets]
        if target_prices:
            zone_hi = max([last_swing.price] + target_prices)
            zone_lo = min([last_swing.price] + target_prices)
            forecast_color = (REGIME_COLORS["Low Vol"]
                              if fc.direction == "up"
                              else REGIME_COLORS["High Vol"])
            fig.add_shape(
                type="rect", xref="x", yref="y",
                x0=last_ts, x1=forecast_end_ts,
                y0=zone_lo, y1=zone_hi,
                fillcolor=hex_to_rgba(forecast_color, 0.10),
                line=dict(color=hex_to_rgba(forecast_color, 0.35),
                          width=1, dash="dot"),
                layer="below",
            )
            # Dotted horizontal lines for each target with labels.
            for tg in fc.targets:
                fig.add_shape(
                    type="line", xref="x", yref="y",
                    x0=last_ts, x1=forecast_end_ts,
                    y0=tg.price, y1=tg.price,
                    line=dict(color=hex_to_rgba(forecast_color, 0.55),
                              width=1.2, dash="dot"),
                )
                fig.add_annotation(
                    x=forecast_end_ts, y=tg.price,
                    text=f"{tg.fib_label}  {tg.price:.4f}",
                    showarrow=False,
                    font=dict(color=forecast_color, size=10),
                    xanchor="left", yanchor="middle", xshift=4,
                )
            # Invalidation horizontal — red dashed line.
            if fc.invalidation_level is not None:
                fig.add_shape(
                    type="line", xref="x", yref="y",
                    x0=last_ts, x1=forecast_end_ts,
                    y0=fc.invalidation_level, y1=fc.invalidation_level,
                    line=dict(color=REGIME_COLORS["High Vol"],
                              width=1.4, dash="dash"),
                )
                fig.add_annotation(
                    x=forecast_end_ts, y=fc.invalidation_level,
                    text=f"❌ {fc.invalidation_level:.4f}",
                    showarrow=False,
                    font=dict(color=REGIME_COLORS["High Vol"], size=10),
                    xanchor="left", yanchor="middle", xshift=4,
                )

    # Wave labels — MULTI-PATTERN: cover the WHOLE swing series with
    # back-to-back labeled patterns (not just the last 6 swings).
    # This mirrors how a professional analyst would mark the chart
    # (impulse 1-2-3-4-5 followed by zigzag A-B-C, etc.).
    labeled = multi_label_swings(r.major_swings, min_score=40.0)
    if labeled:
        # Anti-collision: group labels falling within a tight bar window
        # and stack them vertically (yshift) so they don't overlap.
        # Without this, overlapping back-to-back patterns produce a single
        # illegible blob of labels at the right edge of the chart.
        # We group by 3-bar buckets and apply progressive yshift.
        max_pattern_idx = max((x.pattern_index for x in labeled), default=0)
        # Sort by chronological bar_index for deterministic stacking order
        labeled_sorted = sorted(labeled, key=lambda l: (l.swing.bar_index,
                                                          l.pattern_index))
        # Track occupied (bar_bucket, slot) so we can offset overlapping labels
        slots_per_bucket: dict[int, int] = {}
        BUCKET = 3   # bars per collision bucket
        SLOT_PX = 18  # vertical offset per stacked label, pixels

        for ls in labeled_sorted:
            bg = (REGIME_COLORS["Low Vol"] if ls.is_up
                  else REGIME_COLORS["High Vol"])
            # Fade earlier patterns slightly so the most recent stands out
            opacity = (0.85 if ls.pattern_index >= max_pattern_idx - 1
                       else 0.55)
            bucket = ls.swing.bar_index // BUCKET
            slot = slots_per_bucket.get(bucket, 0)
            slots_per_bucket[bucket] = slot + 1
            # Direction: low swings → labels go DOWN, high swings → UP
            sign = -1 if ls.swing.kind == "high" else 1
            yshift_px = sign * slot * SLOT_PX
            # Smaller font for stacked overflow labels (slot ≥ 1) to keep
            # the visual hierarchy: first label = main, rest = annotations
            fsize = 11 if slot == 0 else 9
            fig.add_annotation(
                x=ls.swing.index, y=ls.swing.price, text=f"<b>{ls.label}</b>",
                showarrow=False,
                bgcolor=hex_to_rgba(bg, opacity),
                font=dict(color="#0B0E13", size=fsize),
                bordercolor=bg, borderwidth=1, borderpad=2,
                xanchor="center", yanchor="middle",
                yshift=yshift_px,
            )

    # Fallback for the very top candidate when multi-label didn't cover the
    # tail — keep showing top.labels in the recent zone if multi missed it.
    top = r.top
    if top is not None and top.labels and top.score > 0:
        labeled_swing_ids = {id(ls.swing) for ls in labeled}
        if not all(id(s) in labeled_swing_ids for s in top.swings_used):
            is_up = "up" in top.pattern
            bg = REGIME_COLORS["Low Vol"] if is_up else REGIME_COLORS["High Vol"]
            for s, label in zip(top.swings_used, top.labels):
                if id(s) in labeled_swing_ids:
                    continue
                fig.add_annotation(
                    x=s.index, y=s.price, text=f"<b>{label}</b>",
                    showarrow=False,
                    bgcolor=hex_to_rgba(bg, 0.85),
                    font=dict(color="#0B0E13", size=12),
                    bordercolor=bg, borderwidth=1, borderpad=2,
                    xanchor="center", yanchor="middle",
                )

    # Phase 3.1 — Multi-degree labeling: classify the MINOR ZigZag too and
    # draw sub-wave labels in parentheses, smaller and less prominent.
    # This mimics TradingView's overlay of multiple degrees on one chart.
    if r.minor_swings and len(r.minor_swings) >= 4:
        try:
            minor_candidates = classify(r.minor_swings, top_k=1)
        except Exception:
            minor_candidates = []
        if minor_candidates and minor_candidates[0].score > 0:
            mc = minor_candidates[0]
            # Skip if minor pattern coincides exactly with major (avoid clutter).
            if mc.swings_used != top.swings_used if top else True:
                mc_is_up = "up" in mc.pattern
                mc_color = (REGIME_COLORS["Low Vol"] if mc_is_up
                            else REGIME_COLORS["High Vol"])
                for s, label in zip(mc.swings_used, mc.labels):
                    fig.add_annotation(
                        x=s.index, y=s.price,
                        text=f"<i>({label})</i>",
                        showarrow=False,
                        bgcolor=hex_to_rgba("#0B0E13", 0.7),
                        font=dict(color=hex_to_rgba(mc_color, 0.8), size=9),
                        bordercolor=hex_to_rgba(mc_color, 0.4),
                        borderwidth=1, borderpad=1,
                        xanchor="center", yanchor="middle",
                        yshift=-12,    # offset below the swing point
                    )

        # Fibonacci levels — drawn only across the labeled wave region.
        # Guard against r.top being None (no decisive pattern detected on
        # this TF) — without this guard the page crashes on AttributeError.
        if top is not None and top.fibs and top.swings_used:
            x_start = top.swings_used[0].index
            x_end = r.prices.index[-1]
            shown_fibs = ("38.2%", "50%", "61.8%", "78.6%", "100%")
            for fib_label, fib_price in top.fibs.items():
                if fib_label not in shown_fibs:
                    continue
                fig.add_shape(
                    type="line", xref="x", yref="y",
                    x0=x_start, x1=x_end,
                    y0=fib_price, y1=fib_price,
                    line=dict(color=hex_to_rgba(TEXT_MUTED, 0.45),
                              width=1, dash="dot"),
                    layer="below",
                )
                fig.add_annotation(
                    x=x_end, y=fib_price,
                    text=f"Fib {fib_label}",
                    showarrow=False,
                    font=dict(color=TEXT_MUTED, size=9),
                    xanchor="left", yanchor="middle", xshift=4,
                )

    # Visual badge for which resolution layer produced the threshold.
    # Vol-formula label notes that it's 1d-fit scaled across TFs via √time.
    src_badge = {
        "per_ticker": "🎯 per-ticker",
        "vol_formula": ("📐 vol-formula" if r.name == "1d"
                         else f"📐 vol-formula (scaled from 1d)"),
        "aggregate":  "Σ aggregate",
        "builtin":    "default",
    }.get(r.threshold_source, r.threshold_source)

    layout = get_plotly_layout()
    title_bits = [
        f"<b>{r.name}</b>",
        f"major {r.major_threshold_pct:.1%} ({len(r.major_swings)} swings) "
        f"<span style='color:#8B949E; font-size:11px;'>· {src_badge}</span>",
        f"minor {r.minor_threshold_pct:.1%} ({len(r.minor_swings)} swings)",
    ]
    layout.update({
        "height": 360,
        "title": dict(
            text="  ·  ".join(title_bits),
            font=dict(color=TEXT_PRIMARY, size=14),
            x=0.02, xanchor="left",
        ),
        "xaxis": {**layout["xaxis"], "title": ""},
        "yaxis": {**layout["yaxis"], "title": "Price"},
        "margin": {"l": 60, "r": 60, "t": 40, "b": 30},
        "showlegend": False,
    })
    fig.update_layout(**layout)
    return fig


section_header(t("section.per_tf"))


def _zoom_plot(r: TFAnalysis, n_recent: int) -> go.Figure:
    """Re-render the same chart but limited to the last n_recent bars,
    keeping the forecast zone fully visible past the right edge."""
    if len(r.prices) <= n_recent:
        return _plot_tf(r)
    cutoff = r.prices.index[-n_recent]

    sliced = TFAnalysis(
        name=r.name, interval=r.interval,
        prices=r.prices.loc[cutoff:],
        major_swings=[s for s in r.major_swings if s.index >= cutoff],
        minor_swings=[s for s in r.minor_swings if s.index >= cutoff],
        candidates=r.candidates,  # keep same hypotheses (built on full series)
        major_threshold_pct=r.major_threshold_pct,
        minor_threshold_pct=r.minor_threshold_pct,
        source=r.source,
        threshold_source=r.threshold_source,
        forecast=r.forecast,
        error="",
    )
    return _plot_tf(sliced)


# How many recent bars to show in the zoom expander, per TF.
ZOOM_BARS = {"1d": 120, "4h": 180, "15m": 200, "5m": 200}


def _overlay_claude_opinion(fig: go.Figure, opinion: dict,
                              prices: pd.Series, tf_name: str) -> None:
    """Paint Claude's holistic plan on top of one TF chart.

    Draws:
      * horizontal entry/stop/target lines with labels (only when the level
        falls inside the visible price range — otherwise the line would be
        off-screen on small TFs)
      * a translucent risk zone (entry ↔ stop) and reward zone (entry ↔ target)
      * a corner annotation with bias emoji + structure_position summary
    """
    if not opinion or prices.empty:
        return

    entry = opinion.get("entry")
    stop = opinion.get("stop_loss")
    target = opinion.get("target")
    bias = (opinion.get("bias") or "neutral").lower()
    action = (opinion.get("action") or "wait").lower()
    sq = (opinion.get("setup_quality") or "none").lower()

    # Visible price range with some breathing room
    p_min = float(prices.min())
    p_max = float(prices.max())
    span = max(p_max - p_min, 1e-9)
    visible_lo = p_min - span * 0.05
    visible_hi = p_max + span * 0.05

    def _in_range(v) -> bool:
        return (isinstance(v, (int, float))
                and v == v  # not NaN
                and visible_lo <= v <= visible_hi)

    x_start = prices.index[0]
    x_end = prices.index[-1]

    # Colors by bias
    long_color = REGIME_COLORS["Low Vol"]
    short_color = REGIME_COLORS["High Vol"]
    entry_color = long_color if bias == "long" else short_color if bias == "short" else ACCENT_CYAN

    # Risk zone — entry to stop
    if _in_range(entry) and _in_range(stop):
        fig.add_shape(
            type="rect", xref="x", yref="y",
            x0=x_start, x1=x_end,
            y0=min(float(entry), float(stop)), y1=max(float(entry), float(stop)),
            fillcolor=hex_to_rgba(short_color, 0.08),
            line=dict(color=hex_to_rgba(short_color, 0.0), width=0),
            layer="below",
        )

    # Reward zone — entry to target
    if _in_range(entry) and _in_range(target):
        fig.add_shape(
            type="rect", xref="x", yref="y",
            x0=x_start, x1=x_end,
            y0=min(float(entry), float(target)), y1=max(float(entry), float(target)),
            fillcolor=hex_to_rgba(long_color, 0.08),
            line=dict(color=hex_to_rgba(long_color, 0.0), width=0),
            layer="below",
        )

    # Horizontal lines + right-edge labels
    def _hline(price: float, color: str, dash: str, label: str) -> None:
        fig.add_shape(
            type="line", xref="x", yref="y",
            x0=x_start, x1=x_end, y0=price, y1=price,
            line=dict(color=color, width=1.5, dash=dash),
        )
        fig.add_annotation(
            x=x_end, y=price, text=label,
            showarrow=False,
            font=dict(color=color, size=11, family="Arial"),
            bgcolor="rgba(11,14,19,0.85)",
            bordercolor=color, borderwidth=1, borderpad=3,
            xanchor="left", yanchor="middle", xshift=6,
        )

    if _in_range(entry):
        _hline(float(entry), entry_color, "solid", f"🟢 entry {entry:.4f}")
    if _in_range(stop):
        _hline(float(stop), short_color, "dash", f"❌ stop {stop:.4f}")
    if _in_range(target):
        _hline(float(target), long_color, "dot", f"🎯 target {target:.4f}")

    # When action=wait (no entry/stop/target), still extract invalidation
    # levels from the text — Claude often mentions concrete prices like
    # "при закрытии выше 2454" or "при пробое 2236" in invalidation_explained.
    inv_text = (opinion.get("invalidation_explained") or "")
    if inv_text and not _in_range(entry):
        import re
        # Find float numbers in the invalidation prose (basic 4-digit-or-decimal)
        nums = re.findall(r"\d+\.?\d*", inv_text)
        seen = set()
        for n in nums[:4]:
            try:
                v = float(n)
            except ValueError:
                continue
            if v in seen or not _in_range(v):
                continue
            # Skip year-like numbers, percentages, small ints
            if v < 1.0 or v > visible_hi * 2 or len(n) <= 2:
                continue
            seen.add(v)
            fig.add_shape(
                type="line", xref="x", yref="y",
                x0=x_start, x1=x_end, y0=v, y1=v,
                line=dict(color=hex_to_rgba(ACCENT_CYAN, 0.45),
                          width=1, dash="dashdot"),
            )
            fig.add_annotation(
                x=x_end, y=v, text=f"⚠️ inv {v:.4f}",
                showarrow=False,
                font=dict(color=ACCENT_CYAN, size=10),
                bgcolor="rgba(11,14,19,0.85)",
                bordercolor=hex_to_rgba(ACCENT_CYAN, 0.5),
                borderwidth=1, borderpad=2,
                xanchor="left", yanchor="middle", xshift=6,
            )

    # Corner annotation — bias + setup quality + structure position
    bias_emoji = {"long": "🟢", "short": "🔴",
                  "neutral": "⚪️"}.get(bias, "⚪️")
    action_text = {"long": "ЛОНГ", "short": "ШОРТ",
                   "wait": "ЖДЁМ", "close": "ЗАКР"}.get(action, action.upper())
    structure_pos = opinion.get("structure_position") or ""
    if structure_pos and len(structure_pos) > 140:
        structure_pos = structure_pos[:137] + "…"

    header = f"<b>{bias_emoji} CLAUDE: {action_text}</b> · setup: {sq}"
    body = (f"<br><span style='font-size:10px'>{structure_pos}</span>"
            if structure_pos else "")
    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.99,
        text=header + body,
        showarrow=False,
        align="left",
        bgcolor="rgba(11,14,19,0.85)",
        bordercolor=entry_color, borderwidth=1, borderpad=6,
        font=dict(color="#E6EDF3", size=11),
        xanchor="left", yanchor="top",
    )

    # ----------------------------------------------------------------------
    # Phase 15 — structured chart_annotations from Claude. Each entry is a
    # typed dict (pivot/fib_level/zone/arrow/wave_label/note) targeting a
    # specific TF. We draw only ones matching this TF (or 'all').
    # ----------------------------------------------------------------------
    color_map = {
        "red":     short_color,
        "green":   long_color,
        "blue":    "#4493F8",
        "cyan":    ACCENT_CYAN,
        "yellow":  "#D29922",
        "gray":    TEXT_MUTED,
        "info":    ACCENT_CYAN,
        "warning": "#D29922",
        "danger":  short_color,
    }

    def _parse_date(d):
        if not d:
            return None
        try:
            return pd.Timestamp(d)
        except Exception:
            return None

    def _date_in_range(d):
        if d is None:
            return False
        try:
            return prices.index[0] <= d <= prices.index[-1]
        except Exception:
            return False

    annotations = opinion.get("chart_annotations") or []
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        a_tf = str(ann.get("tf", "all")).lower()
        if a_tf not in ("all", tf_name.lower()):
            continue

        a_type = str(ann.get("type", "")).lower()
        a_color = color_map.get(str(ann.get("color", "cyan")).lower(),
                                 ACCENT_CYAN)
        a_label = str(ann.get("label") or "")[:40]

        if a_type == "pivot":
            price = ann.get("price")
            if not _in_range(price):
                continue
            d = _parse_date(ann.get("date"))
            if not _date_in_range(d):
                # If date unknown — place at the right edge so the marker
                # is still visible. Otherwise use the parsed date.
                d = prices.index[-1]
            fig.add_trace(go.Scatter(
                x=[d], y=[float(price)],
                mode="markers+text",
                marker=dict(symbol="diamond", size=12, color=a_color,
                            line=dict(color="#0B0E13", width=1.5)),
                text=[a_label],
                textposition="top center",
                textfont=dict(color=a_color, size=10),
                showlegend=False, hoverinfo="text",
                hovertext=f"PIVOT {a_label}: {price}",
            ))

        elif a_type == "fib_level":
            price = ann.get("price")
            if not _in_range(price):
                continue
            fig.add_shape(
                type="line", xref="x", yref="y",
                x0=x_start, x1=x_end, y0=float(price), y1=float(price),
                line=dict(color=hex_to_rgba(a_color, 0.55),
                          width=1, dash="dot"),
            )
            fig.add_annotation(
                x=x_end, y=float(price),
                text=f"📐 {a_label} · {price}",
                showarrow=False,
                font=dict(color=a_color, size=10),
                bgcolor="rgba(11,14,19,0.85)",
                bordercolor=hex_to_rgba(a_color, 0.5),
                borderwidth=1, borderpad=2,
                xanchor="left", yanchor="middle", xshift=6,
            )

        elif a_type == "zone":
            lo = ann.get("price_low")
            hi = ann.get("price_high")
            if not (_in_range(lo) or _in_range(hi)):
                continue
            try:
                y0 = min(float(lo), float(hi))
                y1 = max(float(lo), float(hi))
            except (TypeError, ValueError):
                continue
            fig.add_shape(
                type="rect", xref="x", yref="y",
                x0=x_start, x1=x_end, y0=y0, y1=y1,
                fillcolor=hex_to_rgba(a_color, 0.12),
                line=dict(color=hex_to_rgba(a_color, 0.35),
                          width=1, dash="dot"),
                layer="below",
            )
            fig.add_annotation(
                x=x_end, y=(y0 + y1) / 2,
                text=f"▥ {a_label}",
                showarrow=False,
                font=dict(color=a_color, size=10),
                bgcolor="rgba(11,14,19,0.85)",
                bordercolor=hex_to_rgba(a_color, 0.5),
                borderwidth=1, borderpad=2,
                xanchor="left", yanchor="middle", xshift=6,
            )

        elif a_type == "arrow":
            fp = ann.get("from_price")
            tp = ann.get("to_price")
            if not (_in_range(fp) and _in_range(tp)):
                continue
            fd = _parse_date(ann.get("from_date"))
            td = _parse_date(ann.get("to_date"))
            # Fallback: span the full visible range when dates are missing
            if not _date_in_range(fd):
                fd = prices.index[max(0, len(prices) - 60)]
            if not _date_in_range(td):
                td = prices.index[-1]
            fig.add_annotation(
                x=td, y=float(tp),
                ax=fd, ay=float(fp),
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=2, arrowsize=1.2, arrowwidth=2,
                arrowcolor=a_color,
                text=f"<b>{a_label}</b>" if a_label else "",
                font=dict(color=a_color, size=10),
                bgcolor="rgba(11,14,19,0.85)" if a_label else None,
                bordercolor=a_color if a_label else None,
                borderwidth=1 if a_label else 0, borderpad=2,
            )

        elif a_type == "wave_label":
            price = ann.get("price")
            if not _in_range(price):
                continue
            d = _parse_date(ann.get("date"))
            if not _date_in_range(d):
                d = prices.index[-1]
            fig.add_annotation(
                x=d, y=float(price),
                text=f"<b>{a_label}</b>",
                showarrow=False,
                bgcolor=hex_to_rgba(a_color, 0.85),
                font=dict(color="#0B0E13", size=12),
                bordercolor=a_color, borderwidth=1, borderpad=3,
                xanchor="center", yanchor="middle",
            )

        elif a_type == "note":
            price = ann.get("price")
            if not _in_range(price):
                continue
            fig.add_shape(
                type="line", xref="x", yref="y",
                x0=x_start, x1=x_end, y0=float(price), y1=float(price),
                line=dict(color=hex_to_rgba(a_color, 0.45),
                          width=1, dash="dashdot"),
            )
            fig.add_annotation(
                x=x_end, y=float(price),
                text=f"📝 {a_label}",
                showarrow=False,
                font=dict(color=a_color, size=10),
                bgcolor="rgba(11,14,19,0.85)",
                bordercolor=hex_to_rgba(a_color, 0.5),
                borderwidth=1, borderpad=2,
                xanchor="left", yanchor="middle", xshift=6,
            )


for r in results:
    if not r.ok:
        st.error(f"{r.name}: {r.error}")
        continue

    fig_main = _plot_tf(r)

    # Phase 14 — paint Claude's plan on top of every TF, if available
    if show_claude_overlay:
        _opinion = st.session_state.get("last_holistic_full")
        # Only show for the ticker that the opinion was about
        if _opinion and _opinion.get("ticker") == ticker:
            _overlay_claude_opinion(fig_main, _opinion, r.prices, r.name)

    st.plotly_chart(fig_main, use_container_width=True)

    # Phase 16 — show the TV screenshot with Claude annotations side-by-side
    # if we made one. We don't auto-render the raw TV screenshot — that's
    # available on the Visual Compare page. Here we only surface the
    # annotated version because the Plotly chart above already covers the
    # un-annotated view.
    _holistic_full = st.session_state.get("last_holistic_full")
    if (_holistic_full and _holistic_full.get("ticker") == ticker
            and _holistic_full.get("annotated_screenshots")):
        _ann_path = _holistic_full["annotated_screenshots"].get(r.name)
        if _ann_path and os.path.isfile(_ann_path):
            with st.expander(
                f"📌 {r.name} — TV-скриншот с разметкой Claude",
                expanded=False,
            ):
                st.image(_ann_path, use_container_width=True,
                          caption=f"Claude annotations on TV chart · {r.name}")

    # Zoom-in expander — useful on 1d/4h where the labeled structure is
    # tiny relative to the full lookback.
    n_recent = ZOOM_BARS.get(r.name, 150)
    if len(r.prices) > n_recent:
        with st.expander(
            f"🔍 {r.name} — крупным планом (последние {n_recent} баров + прогноз)"
        ):
            fig_zoom = _zoom_plot(r, n_recent)
            if show_claude_overlay:
                _opinion = st.session_state.get("last_holistic_full")
                if _opinion and _opinion.get("ticker") == ticker:
                    zoomed_prices = r.prices.iloc[-n_recent:] \
                        if len(r.prices) > n_recent else r.prices
                    _overlay_claude_opinion(fig_zoom, _opinion,
                                              zoomed_prices, r.name)
            st.plotly_chart(fig_zoom, use_container_width=True)

    # Forecast card — what comes next after the labeled structure.
    fc = r.forecast
    if fc is not None:
        pattern_label = _RU.get(
            f"forecast.pattern.{fc.next_pattern}",
            fc.next_pattern.replace("_", " "),
        )
        dir_label = _RU.get(f"forecast.dir.{fc.direction}", fc.direction)

        # Border color by direction.
        fc_color = (
            REGIME_COLORS["Low Vol"]  if fc.direction == "up"
            else REGIME_COLORS["High Vol"] if fc.direction == "down"
            else TEXT_MUTED
        )

        # Top row — pattern, direction, confidence.
        f1, f2, f3 = st.columns(3)
        with f1:
            st.markdown(
                metric_card(
                    title=t("forecast.pattern"),
                    value=pattern_label,
                    subtitle=fc.next_pattern,
                    border_color=fc_color,
                ),
                unsafe_allow_html=True,
            )
        with f2:
            duration_str = "—"
            if fc.expected_duration_bars is not None:
                duration_str = t(
                    "forecast.duration_bars",
                    lo=fc.expected_duration_bars[0],
                    hi=fc.expected_duration_bars[1],
                )
            st.markdown(
                metric_card(
                    title=t("forecast.direction"),
                    value=dir_label,
                    subtitle=f"{t('forecast.duration')}: {duration_str}",
                    border_color=fc_color,
                ),
                unsafe_allow_html=True,
            )
        with f3:
            inv_value = (f"{fc.invalidation_level:.4f}"
                          if fc.invalidation_level is not None else "—")
            st.markdown(
                metric_card(
                    title=t("forecast.invalidation"),
                    value=inv_value,
                    subtitle=f"{t('forecast.confidence')}: {fc.confidence}",
                    border_color=REGIME_COLORS["High Vol"],
                ),
                unsafe_allow_html=True,
            )

        # Targets table.
        if fc.targets:
            tgt_df = pd.DataFrame([
                {
                    "Уровень": tg.fib_label,
                    "Цена":    f"{tg.price:.4f}",
                    "Что это": tg.label,
                    "Вероятность": tg.probability,
                }
                for tg in fc.targets
            ])
            st.dataframe(tgt_df, use_container_width=True, hide_index=True)

        with st.expander(t("forecast.rationale")):
            st.write(fc.rationale)
            if fc.invalidation_reason:
                st.markdown(f"**Условие отмены:** {fc.invalidation_reason}")

    top = r.top
    if top is None:
        continue

    with st.expander(f"{r.name} — full analysis (alternatives, rules, levels)"):
        # Ranked candidates table.
        st.markdown("**Ранжированные гипотезы**")
        cand_df = pd.DataFrame([
            {
                "Паттерн":       translate_pattern_slug(c.pattern),
                "Score":         f"{c.score:.0f}",
                "Нарушений":     len(c.rule_violations),
                "Текущая позиция": translate_detector_text(c.current_position or ""),
            }
            for c in r.candidates
        ])
        st.dataframe(cand_df, use_container_width=True, hide_index=True)

        if top.rule_violations:
            st.markdown("**Rule violations (top hypothesis):**")
            for v in top.rule_violations:
                st.markdown(f"- ⚠️ {v}")
        if top.guideline_notes:
            st.markdown("**Guideline fits:**")
            for n in top.guideline_notes:
                st.markdown(f"- {n}")
        if top.fibs:
            st.markdown("**Fibonacci retracement levels (from leg 0→1):**")
            fib_df = pd.DataFrame(
                [{"Level": k, "Price": f"{v:.4f}"} for k, v in top.fibs.items()]
            )
            st.dataframe(fib_df, use_container_width=True, hide_index=True)
        if top.next_targets:
            st.markdown("**Projected next targets:**")
            tgt_df = pd.DataFrame(
                [{"Target": k, "Price": f"{v:.4f}"} for k, v in top.next_targets.items()]
            )
            st.dataframe(tgt_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Detector quality — backtest on this ticker's full history
# ---------------------------------------------------------------------------
section_header(t("section.backtest"))
st.caption(
    "For every completed 5-wave impulse the detector found in the past, "
    "we check whether the next "
    f"bars moved counter-trend by ≥ 5% (the Elliott prediction)."
)

val_tfs = ["1d", "4h", "15m"]
if include_5m:
    val_tfs.append("5m")
val_cols = st.columns(len(val_tfs))
for col, tf_name in zip(val_cols, val_tfs):
    tf = next((r for r in results if r.name == tf_name and r.ok), None)
    if tf is None or tf.prices.empty:
        with col:
            st.markdown(
                metric_card(
                    title=f"{tf_name} backtest",
                    value="—",
                    subtitle="no data",
                    border_color=TEXT_MUTED,
                ),
                unsafe_allow_html=True,
            )
        continue
    report = validate_detector(
        tf.prices,
        threshold_pct=tf.major_threshold_pct,
        lookahead_bars=20,
        move_threshold_pct=0.05,
        min_score=50.0,
    )
    if report.total_decisive == 0:
        with col:
            st.markdown(
                metric_card(
                    title=f"{tf_name} backtest",
                    value=f"{report.n_cases} cases",
                    subtitle="not enough decisive moves",
                    border_color=TEXT_MUTED,
                ),
                unsafe_allow_html=True,
            )
        continue
    # Colour: green if hit rate > 60%, red if < 40%, grey otherwise.
    color = (
        REGIME_COLORS["Low Vol"] if report.hit_rate >= 0.60 else
        REGIME_COLORS["High Vol"] if report.hit_rate <= 0.40 else
        TEXT_MUTED
    )
    with col:
        st.markdown(
            metric_card(
                title=f"{tf_name} hit rate",
                value=f"{report.hit_rate:.0%}",
                subtitle=(
                    f"{report.n_hits} hits / {report.n_misses} misses / "
                    f"{report.n_ambiguous} ambiguous"
                ),
                border_color=color,
            ),
            unsafe_allow_html=True,
        )
st.caption(
    f"**{t('backtest.hit_signal_real')}.** "
    f"**{t('backtest.hit_noise')}.** "
    f"**{t('backtest.hit_backwards')}** (попробуй другой threshold)."
)


# ---------------------------------------------------------------------------
# Claude second opinion — optional, requires API key
# ---------------------------------------------------------------------------
section_header(t("section.claude_opinion"))

if not is_available():
    st.caption(f"⚪️ {availability_message()}")
else:
    st.caption(f"🟢 {availability_message()}")

    # NEW: holistic top-down opinion — one request for all 4 TFs.
    st.markdown("**Holistic top-down — все ТФ в одном запросе:**")

    _hb1, _hb2 = st.columns([2, 1])
    with _hb1:
        _do_holistic = st.button(
            "🧠 Получить holistic мнение Claude (все ТФ)",
            type="primary", key="holistic_btn", use_container_width=True,
        )
    with _hb2:
        # Re-run only screenshot capture + annotation overlay (no new Claude
        # call — uses the last cached opinion). Useful when TV bridge wasn't
        # available during the original holistic call, but is now.
        _can_retry = bool(
            st.session_state.get("last_holistic_full")
            and st.session_state.get("last_holistic_full", {}).get("ticker") == ticker
            and st.session_state.get("last_holistic_pid")
        )
        _do_retry_capture = st.button(
            "🔄 Повторить захват TV + аннотации",
            disabled=not _can_retry,
            key="retry_capture_btn", use_container_width=True,
            help=("Берёт уже сохранённый holistic-ответ Claude и пытается снова "
                  "захватить TV-скриншоты + наложить аннотации. Не делает новый "
                  "вызов к Claude — экономит токены. Используй после того как "
                  "запустил tv_bridge.py или подкрутил TV layout."),
        )

    # Handle retry-capture branch first (it doesn't need the heavy holistic
    # request flow below).
    if _do_retry_capture and _can_retry:
        _hf = st.session_state["last_holistic_full"]
        _pid = st.session_state["last_holistic_pid"]
        _annotations = _hf.get("chart_annotations") or []
        if not _annotations:
            st.warning("В сохранённом ответе Claude нет chart_annotations — "
                       "нечего наносить. Сделай новый holistic-запрос.")
        else:
            with st.spinner("Повторный захват TV-скриншотов…"):
                # Force re-capture for every TF (delete old + try fresh).
                # Pass prices/swings so the Plotly fallback can render even
                # when TV bridge and chart-img.com both fail.
                try:
                    from tv_screenshot import capture
                    for _r in results:
                        if _r.ok:
                            capture(f"{_pid}_{_r.name}", ticker, _r.name,
                                     force=True,
                                     prices=_r.prices,
                                     major_swings=_r.major_swings,
                                     minor_swings=_r.minor_swings)
                except Exception as exc:
                    st.caption(f"⚠️ Re-capture failed: "
                                f"{type(exc).__name__}: {exc}")
                # Re-run annotation pipeline
                try:
                    from tv_annotate import annotate_for_holistic
                    ohlc_by_tf = {_r.name: _r.prices for _r in results
                                   if _r.ok and len(_r.prices) > 0}
                    new_paths = annotate_for_holistic(
                        prediction_id=_pid,
                        chart_annotations=_annotations,
                        ohlc_by_tf=ohlc_by_tf,
                        layout_name=tv_layout_name,
                    )
                except Exception as exc:
                    new_paths = {}
                    st.caption(f"⚠️ Annotation failed: "
                                f"{type(exc).__name__}: {exc}")
            if new_paths:
                _hf["annotated_screenshots"] = new_paths
                st.session_state["last_holistic_full"] = _hf
                missing = (set([_r.name for _r in results if _r.ok])
                            - set(new_paths.keys()))
                msg = (f"📌 Повторная аннотация удалась для: "
                       f"{', '.join(new_paths.keys())}")
                if missing:
                    msg += f"  ·  пропущены: {', '.join(sorted(missing))}"
                st.success(msg)
                st.info("Чтобы увидеть обновлённые скрины — раскрой "
                        "expander '📌 {tf} — TV-скриншот с разметкой "
                        "Claude' под каждым графиком ниже.")
            else:
                st.error(
                    "Не удалось захватить ни одного TV-скриншота. Проверь:\n"
                    "  • `ps aux | grep tv_bridge` — bridge запущен?\n"
                    "  • `curl http://localhost:9222/json/version` — TV "
                    "Desktop открыт с debug-портом?\n"
                    "  • Сеть к chart-img.com доступна (fallback)?\n"
                    f"  • predictions/screenshots/ имеет права на запись?"
                )

    # CRITICAL guard — after strict lazy mode, user may change ticker in
    # sidebar WITHOUT clicking Run Analysis. Then `results` still hold the
    # PREVIOUS ticker's data, but `ticker` variable reflects the NEW one,
    # so the holistic call would feed Claude old data while logging the
    # new ticker. Track Record ends up with phantom predictions for tickers
    # that never actually got analyzed.
    if _do_holistic:
        _cached_ticker_now = (_cached_key[0]
                              if _cached_key and len(_cached_key) > 0
                              else None)
        if _cached_ticker_now and _cached_ticker_now != ticker:
            st.error(
                f"🚫 **Ticker mismatch** — графики и swings всё ещё для "
                f"`{_cached_ticker_now}`, а в sidebar выбран `{ticker}`. "
                f"Holistic-запрос **отменён** чтобы не записать ложный "
                f"прогноз в Track Record (Claude получил бы данные одного "
                f"тикера, а сохранили бы как другой).\n\n"
                f"**Что сделать:** нажми **Run Analysis** в sidebar для "
                f"{ticker}, потом снова нажми «Получить holistic мнение»."
            )
            _do_holistic = False

    # Guard against accidental double-fire AND wasteful re-runs within a
    # short window. Two layers of protection:
    #   1. Per-(ticker, 10-minute-bucket) lock — if same ticker was already
    #      analyzed within last 10 minutes, refuse new Claude call and
    #      surface the persisted result instead.
    #   2. Content-hash dedup — if the OHLC fingerprint of the swings is
    #      identical to the last call's fingerprint, also refuse.
    if _do_holistic:
        # 10-minute bucket: floor minute to nearest 10
        _now = pd.Timestamp.utcnow()
        _bucket = f"{_now.strftime('%Y%m%dT%H')}{_now.minute // 10}0"
        _lock_key = f"holistic_inflight_{ticker}_{_bucket}"

        # Build content hash from current results (each TF swings count + last swing price)
        _content_parts = [ticker]
        for _r in results:
            if _r.ok and _r.major_swings:
                _last = _r.major_swings[-1]
                _content_parts.append(
                    f"{_r.name}:{len(_r.major_swings)}:{_last.price:.6f}"
                )
        _content_hash = hash(tuple(_content_parts))
        _last_hash_key = f"last_holistic_content_hash_{ticker}"

        if st.session_state.get(_lock_key):
            # Already running in this 10-min bucket
            st.warning(
                "⏳ Holistic-запрос для этого тикера уже был сделан в последние "
                "10 минут. Повторный вызов заблокирован чтобы не тратить токены. "
                "Подожди или нажми «🔄 Повторить захват TV + аннотации» — "
                "она использует уже сохранённый ответ."
            )
            _do_holistic = False
        elif (st.session_state.get(_last_hash_key) == _content_hash
              and st.session_state.get("last_holistic_full", {}).get("ticker") == ticker):
            # Same content as last successful call — point at the cached one
            st.info(
                "💾 Данные не изменились с прошлого holistic-вызова для этого "
                "тикера. Использую сохранённый ответ. Если нужно принудительно "
                "повторить — измени один из threshold-слайдеров в sidebar."
            )
            _do_holistic = False
        else:
            st.session_state[_lock_key] = True
            st.session_state[_last_hash_key] = _content_hash

    if _do_holistic:
        # Phase 18 — capture TV screenshots BEFORE asking Claude, so the
        # annotated screenshot matches the same moment in time as the
        # analysis. We pre-compute the future prediction_id deterministically
        # from (ticker, minute) so log_holistic finds the screenshots later.
        pre_capture_paths: dict[str, str] = {}
        if make_tv_annotations:
            with st.spinner("Захватываем TV-скриншоты перед анализом…"):
                try:
                    from tv_screenshot import capture
                    # Use a temp prefix; we'll rename to the real pid after
                    # log_holistic returns. Temp id = ticker+minute hash.
                    import hashlib
                    _temp_pid = "pre_" + hashlib.md5(
                        f"{ticker}|{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M')}"
                        .encode()).hexdigest()[:12]
                    for _r in results:
                        if _r.ok:
                            _p = capture(f"{_temp_pid}_{_r.name}", ticker,
                                          _r.name, force=True,
                                          prices=_r.prices,
                                          major_swings=_r.major_swings,
                                          minor_swings=_r.minor_swings)
                            if _p:
                                pre_capture_paths[_r.name] = _p
                    if pre_capture_paths:
                        st.caption(
                            f"📸 Захвачено TV-скриншотов: "
                            f"{', '.join(pre_capture_paths.keys())} "
                            f"(до анализа Claude)"
                        )
                    st.session_state["_temp_screenshot_pid"] = _temp_pid
                except Exception as _e:
                    st.caption(f"⚠️ Pre-capture: {type(_e).__name__}: {_e}")

        tf_blocks = [
            {
                "name": r.name,
                "swings": r.major_swings,
                "candidates": r.candidates,
                "forecast": r.forecast,
            }
            for r in results if r.ok and r.candidates
        ]
        if not tf_blocks:
            st.warning("Нет данных ни одного ТФ для holistic анализа.")
        else:
            with st.spinner("Top-down синтез Claude…"):
                # Phase 41 — pull previous holistic from persistent store
                _prev_entry = load_last_holistic(ticker)
                _prev_holistic = (_prev_entry.get("opinion")
                                   if isinstance(_prev_entry, dict) else None)
                # Phase 43 — verify track record of past predictions vs real
                # OHLC, so Claude sees which previous setups hit / stopped.
                _track = []
                try:
                    from datetime import datetime, timezone
                    # ohlc_fetcher: pull recent 5m bars since the given datetime
                    def _ohlc_fetcher(since_dt):
                        from data import get_ohlc  # type: ignore
                        df = get_ohlc(ticker, period="30d", interval="5m")
                        if df is None or df.empty:
                            return None
                        # Slice from since_dt onwards
                        try:
                            mask = df.index >= since_dt
                            return df.loc[mask]
                        except Exception:
                            return df
                    _track = verify_history(ticker, _ohlc_fetcher, max_n=5)
                except Exception:
                    _track = load_history(ticker, max_n=5)
                try:
                    h = ask_claude_holistic(ticker, tf_blocks,
                                             prev_holistic=_prev_holistic,
                                             track_record=_track)
                except ClaudeSpendLimitError as exc:
                    st.warning(
                        f"💸 **Лимит расходов Anthropic API исчерпан.**\n\n"
                        f"{exc}\n\n"
                        f"Пока лимит не восстановится, можешь использовать "
                        f"rule-engine прогноз (вкладка «What's next» по каждому "
                        f"ТФ) и Track Record страницу — они работают без Claude."
                    )
                    h = None
                except Exception as exc:
                    st.error(f"Ошибка holistic Claude: "
                             f"{type(exc).__name__}: {exc}")
                    h = None
            if h is not None:
                # Persist for later track-record analysis. Use 1d-anchored price
                # as the "snapshot price" for the holistic call.
                snapshot_price = None
                for r in results:
                    if r.ok and r.name == "1d" and len(r.prices):
                        snapshot_price = float(r.prices.iloc[-1])
                        break
                if snapshot_price is None:
                    for r in results:
                        if r.ok and len(r.prices):
                            snapshot_price = float(r.prices.iloc[-1])
                            break
                pid_for_screens = None
                try:
                    _logged = log_holistic(ticker, h,
                                  current_price=snapshot_price or 0.0)
                    # Capture the REAL prediction_id that log_holistic
                    # generated — tv_screenshot.capture saves files using
                    # this exact id, so we must reuse it (not recompute).
                    if isinstance(_logged, dict):
                        pid_for_screens = _logged.get("id")
                except Exception as exc:
                    # Never let logging break the UI
                    st.warning(f"Не удалось записать предсказание в журнал: "
                                f"{type(exc).__name__}: {exc}")

                # Phase 18 — rename pre-captured screenshots from temp pid
                # to the real prediction_id so annotate_for_holistic can
                # find them. This avoids re-capturing (saves 4 × ~30s).
                _temp_pid = st.session_state.pop("_temp_screenshot_pid", None)
                if _temp_pid and pid_for_screens and pre_capture_paths:
                    try:
                        for _tf, _src in pre_capture_paths.items():
                            if os.path.isfile(_src):
                                _dst = _src.replace(_temp_pid, pid_for_screens)
                                if _src != _dst:
                                    try:
                                        if os.path.isfile(_dst):
                                            os.remove(_dst)
                                        os.rename(_src, _dst)
                                    except OSError:
                                        pass
                    except Exception:
                        pass

                # Save the pid into full state so the "🔄 Повторить захват"
                # button can find it later without re-doing the holistic call.
                if pid_for_screens:
                    st.session_state["last_holistic_pid"] = pid_for_screens

                # Cache for the «Журнал действий» block: prefill entry/stop/target
                # when the user is about to log a "took the setup" action.
                st.session_state["last_holistic_for_action"] = {
                    "ticker":    ticker,
                    "bias":      h.bias,
                    "action":    h.action,
                    "entry":     h.entry,
                    "stop_loss": h.stop_loss,
                    "target":    h.target,
                }

                # Phase 16 — paint Claude's structured chart_annotations on
                # top of the TV screenshots we just captured (via log_holistic).
                # PIL overlay produces predictions/screenshots_annotated/
                # {pid}_{tf}_annotated.png that's shown side-by-side later.
                annotated_paths: dict[str, str] = {}
                try:
                    if (make_tv_annotations
                            and getattr(h, "chart_annotations", [])
                            and pid_for_screens):
                        from tv_annotate import annotate_for_holistic
                        # Build per-TF OHLC windows from the loaded results
                        ohlc_by_tf = {}
                        for _r in results:
                            if _r.ok and len(_r.prices) > 0:
                                ohlc_by_tf[_r.name] = _r.prices
                        annotated_paths = annotate_for_holistic(
                            prediction_id=pid_for_screens,
                            chart_annotations=h.chart_annotations,
                            ohlc_by_tf=ohlc_by_tf,
                            layout_name=tv_layout_name,
                        )
                        if annotated_paths:
                            missing = (set(ohlc_by_tf.keys())
                                        - set(annotated_paths.keys()))
                            msg = (f"📌 Claude-аннотации нанесены на TV-скриншоты: "
                                   f"{', '.join(annotated_paths.keys())}")
                            if missing:
                                msg += (f"  ·  пропущены (нет скрина): "
                                        f"{', '.join(sorted(missing))}")
                            st.success(msg)
                        else:
                            st.caption(
                                f"⚠️ Claude-аннотации не нанесены ни на один "
                                f"TV-скриншот. Проверь что TV bridge запущен "
                                f"и в predictions/screenshots/ есть файлы "
                                f"{pid_for_screens}_*.png"
                            )
                except Exception as _ann_exc:
                    st.caption(
                        f"⚠️ tv_annotate failed: "
                        f"{type(_ann_exc).__name__}: {_ann_exc}"
                    )

                # Phase 14 — full opinion for chart overlay. Stored alongside
                # the action-cache so the _overlay_claude_opinion() helper can
                # paint Claude's levels/structure-notes onto every TF chart.
                st.session_state["last_holistic_full"] = {
                    "ticker":                  ticker,
                    "bias":                    h.bias,
                    "action":                  h.action,
                    "setup_quality":           h.setup_quality,
                    "entry":                   h.entry,
                    "stop_loss":               h.stop_loss,
                    "target":                  h.target,
                    "risk_reward":             h.risk_reward,
                    "summary":                 h.summary,
                    "structure_position":      getattr(h, "structure_position", ""),
                    "invalidation_explained":  h.invalidation_explained,
                    "rules_check":             getattr(h, "rules_check", ""),
                    "fibonacci_check":         getattr(h, "fibonacci_check", ""),
                    "alternative_interpretation": getattr(h, "alternative_interpretation", ""),
                    "per_tf_notes":            h.per_tf_notes or {},
                    "chart_annotations":       getattr(h, "chart_annotations", []),
                    "annotated_screenshots":   annotated_paths,
                    "scalp_plan":              getattr(h, "scalp_plan", {}),
                    "higher_degree_context":   getattr(h, "higher_degree_context", {}),
                    "monitoring_levels":       getattr(h, "monitoring_levels", []),
                    "entry_alternatives":      getattr(h, "entry_alternatives", []),
                    "target_pyramid":          getattr(h, "target_pyramid", []),
                    "cross_correlation_notes": getattr(h, "cross_correlation_notes", ""),
                    "scenarios":               getattr(h, "scenarios", []),
                    "waiting_conditions":      getattr(h, "waiting_conditions", []),
                    "invalidation_of_previous": getattr(h, "invalidation_of_previous", ""),
                    "cross_asset_comparison":  getattr(h, "cross_asset_comparison", []),
                    "leader_follower_note":    getattr(h, "leader_follower_note", ""),
                    "action_plan":             getattr(h, "action_plan", []),
                    "entry_zone_shift":        getattr(h, "entry_zone_shift", ""),
                    "entry_variants":          getattr(h, "entry_variants", []),
                    "track_record":            _track,
                    "was_truncated":           getattr(h, "was_truncated", False),
                    "structural_events_timeline": getattr(h, "structural_events_timeline", []),
                    "invalidation_layers":     getattr(h, "invalidation_layers", {}),
                    "gut_probability_next_window": getattr(h, "gut_probability_next_window", {}),
                    "period_context":          getattr(h, "period_context", {}),
                    "tf_analysis":             getattr(h, "tf_analysis", {}),
                    "verdict_per_tf":          getattr(h, "verdict_per_tf", {}),
                    "confluence_factors":      getattr(h, "confluence_factors", []),
                    "comparison_with_previous": getattr(h, "comparison_with_previous", {}),
                    "monitor_next_hours":      getattr(h, "monitor_next_hours", {}),
                    "critical_level":          getattr(h, "critical_level", ""),
                    "golden_entry_zone":       getattr(h, "golden_entry_zone", ""),
                }

                # Phase 20 — persist to disk so the answer survives page
                # navigation / Streamlit hard-reload. Auto-loaded on next
                # page entry for this ticker.
                try:
                    save_last_holistic(
                        ticker,
                        st.session_state["last_holistic_full"],
                        pid=pid_for_screens,
                    )
                except Exception:
                    pass

                # Phase 43 — append to track-record history so next holistic
                # call can show "v1 hit, v2 stopped, v3 hit" verification.
                try:
                    save_holistic_entry(
                        ticker,
                        st.session_state["last_holistic_full"],
                    )
                except Exception:
                    pass

                # Phase 23 — rerun so the Plotly charts (rendered EARLIER
                # in the script) pick up the new entry/stop/target/
                # chart_annotations from session_state. The banner+cards+
                # chain-of-thought displayed AFTER this block are now
                # rendered from session_state in a separate block below
                # (see _render_holistic_banner_from_state), so the rerun
                # doesn't lose them.
                st.rerun()

                # Big verdict banner.
                bias_color = {
                    "long":    REGIME_COLORS["Low Vol"],
                    "short":   REGIME_COLORS["High Vol"],
                    "neutral": TEXT_MUTED,
                }.get(h.bias, TEXT_MUTED)
                bias_emoji = {"long": "🟢", "short": "🔴",
                              "neutral": "⚪️"}.get(h.bias, "⚪️")
                st.markdown(
                    f'<div style="padding:14px 18px; '
                    f'border-left: 4px solid {bias_color}; '
                    f'background:#161B22; border-radius: 8px;">'
                    f'<div style="font-size:11px; color:#8B949E; '
                    f'text-transform:uppercase; letter-spacing:.08em; '
                    f'margin-bottom:6px;">Top-down вердикт Claude '
                    f'(качество сетапа: {h.setup_quality})</div>'
                    f'<div style="font-size:20px; font-weight:700; '
                    f'color:{bias_color};">{bias_emoji} BIAS: '
                    f'{h.bias.upper()} · action: {h.action.upper()}</div>'
                    f'<div style="font-size:14px; color:#E6EDF3; '
                    f'margin-top:8px;">{h.summary}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Trade plan — 4 cards.
                hp1, hp2, hp3, hp4 = st.columns(4)
                with hp1:
                    st.markdown(
                        metric_card(
                            title=t("claude.entry"),
                            value=(f"{h.entry:.4f}" if h.entry is not None
                                    else "—"),
                            subtitle="", border_color=bias_color,
                        ),
                        unsafe_allow_html=True,
                    )
                with hp2:
                    st.markdown(
                        metric_card(
                            title=t("claude.stop"),
                            value=(f"{h.stop_loss:.4f}"
                                    if h.stop_loss is not None else "—"),
                            subtitle="",
                            border_color=REGIME_COLORS["High Vol"],
                        ),
                        unsafe_allow_html=True,
                    )
                with hp3:
                    st.markdown(
                        metric_card(
                            title=t("claude.target"),
                            value=(f"{h.target:.4f}"
                                    if h.target is not None else "—"),
                            subtitle="",
                            border_color=REGIME_COLORS["Low Vol"],
                        ),
                        unsafe_allow_html=True,
                    )
                with hp4:
                    rr_value = (f"{h.risk_reward:.2f}"
                                 if h.risk_reward is not None else "—")
                    rr_color = (REGIME_COLORS["Low Vol"]
                                if (h.risk_reward or 0) >= 1.5
                                else REGIME_COLORS["High Vol"]
                                if (h.risk_reward or 0) < 1
                                else ACCENT_CYAN)
                    st.markdown(
                        metric_card(
                            title="R/R",
                            value=rr_value,
                            subtitle="≥1.5 — хороший сетап",
                            border_color=rr_color,
                        ),
                        unsafe_allow_html=True,
                    )

                # Per-TF notes
                if h.per_tf_notes:
                    st.markdown("**Заметки по таймфреймам:**")
                    for tf_name, note in h.per_tf_notes.items():
                        st.markdown(f"- **{tf_name}** — {note}")

                if h.invalidation_explained:
                    st.markdown(f"**Условие отмены плана:** "
                                f"{h.invalidation_explained}")

                # Phase 13 — chain-of-thought self-analysis
                cot_present = any([
                    getattr(h, "structure_position", ""),
                    getattr(h, "rules_check", ""),
                    getattr(h, "alternation_check", ""),
                    getattr(h, "fibonacci_check", ""),
                    getattr(h, "alternative_interpretation", ""),
                    getattr(h, "confidence_factors", []),
                    getattr(h, "risk_factors", []),
                ])
                if cot_present:
                    with st.expander("🧠 Цепочка рассуждения Claude "
                                      "(chain-of-thought self-analysis)",
                                      expanded=True):
                        if h.structure_position:
                            st.markdown(f"**📍 Где мы в структуре:** "
                                         f"{h.structure_position}")
                        if h.rules_check:
                            st.markdown(f"**📜 Проверка правил:** "
                                         f"{h.rules_check}")
                        if h.alternation_check:
                            st.markdown(f"**🔁 Альтернация:** "
                                         f"{h.alternation_check}")
                        if h.fibonacci_check:
                            st.markdown(f"**📐 Fibonacci:** "
                                         f"{h.fibonacci_check}")
                        if h.alternative_interpretation:
                            st.markdown(f"**⚠️ Альтернативная разметка:** "
                                         f"{h.alternative_interpretation}")
                        cf_col, rf_col = st.columns(2)
                        with cf_col:
                            if h.confidence_factors:
                                st.markdown("**✅ За сетап:**")
                                for f in h.confidence_factors:
                                    st.markdown(f"- {f}")
                        with rf_col:
                            if h.risk_factors:
                                st.markdown("**❌ Против сетапа:**")
                                for f in h.risk_factors:
                                    st.markdown(f"- {f}")

                with st.expander("Raw Claude response"):
                    st.code(h.raw_text, language="json")
            st.divider()

    # Phase 23 — render holistic banner from session_state, so it survives
    # the st.rerun() that happens right after a successful Claude call.
    # The block ABOVE (inside `if _do_holistic:`) only renders on the SAME
    # script run that made the API call; after rerun, _do_holistic is False
    # but session_state still has the answer — show it from there.
    _st_h = st.session_state.get("last_holistic_full")
    if (_st_h and _st_h.get("ticker") == ticker
            # Skip if we already rendered above (avoid duplicate banner)
            and not _do_holistic):
        # Phase 32 — MCP-style markdown report vs compact widget-grid
        if holistic_view_mode == "MCP-style report":
            st.markdown(
                build_mcp_style_report(_st_h),
                unsafe_allow_html=True,
            )
            st.divider()
        else:
            bias_v = (_st_h.get("bias") or "neutral").lower()
            action_v = (_st_h.get("action") or "wait").lower()
            sq_v = _st_h.get("setup_quality", "low")
            bias_color = {
                "long":    REGIME_COLORS["Low Vol"],
                "short":   REGIME_COLORS["High Vol"],
                "neutral": TEXT_MUTED,
            }.get(bias_v, TEXT_MUTED)
            bias_emoji = {"long": "🟢", "short": "🔴",
                          "neutral": "⚪️"}.get(bias_v, "⚪️")
            st.markdown(
                f'<div style="padding:14px 18px; '
                f'border-left: 4px solid {bias_color}; '
                f'background:#161B22; border-radius: 8px;">'
                f'<div style="font-size:11px; color:#8B949E; '
                f'text-transform:uppercase; letter-spacing:.08em; '
                f'margin-bottom:6px;">Top-down вердикт Claude '
                f'(качество сетапа: {sq_v}) · сохранён в session</div>'
                f'<div style="font-size:20px; font-weight:700; '
                f'color:{bias_color};">{bias_emoji} BIAS: '
                f'{bias_v.upper()} · action: {action_v.upper()}</div>'
                f'<div style="font-size:14px; color:#E6EDF3; '
                f'margin-top:8px;">{_st_h.get("summary","")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Trade plan — 4 cards
            _entry = _st_h.get("entry")
            _stop = _st_h.get("stop_loss")
            _target = _st_h.get("target")
            _rr = _st_h.get("risk_reward")
            hp1, hp2, hp3, hp4 = st.columns(4)
            with hp1:
                st.markdown(metric_card(
                    title=t("claude.entry"),
                    value=(f"{_entry:.4f}" if isinstance(_entry, (int, float)) else "—"),
                    subtitle="", border_color=bias_color,
                ), unsafe_allow_html=True)
            with hp2:
                st.markdown(metric_card(
                    title=t("claude.stop"),
                    value=(f"{_stop:.4f}" if isinstance(_stop, (int, float)) else "—"),
                    subtitle="", border_color=REGIME_COLORS["High Vol"],
                ), unsafe_allow_html=True)
            with hp3:
                st.markdown(metric_card(
                    title=t("claude.target"),
                    value=(f"{_target:.4f}" if isinstance(_target, (int, float)) else "—"),
                    subtitle="", border_color=REGIME_COLORS["Low Vol"],
                ), unsafe_allow_html=True)
            with hp4:
                rr_v = f"{_rr:.2f}" if isinstance(_rr, (int, float)) else "—"
                rr_c = (REGIME_COLORS["Low Vol"] if (_rr or 0) >= 1.5
                        else REGIME_COLORS["High Vol"] if (_rr or 0) < 1
                        else ACCENT_CYAN)
                st.markdown(metric_card(
                    title="R/R", value=rr_v,
                    subtitle="≥1.5 — хороший сетап", border_color=rr_c,
                ), unsafe_allow_html=True)

            # Phase 31 — ACTION PLAN (главный output трейдера, ВЫШЕ всего)
            action_plan = _st_h.get("action_plan") or []
            if action_plan:
                st.markdown("### ▶ План действий")
                plan_md_lines = []
                for s in action_plan:
                    if not isinstance(s, dict):
                        continue
                    step_n = s.get("step", "?")
                    act = s.get("action", "")
                    if act:
                        plan_md_lines.append(f"**{step_n}.** {act}")
                if plan_md_lines:
                    st.markdown(
                        f'<div style="padding:14px 18px; '
                        f'background:#0E1015; border-left: 4px solid {ACCENT_CYAN}; '
                        f'border-radius: 6px; margin: 12px 0; '
                        f'font-size: 14px; line-height: 1.65;">'
                        + "<br>".join(plan_md_lines)
                        + '</div>',
                        unsafe_allow_html=True,
                    )

            # Phase 29 — Honest invalidation of previous prediction
            inv_prev = _st_h.get("invalidation_of_previous") or ""
            if inv_prev:
                st.warning(f"⚠️ **Invalidation предыдущего setup'а:** {inv_prev}")

            # Phase 29 — Multi-scenario plan with probability weights
            scenarios = _st_h.get("scenarios") or []
            if scenarios:
                st.markdown(
                    "**🎲 Сценарии с вероятностями** "
                    "(if-this-then-that план):"
                )
                rows = []
                for sc in scenarios:
                    if not isinstance(sc, dict):
                        continue
                    rows.append({
                        "Сценарий":      sc.get("name", "—"),
                        "Вероятность":   (f"{sc['probability_pct']}%"
                                           if isinstance(sc.get("probability_pct"),
                                                         (int, float))
                                           else "—"),
                        "🚨 Триггер":    (sc.get("trigger") or "")[:60],
                        "▶ Действие":    (sc.get("action") or "")[:60],
                        "Entry":         sc.get("entry_zone", "—"),
                        "Stop":          sc.get("stop", "—"),
                    })
                if rows:
                    st.dataframe(pd.DataFrame(rows), hide_index=True,
                                  use_container_width=True)

            # Phase 29 — Waiting conditions checklist
            waiting = _st_h.get("waiting_conditions") or []
            if waiting:
                st.markdown("**📋 Waiting checklist** "
                            "(что должно произойти прежде чем войти):")
                for w in waiting:
                    if isinstance(w, dict):
                        step = w.get("step", "?")
                        cond = w.get("condition", "")
                        st.markdown(f"- **{step}.** {cond}")

            # Phase 28 — Entry alternatives (tight vs wider stop)
            ea = _st_h.get("entry_alternatives") or []
            if ea and any(isinstance(x, dict) and x.get("entry") for x in ea):
                st.markdown("**🎯 Entry alternatives** (tight vs wider stop):")
                rows = []
                for alt in ea:
                    if not isinstance(alt, dict):
                        continue
                    rows.append({
                        "Версия":    alt.get("name", "—"),
                        "Entry":     alt.get("entry", "—"),
                        "Stop":      alt.get("stop", "—"),
                        "Risk":      alt.get("risk", "—"),
                        "Обоснование": (alt.get("rationale") or "")[:80],
                    })
                if rows:
                    st.dataframe(pd.DataFrame(rows), hide_index=True,
                                  use_container_width=True)

            # Phase 28 — Target pyramid with per-stop R/R
            tp = _st_h.get("target_pyramid") or []
            if tp and any(isinstance(t, dict) and t.get("price") for t in tp):
                st.markdown("**📈 Target pyramid:**")
                rows = []
                for t in tp:
                    if not isinstance(t, dict):
                        continue
                    rows.append({
                        "Target":   t.get("name", "—"),
                        "Цена":     t.get("price", "—"),
                        "R/R tight": t.get("rr_tight", "—"),
                        "R/R wider": t.get("rr_wider", "—"),
                    })
                if rows:
                    st.dataframe(pd.DataFrame(rows), hide_index=True,
                                  use_container_width=True)

            # Phase 28 — Cross-correlation notes
            cc = _st_h.get("cross_correlation_notes") or ""
            if cc:
                st.markdown(f"**🔗 Cross-correlation:** {cc}")

            # Phase 30 — Cross-asset comparison table (BTC vs ETH, etc.)
            cac = _st_h.get("cross_asset_comparison") or []
            lfn = _st_h.get("leader_follower_note") or ""
            if cac or lfn:
                with st.expander(
                    f"⚖ Cross-asset comparison ({len(cac)} строк) — "
                    "leader vs follower",
                    expanded=False,
                ):
                    if lfn:
                        st.markdown(f"**👑 Leader/Follower:** {lfn}")
                    if cac:
                        rows = []
                        for row in cac:
                            if not isinstance(row, dict):
                                continue
                            rows.append({
                                "Параметр":  row.get("parameter", "—"),
                                "Primary":   row.get("primary", "—"),
                                "Secondary": row.get("secondary", "—"),
                            })
                        if rows:
                            st.dataframe(pd.DataFrame(rows), hide_index=True,
                                          use_container_width=True)

            # Phase 26 — Higher-degree (super-cycle) context
            hdc = _st_h.get("higher_degree_context") or {}
            if isinstance(hdc, dict) and any(hdc.values()):
                with st.expander("🌐 Higher-degree context (где мы в супер-цикле)",
                                  expanded=False):
                    if hdc.get("cycle_peak"):
                        st.markdown(f"**🔝 Cycle peak:** {hdc['cycle_peak']}")
                    if hdc.get("cycle_low"):
                        st.markdown(f"**📉 Cycle low:** {hdc['cycle_low']}")
                    if hdc.get("current_phase"):
                        st.markdown(f"**📍 Current phase:** {hdc['current_phase']}")

            # Phase 26 — Monitoring matrix (if-then triggers)
            monitoring = _st_h.get("monitoring_levels") or []
            if monitoring:
                with st.expander(
                    f"🎯 Monitoring matrix ({len(monitoring)} триггеров для "
                    "следующего хода)",
                    expanded=(action_v == "wait"),  # auto-expand for wait setups
                ):
                    st.markdown(
                        "Конкретные ценовые уровни и действия которые нужно "
                        "сделать ПРИ ИХ ПРОБОЕ. Особенно важно когда "
                        "swing-setup = wait."
                    )
                    rows = []
                    for lvl in monitoring:
                        if not isinstance(lvl, dict):
                            continue
                        rows.append({
                            "🚨 Триггер":   lvl.get("trigger", "—"),
                            "📋 Сценарий": lvl.get("scenario", "—"),
                            "▶ Действие":   lvl.get("action", "—"),
                        })
                    if rows:
                        st.dataframe(pd.DataFrame(rows), hide_index=True,
                                      use_container_width=True)

            # Phase 24 — 5m scalp plan (aligned with 1d bias)
            scalp = _st_h.get("scalp_plan") or {}
            if isinstance(scalp, dict) and scalp.get("direction"):
                scalp_dir = (scalp.get("direction") or "wait").lower()
                scalp_color = {
                    "long":  REGIME_COLORS["Low Vol"],
                    "short": REGIME_COLORS["High Vol"],
                    "wait":  TEXT_MUTED,
                }.get(scalp_dir, TEXT_MUTED)
                scalp_emoji = {"long": "🟢", "short": "🔴",
                               "wait": "⚪️"}.get(scalp_dir, "⚪️")
                st.markdown(
                    f'<div style="padding:10px 14px; '
                    f'border-left: 3px solid {scalp_color}; '
                    f'background:#0E1015; border-radius: 6px; margin-top: 12px;">'
                    f'<div style="font-size:11px; color:#8B949E; '
                    f'text-transform:uppercase; letter-spacing:.08em; '
                    f'margin-bottom:4px;">🎯 5m SCALP-план '
                    f'(в направлении 1d bias = {bias_v.upper()})</div>'
                    f'<div style="font-size:16px; font-weight:700; '
                    f'color:{scalp_color};">{scalp_emoji} '
                    f'{scalp_dir.upper()}</div>'
                    f'<div style="font-size:13px; color:#E6EDF3; '
                    f'margin-top:6px;">{scalp.get("rationale","")}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if scalp_dir != "wait":
                    _se = scalp.get("entry")
                    _ss = scalp.get("stop_loss")
                    _st1 = scalp.get("target_1")
                    _st2 = scalp.get("target_2")
                    _srr = scalp.get("risk_reward")
                    sp1, sp2, sp3, sp4, sp5 = st.columns(5)
                    def _fmt(v):
                        return f"{v:.4f}" if isinstance(v, (int, float)) else "—"
                    with sp1:
                        st.markdown(metric_card(
                            title="Scalp entry", value=_fmt(_se),
                            subtitle="", border_color=scalp_color,
                        ), unsafe_allow_html=True)
                    with sp2:
                        st.markdown(metric_card(
                            title="Scalp stop", value=_fmt(_ss),
                            subtitle="", border_color=REGIME_COLORS["High Vol"],
                        ), unsafe_allow_html=True)
                    with sp3:
                        st.markdown(metric_card(
                            title="Target 1 (2R)", value=_fmt(_st1),
                            subtitle="", border_color=REGIME_COLORS["Low Vol"],
                        ), unsafe_allow_html=True)
                    with sp4:
                        st.markdown(metric_card(
                            title="Target 2", value=_fmt(_st2),
                            subtitle="", border_color=REGIME_COLORS["Low Vol"],
                        ), unsafe_allow_html=True)
                    with sp5:
                        srr_v = (f"{_srr:.2f}"
                                 if isinstance(_srr, (int, float)) else "—")
                        st.markdown(metric_card(
                            title="Scalp R/R", value=srr_v,
                            subtitle="до target 1", border_color=ACCENT_CYAN,
                        ), unsafe_allow_html=True)
                    if scalp.get("trigger_condition"):
                        st.caption(
                            f"⚡ **Триггер активации:** "
                            f"{scalp['trigger_condition']}"
                        )
                    if scalp.get("invalidation"):
                        st.caption(
                            f"❌ **Инвалидация скальпа:** "
                            f"{scalp['invalidation']}"
                        )
                    if scalp.get("max_hold_bars"):
                        _hold = int(scalp["max_hold_bars"])
                        _hours = _hold * 5 / 60.0
                        st.caption(
                            f"⏱ **Ожидаемое время в сделке:** {_hold} 5m-баров "
                            f"(~{_hours:.1f} ч)"
                        )

                # Show wave_structure_5m and forecast regardless of direction
                # (useful even when scalp=wait — explains WHY we're waiting)
                if scalp.get("wave_structure_5m"):
                    st.caption(f"📐 **Структура на 5m:** "
                               f"{scalp['wave_structure_5m']}")
                if scalp.get("expected_timing"):
                    st.caption(f"⏰ **Ожидаемое время setup'а:** "
                               f"{scalp['expected_timing']}")

                # Fibonacci retrace table for the scalp
                fib_retr = scalp.get("fib_retrace_5m") or {}
                if isinstance(fib_retr, dict) and any(
                        isinstance(v, (int, float)) for v in fib_retr.values()):
                    with st.expander("📐 Fibonacci retrace levels (5m)",
                                      expanded=False):
                        rows = []
                        for k in ("23.6%", "38.2%", "50.0%", "61.8%",
                                  "78.6%", "100.0%"):
                            v = fib_retr.get(k)
                            if isinstance(v, (int, float)):
                                rows.append({"Level": k, "Price": f"{v:.4f}"})
                        if rows:
                            st.dataframe(pd.DataFrame(rows), hide_index=True,
                                          use_container_width=True)

                # 5m monitoring triggers
                mon5 = scalp.get("monitoring_5m") or []
                if mon5:
                    with st.expander(
                        f"🎯 5m Monitoring triggers ({len(mon5)} условий)",
                        expanded=False,
                    ):
                        rows = []
                        for lvl in mon5:
                            if isinstance(lvl, dict):
                                rows.append({
                                    "🚨 Триггер":   lvl.get("trigger", "—"),
                                    "📋 Сценарий": lvl.get("scenario", "—"),
                                    "▶ Действие":   lvl.get("action", "—"),
                                })
                        if rows:
                            st.dataframe(pd.DataFrame(rows), hide_index=True,
                                          use_container_width=True)

            # Per-TF notes
            per_tf = _st_h.get("per_tf_notes") or {}
            if per_tf:
                st.markdown("**Заметки по таймфреймам:**")
                for tf_name, note in per_tf.items():
                    st.markdown(f"- **{tf_name}** — {note}")

            if _st_h.get("invalidation_explained"):
                st.markdown(f"**Условие отмены плана:** "
                            f"{_st_h['invalidation_explained']}")

            # Chain-of-thought
            cot_keys = ["structure_position", "rules_check", "alternation_check",
                        "fibonacci_check", "alternative_interpretation"]
            cot_present = any(_st_h.get(k) for k in cot_keys) \
                or _st_h.get("confidence_factors") or _st_h.get("risk_factors")
            if cot_present:
                with st.expander("🧠 Цепочка рассуждения Claude "
                                  "(chain-of-thought self-analysis)",
                                  expanded=True):
                    if _st_h.get("structure_position"):
                        st.markdown(f"**📍 Где мы в структуре:** "
                                     f"{_st_h['structure_position']}")
                    if _st_h.get("rules_check"):
                        st.markdown(f"**📜 Проверка правил:** "
                                     f"{_st_h['rules_check']}")
                    if _st_h.get("alternation_check"):
                        st.markdown(f"**🔁 Альтернация:** "
                                     f"{_st_h['alternation_check']}")
                    if _st_h.get("fibonacci_check"):
                        st.markdown(f"**📐 Fibonacci:** "
                                     f"{_st_h['fibonacci_check']}")
                    if _st_h.get("alternative_interpretation"):
                        st.markdown(f"**⚠️ Альтернативная разметка:** "
                                     f"{_st_h['alternative_interpretation']}")
                    cf_col, rf_col = st.columns(2)
                    with cf_col:
                        if _st_h.get("confidence_factors"):
                            st.markdown("**✅ За сетап:**")
                            for f in _st_h["confidence_factors"]:
                                st.markdown(f"- {f}")
                    with rf_col:
                        if _st_h.get("risk_factors"):
                            st.markdown("**❌ Против сетапа:**")
                            for f in _st_h["risk_factors"]:
                                st.markdown(f"- {f}")
            st.divider()

    # ----------------------------------------------------------------------
    # Phase 26 — TV MCP "second opinion" — manual paste OR (future) auto
    # ----------------------------------------------------------------------
    with st.expander("🔮 Второе мнение от TV MCP (LuxAlgo + супер-цикл)",
                      expanded=False):
        st.caption(
            "TV MCP агент имеет доступ к LuxAlgo Pine label'ам и истории "
            "графика — даёт higher-degree wave count с точной разметкой "
            "Wave (1)-(5) из 2024-2025. Вставь сюда его текстовый ответ "
            "чтобы сохранить и сравнить с holistic Claude."
        )

        _mcp_existing = load_mcp_analysis(ticker)
        if _mcp_existing:
            _mage = mcp_age_minutes(_mcp_existing)
            _age_str = ("—" if _mage is None
                        else f"{int(_mage)} мин назад" if _mage < 60
                        else f"{int(_mage/60)} ч {int(_mage%60)} мин назад")
            st.markdown(
                f"📌 **Сохранённый MCP анализ для {ticker}** "
                f"(от {_age_str})"
            )
            st.markdown(_mcp_existing.get("analysis_text", "")[:6000])
            st.caption(f"Источник: {_mcp_existing.get('author','manual')}, "
                       f"длина: {len(_mcp_existing.get('analysis_text',''))} симв.")
            if _mage is not None and _mage > 24 * 60:
                st.warning(f"⏰ Анализ старше суток — рекомендую обновить.")

        with st.form(key=f"mcp_paste_form_{ticker}"):
            st.markdown(f"**Вставить новый MCP анализ для {ticker}:**")
            _mcp_text = st.text_area(
                "Текст анализа",
                height=200,
                placeholder=(
                    "Вставь сюда полный текст ответа от TV MCP агента — "
                    "включая wave count, monitoring levels, verdict. "
                    "Поддерживается markdown."
                ),
                key=f"mcp_text_input_{ticker}",
            )
            _author_label = st.text_input(
                "Метка источника (опционально)",
                value="tv_mcp_manual",
                key=f"mcp_author_{ticker}",
            )
            submitted = st.form_submit_button("💾 Сохранить MCP анализ")
            if submitted and _mcp_text.strip():
                save_mcp_analysis(ticker, _mcp_text.strip(),
                                   author=_author_label or "tv_mcp_manual")
                st.success(f"✓ MCP анализ для {ticker} сохранён "
                           f"({len(_mcp_text)} симв.)")
                st.rerun()

        # Comparison hint
        if _mcp_existing and st.session_state.get("last_holistic_full"):
            _hf = st.session_state["last_holistic_full"]
            if _hf.get("ticker") == ticker:
                st.markdown("---")
                st.markdown(
                    "**🔍 Сравнение с нашим holistic Claude:**\n\n"
                    f"- Наш bias: **{_hf.get('bias','?').upper()}**, "
                    f"action: **{_hf.get('action','?').upper()}**\n"
                    f"- MCP анализ выше — посмотри совпадают ли verdict + "
                    f"higher-degree count + alternative interpretation."
                )

    st.markdown("**Отдельный ТФ:**")
    ai_tf_options = ["1d", "4h", "15m"]
    if include_5m:
        ai_tf_options.append("5m")
    ai_tf_choice = st.radio(
        t("claude.title"),
        options=ai_tf_options, horizontal=True,
        key="ai_tf_choice",
    )
    if st.button(t("claude.btn"), type="primary"):
        tf = next((r for r in results if r.name == ai_tf_choice and r.ok), None)
        if tf is None or not tf.candidates:
            st.warning(f"Нет анализа для {ai_tf_choice}.")
        else:
            with st.spinner(t("claude.calling")):
                try:
                    opinion = ask_claude(
                        ticker, ai_tf_choice,
                        tf.major_swings, tf.candidates,
                        forecast=tf.forecast,
                    )
                except ClaudeSpendLimitError as exc:
                    st.warning(
                        f"💸 **Лимит расходов Anthropic API исчерпан.**\n\n"
                        f"{exc}"
                    )
                    opinion = None
                except Exception as exc:
                    st.error(f"Ошибка вызова Claude: {type(exc).__name__}: {exc}")
                    opinion = None

            if opinion is not None:
                # Persist for later track-record analysis.
                try:
                    snapshot_price = float(tf.prices.iloc[-1]) if len(tf.prices) else 0.0
                    log_per_tf(
                        ticker, ai_tf_choice, opinion,
                        current_price=snapshot_price,
                        forecast_pattern=(tf.forecast.next_pattern
                                           if tf.forecast else None),
                        forecast_direction=(tf.forecast.direction
                                             if tf.forecast else None),
                    )
                except Exception as exc:
                    st.warning(f"Не удалось записать предсказание в журнал: "
                                f"{type(exc).__name__}: {exc}")

                agree_emoji = "✓" if opinion.agrees_with_top else "✗"
                agree_color = (
                    REGIME_COLORS["Low Vol"] if opinion.agrees_with_top
                    else REGIME_COLORS["High Vol"]
                )
                conf_color = {
                    "high":   REGIME_COLORS["Low Vol"],
                    "medium": ACCENT_CYAN,
                    "low":    TEXT_MUTED,
                }.get(opinion.confidence, TEXT_MUTED)

                # First row — agreement + preferred pattern.
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(
                        metric_card(
                            title=t("claude.agree"),
                            value=f"{agree_emoji} {opinion.agrees_with_top}",
                            subtitle=f"{t('claude.confidence')}: {opinion.confidence}",
                            border_color=agree_color,
                        ),
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(
                        metric_card(
                            title=t("claude.preferred"),
                            value=opinion.preferred_pattern or "(такая же)",
                            subtitle="",
                            border_color=conf_color,
                        ),
                        unsafe_allow_html=True,
                    )

                # Second row — actionable trade levels.
                action_color = {
                    "long":  REGIME_COLORS["Low Vol"],
                    "short": REGIME_COLORS["High Vol"],
                    "wait":  TEXT_MUTED,
                    "close": REGIME_COLORS["Medium Vol"],
                }.get(opinion.action, TEXT_MUTED)
                action_label = {
                    "long":  "🟢 ЛОНГ",
                    "short": "🔴 ШОРТ",
                    "wait":  "⚪️ ЖДЁМ",
                    "close": "🟡 ЗАКРЫТЬ",
                }.get(opinion.action, opinion.action.upper())

                t1, t2, t3, t4 = st.columns(4)
                with t1:
                    st.markdown(
                        metric_card(
                            title=t("claude.action"),
                            value=action_label,
                            subtitle="",
                            border_color=action_color,
                        ),
                        unsafe_allow_html=True,
                    )
                with t2:
                    st.markdown(
                        metric_card(
                            title=t("claude.entry"),
                            value=(f"{opinion.entry:.4f}"
                                    if opinion.entry is not None else "—"),
                            subtitle="",
                            border_color=action_color,
                        ),
                        unsafe_allow_html=True,
                    )
                with t3:
                    st.markdown(
                        metric_card(
                            title=t("claude.stop"),
                            value=(f"{opinion.stop_loss:.4f}"
                                    if opinion.stop_loss is not None else "—"),
                            subtitle="",
                            border_color=REGIME_COLORS["High Vol"],
                        ),
                        unsafe_allow_html=True,
                    )
                with t4:
                    st.markdown(
                        metric_card(
                            title=t("claude.target"),
                            value=(f"{opinion.target:.4f}"
                                    if opinion.target is not None else "—"),
                            subtitle="",
                            border_color=REGIME_COLORS["Low Vol"],
                        ),
                        unsafe_allow_html=True,
                    )

                # Forecast critique (only meaningful if forecast was sent)
                if (tf.forecast is not None
                        and opinion.agrees_with_forecast is not None):
                    fc_agree = opinion.agrees_with_forecast
                    fc_color = (REGIME_COLORS["Low Vol"] if fc_agree
                                else REGIME_COLORS["High Vol"])
                    fc_emoji = "✓ согласен с прогнозом" if fc_agree else "✗ не согласен с прогнозом"
                    st.markdown(
                        f'<div style="padding:10px 14px; '
                        f'border-left: 4px solid {fc_color}; '
                        f'background:#161B22; border-radius: 6px; margin: 10px 0;">'
                        f'<div style="font-weight:600; color:{fc_color}; '
                        f'margin-bottom:4px;">Claude о прогнозе: {fc_emoji}</div>'
                        f'<div style="color:#E6EDF3; font-size:13px;">'
                        f'{opinion.forecast_critique or "—"}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(f"**{t('claude.reasoning')}**")
                st.write(opinion.reasoning)
                with st.expander("Raw Claude response"):
                    st.code(opinion.raw_text, language="json")


# ---------------------------------------------------------------------------
# Reading guide
# ---------------------------------------------------------------------------
section_header(t("section.how_to_read"))

st.markdown(t("guide.title"))
st.markdown(t("guide.step_1"))
st.markdown(t("guide.step_2"))
st.markdown(t("guide.step_3"))
st.markdown(t("guide.step_4"))
st.markdown("")
st.markdown(t("guide.rules_title"))
st.markdown(t("guide.rule_r0"))
st.markdown(t("guide.rule_r1"))
st.markdown(t("guide.rule_r2"))
st.markdown(t("guide.rule_r3"))
st.markdown("")
st.markdown(t("guide.score"))
st.markdown(t("guide.alternatives"))
