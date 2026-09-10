"""Track-record page for Claude predictions.

Shows every per-TF / holistic opinion Claude has given (logged from the
Elliott Waves page), the verified outcomes (win / loss / open / expired),
and aggregate hit-rate / average-R statistics with per-bucket breakdowns
(by TF, by bias, by setup_quality, by confidence, by agrees_with_top).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import yfinance as yf

from design_system import (
    ACCENT_CYAN, BORDER_SUBTLE, REGIME_COLORS, TEXT_MUTED, TEXT_PRIMARY,
    apply_theme, metric_card, section_header,
)
from prediction_log import (
    compute_stats, load_all, verify_predictions, PREDICTIONS_PATH,
)
from claude_call_log import (
    load_all as load_claude_calls, cost_summary, CALLS_PATH,
)
from twitter_sentiment import load_sentiment_history, SENTIMENT_HISTORY_PATH
from wave_journal import (
    load_snapshots, load_actions, load_transitions,
    verify_snapshots, verify_actions,
    compute_detector_stats, compute_action_stats, compute_stability_stats,
    SNAPSHOTS_PATH, ACTIONS_PATH, TRANSITIONS_PATH,
)


apply_theme()

st.markdown(
    '<div style="display:flex; align-items:baseline; gap:14px;">'
    '<div style="font-size:24px; font-weight:700; letter-spacing:-0.02em;">'
    '📊 Track Record Claude</div>'
    '<div style="color:#8B949E; font-size:13px;">Все мнения Claude → '
    'верификация на исторических данных → статистика угадываний.</div>'
    '</div>',
    unsafe_allow_html=True,
)
st.divider()


# ---------------------------------------------------------------------------
# OHLC fetcher used by verify_predictions
# ---------------------------------------------------------------------------
_TF_TO_YF: dict[str, str] = {
    "1d":  "1d",
    "4h":  "1h",      # resampled below
    "15m": "15m",
    "5m":  "5m",
}
_TF_LOOKBACK_DAYS: dict[str, int] = {
    "1d":  120,
    "4h":  30,
    "15m": 14,
    "5m":  7,
}


def _fetch_ohlc(ticker: str, tf: str, start_ts: pd.Timestamp) -> pd.DataFrame:
    """Return high/low/close OHLC at `tf` resolution for bars at-or-after start_ts.

    yfinance has tight intraday history limits — 5m gets ~7 days, 15m ~14 days
    backwards. For 4h we pull 1h and resample.
    """
    yf_interval = _TF_TO_YF.get(tf, "1d")
    lookback = _TF_LOOKBACK_DAYS.get(tf, 60)
    # Always ask for "now - lookback" — we can't fetch beyond yf's window anyway.
    end = datetime.utcnow()
    start = end - timedelta(days=lookback)
    # If the prediction is older than yf can serve us — return empty: caller
    # will leave the prediction as 'open' (no_data).
    df = yf.download(
        ticker, start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval=yf_interval, auto_adjust=True, progress=False, threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if tf == "4h":
        try:
            resampled = df[["Open", "High", "Low", "Close"]].resample(
                "4h", label="right", closed="right"
            )
        except ValueError:
            resampled = df[["Open", "High", "Low", "Close"]].resample(
                "4H", label="right", closed="right"
            )
        df = resampled.agg({"Open": "first", "High": "max",
                            "Low": "min", "Close": "last"}).dropna()
    df = df.rename(columns={"High": "high", "Low": "low", "Close": "close"})
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    # Slice to bars strictly after start_ts
    cutoff = pd.Timestamp(start_ts).tz_localize(None) if pd.Timestamp(start_ts).tz is not None else pd.Timestamp(start_ts)
    df = df[df.index > cutoff]
    return df[["high", "low", "close"]]


# ---------------------------------------------------------------------------
# Action bar
# ---------------------------------------------------------------------------
predictions = load_all()
total = len(predictions)

action_col1, action_col2, action_col3 = st.columns([1, 1, 2])
with action_col1:
    if st.button("🔄 Проверить открытые прогнозы", type="primary",
                 disabled=(total == 0)):
        with st.spinner("Получаем историю цен и сверяемся…"):
            stats = verify_predictions(_fetch_ohlc, only_open=True)
        st.success(
            f"Проверено: {stats['checked']}  ·  выиграло: {stats['won']}  ·  "
            f"проиграло: {stats['lost']}  ·  истёк горизонт: {stats['expired']}"
            f"  ·  ещё открыто: {stats['still_open']}  ·  без сделки: "
            f"{stats['no_trade']}"
        )
        predictions = load_all()  # reload after writes

with action_col2:
    if st.button("🔁 Перепроверить все", disabled=(total == 0)):
        with st.spinner("Перепроверяем всю историю предсказаний…"):
            stats = verify_predictions(_fetch_ohlc, only_open=False)
        st.success(f"Перепроверено: {stats['checked']}")
        predictions = load_all()

with action_col3:
    st.markdown(
        f'<div style="color:#8B949E; font-size:12px; line-height:1.6;">'
        f'Файл журнала: <code>{PREDICTIONS_PATH}</code><br/>'
        f'Записей: <b>{total}</b>. Верификация: yfinance OHLC, '
        f'long выигрывает когда high≥target раньше low≤stop.</div>',
        unsafe_allow_html=True,
    )

if total == 0:
    st.info(
        "Пока нет ни одного прогноза. Пойди на страницу **Elliott Waves**, "
        "нажми «🧠 Получить holistic мнение Claude (все ТФ)» или «Получить "
        "мнение Claude» — каждое мнение автоматически попадёт сюда."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Aggregate stats
# ---------------------------------------------------------------------------
stats = compute_stats(predictions)

section_header("Общая статистика")
sc1, sc2, sc3, sc4 = st.columns(4)
with sc1:
    st.markdown(metric_card(
        title="Всего прогнозов",
        value=str(stats["total"]),
        subtitle=f"проверено: {stats['verified']}",
        border_color=ACCENT_CYAN,
    ), unsafe_allow_html=True)
with sc2:
    hit = stats["hit_rate"]
    hit_str = f"{hit:.0%}" if hit is not None else "—"
    hit_color = (REGIME_COLORS["Low Vol"] if (hit or 0) >= 0.55
                 else REGIME_COLORS["High Vol"] if (hit or 0) < 0.45
                 else ACCENT_CYAN)
    st.markdown(metric_card(
        title="Hit rate",
        value=hit_str,
        subtitle=f"{stats['won']}W / {stats['lost']}L",
        border_color=hit_color,
    ), unsafe_allow_html=True)
with sc3:
    avgR = stats["avg_r_realized"]
    avgR_str = f"{avgR:+.2f}R" if avgR is not None else "—"
    r_color = (REGIME_COLORS["Low Vol"] if (avgR or 0) > 0
               else REGIME_COLORS["High Vol"] if (avgR or 0) < 0
               else TEXT_MUTED)
    st.markdown(metric_card(
        title="Средний R",
        value=avgR_str,
        subtitle="на одну сделку",
        border_color=r_color,
    ), unsafe_allow_html=True)
with sc4:
    st.markdown(metric_card(
        title="Открыто / без сделки",
        value=f"{stats['open']} / {stats['no_trade']}",
        subtitle=f"истёкших: {stats['expired']}",
        border_color=TEXT_MUTED,
    ), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Breakdowns
# ---------------------------------------------------------------------------
def _render_breakdown(title: str, bucket: dict) -> None:
    if not bucket:
        return
    st.markdown(f"**{title}**")
    rows = []
    for k, v in bucket.items():
        decisive = v["won"] + v["lost"]
        rate = f"{v['hit_rate']:.0%}" if v["hit_rate"] is not None else "—"
        rows.append({
            "Категория": k,
            "Выиграл": v["won"],
            "Проиграл": v["lost"],
            "Истёк": v["expired"],
            "Hit rate": rate,
            "Решено": decisive,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


section_header("Разрезы")
bc1, bc2 = st.columns(2)
with bc1:
    _render_breakdown("По таймфрейму", stats["by_tf"])
    _render_breakdown("По уверенности (per-TF)", stats["by_confidence"])
with bc2:
    _render_breakdown("По bias (holistic)", stats["by_bias"])
    _render_breakdown("По качеству сетапа (holistic)", stats["by_setup_quality"])
_render_breakdown("Claude согласился с топ-гипотезой? (per-TF)",
                   stats["by_agrees_with_top"])


# ---------------------------------------------------------------------------
# Recent predictions table
# ---------------------------------------------------------------------------
section_header("История предсказаний")

# Filters
fc1, fc2, fc3, fc4 = st.columns(4)
with fc1:
    f_ticker = st.text_input("Тикер фильтр (пусто = все)", "")
with fc2:
    f_type = st.selectbox("Тип", ["все", "per_tf", "holistic"], index=0)
with fc3:
    f_outcome = st.selectbox(
        "Исход", ["все", "win", "loss", "expired", "open", "no_trade"], index=0
    )
with fc4:
    f_action = st.selectbox(
        "Действие", ["все", "long", "short", "wait", "close"], index=0
    )

filtered = predictions
if f_ticker.strip():
    filtered = [p for p in filtered
                if p.get("ticker", "").upper().startswith(f_ticker.upper().strip())]
if f_type != "все":
    filtered = [p for p in filtered if p.get("type") == f_type]
if f_outcome != "все":
    if f_outcome == "open":
        filtered = [p for p in filtered if p.get("outcome") is None]
    else:
        filtered = [p for p in filtered if p.get("outcome") == f_outcome]
if f_action != "все":
    filtered = [p for p in filtered if p.get("action") == f_action]

# Build table — newest first
filtered_sorted = sorted(filtered, key=lambda p: p.get("logged_at", ""),
                          reverse=True)

rows = []
for p in filtered_sorted[:200]:
    rr = p.get("risk_reward")
    rr_str = f"{rr:.2f}" if isinstance(rr, (int, float)) else "—"
    r_real = p.get("r_realized")
    r_real_str = f"{r_real:+.2f}R" if isinstance(r_real, (int, float)) else "—"
    outcome = p.get("outcome") or "open"
    outcome_icon = {"win": "✅", "loss": "❌", "expired": "⏱",
                    "no_trade": "⊘", "open": "🟡"}.get(outcome, "🟡")
    bias_or_top = (p.get("bias") if p.get("type") == "holistic"
                   else ("✓ top" if p.get("agrees_with_top") else "✗ top"))
    rows.append({
        "Дата": (p.get("logged_at") or "")[:16],
        "Тикер": p.get("ticker"),
        "TF": p.get("timeframe"),
        "Тип": p.get("type"),
        "Действие": p.get("action"),
        "Bias / агр.": bias_or_top,
        "Entry": p.get("entry"),
        "Stop": p.get("stop_loss"),
        "Target": p.get("target"),
        "R/R план": rr_str,
        "Цена в момент": p.get("current_price"),
        "Исход": f"{outcome_icon} {outcome}",
        "R факт": r_real_str,
        "id": p.get("id"),
    })

df = pd.DataFrame(rows)
if df.empty:
    st.info("Под фильтр ничего не попало.")
else:
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.caption(
        f"Показано {len(rows)} из {len(filtered)} (полная история выше — "
        f"всего {total}). Прокручивай таблицу для просмотра больших записей."
    )

# Expand a row for full reasoning
with st.expander("📖 Просмотреть полное reasoning / summary по id"):
    target_id = st.text_input("id (первые 16 символов)", "")
    if target_id.strip():
        match = next((p for p in predictions
                       if p.get("id") == target_id.strip()), None)
        if match is None:
            st.warning("Прогноз с таким id не найден.")
        else:
            if match.get("type") == "holistic":
                st.markdown(f"**Summary:** {match.get('summary')}")
                st.markdown(f"**Invalidation:** "
                             f"{match.get('invalidation_explained')}")
                st.markdown("**Per-TF notes:**")
                for k, v in (match.get("per_tf_notes") or {}).items():
                    st.markdown(f"- **{k}** — {v}")
            else:
                st.markdown(f"**Preferred pattern:** "
                             f"{match.get('preferred_pattern')}")
                st.markdown(f"**Reasoning:** {match.get('reasoning')}")
                if match.get("forecast_critique"):
                    st.markdown(f"**Forecast critique:** "
                                 f"{match.get('forecast_critique')}")


# ===========================================================================
# Phase 5.4 — Claude call audit log
# ===========================================================================
st.divider()
section_header("🤖 Claude calls — full audit trail")

calls = load_claude_calls()
costs = cost_summary()

cc1, cc2, cc3, cc4 = st.columns(4)
with cc1:
    st.markdown(metric_card(
        title="Всего вызовов", value=str(costs["n_calls"]),
        subtitle=f"за всё время", border_color=ACCENT_CYAN,
    ), unsafe_allow_html=True)
with cc2:
    st.markdown(metric_card(
        title="Tokens IN",
        value=f"{costs['tokens_in_total']:,}",
        subtitle="suma", border_color=TEXT_MUTED,
    ), unsafe_allow_html=True)
with cc3:
    st.markdown(metric_card(
        title="Tokens OUT",
        value=f"{costs['tokens_out_total']:,}",
        subtitle="suma", border_color=TEXT_MUTED,
    ), unsafe_allow_html=True)
with cc4:
    st.markdown(metric_card(
        title="Расход $",
        value=f"${costs['cost_usd_total']:.2f}",
        subtitle="за всё время",
        border_color=REGIME_COLORS["High Vol"] if costs["cost_usd_total"] > 5 else ACCENT_CYAN,
    ), unsafe_allow_html=True)

st.markdown(f"Файл лога: `{CALLS_PATH}`")

if not calls:
    st.info("Пока нет ни одного вызова Claude. Сделай прогноз на странице Elliott Waves.")
else:
    # Cost by type / ticker
    cb1, cb2 = st.columns(2)
    with cb1:
        st.markdown("**По типу:**")
        rows = [{"Тип": k, "Calls": v["calls"],
                  "Tokens in": v["tokens_in"], "Tokens out": v["tokens_out"],
                  "Cost $": round(v["cost_usd"], 4)}
                 for k, v in costs["by_type"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    with cb2:
        st.markdown("**По тикеру:**")
        rows = [{"Тикер": k, "Calls": v["calls"],
                  "Tokens": v["tokens_in"] + v["tokens_out"],
                  "Cost $": round(v["cost_usd"], 4)}
                 for k, v in costs["by_ticker"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Recent calls table
    st.markdown("**Последние вызовы (newest first):**")
    rows = []
    for c in reversed(calls[-50:]):
        parsed = c.get("parsed") or {}
        action_or_bias = parsed.get("action") or parsed.get("bias") or ""
        rows.append({
            "When": (c.get("logged_at") or "")[:16],
            "Тикер": c.get("ticker"),
            "Тип": c.get("type"),
            "TF": c.get("timeframe"),
            "Action/Bias": action_or_bias,
            "Entry": parsed.get("entry"),
            "Stop": parsed.get("stop_loss"),
            "Target": parsed.get("target"),
            "R/R": parsed.get("risk_reward"),
            "Tokens": (c.get("tokens_in", 0) + c.get("tokens_out", 0)),
            "Cost $": round(c.get("cost_usd", 0), 4),
            "Cached": "✓" if c.get("cached") else "",
            "Error": (c.get("error") or "")[:40],
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Expand a specific call by index
    with st.expander("🔍 Детальный просмотр (full prompt + response)"):
        idx_input = st.number_input(
            "Index (0 = последний)", min_value=0,
            max_value=max(0, len(calls) - 1), value=0, step=1,
        )
        call = list(reversed(calls))[int(idx_input)] if calls else None
        if call:
            st.markdown(f"**{call.get('logged_at')}** · {call.get('ticker')} · "
                         f"{call.get('type')} · "
                         f"tokens={call.get('tokens_in',0)}+{call.get('tokens_out',0)} "
                         f"· cost=${call.get('cost_usd',0):.4f}")
            st.markdown("**Prompt:**")
            st.code(call.get("prompt", "") or "(пусто — cached call)",
                     language="markdown")
            st.markdown("**Raw response:**")
            st.code(call.get("raw_response", "") or "(пусто)",
                     language="json")
            if call.get("parsed"):
                st.markdown("**Parsed:**")
                st.json(call["parsed"])


# ===========================================================================
# Phase 5.4 — Sentiment history
# ===========================================================================
st.divider()
section_header("🐦 Sentiment history — для backtest «двинуло ли соцсетку»")

hist = load_sentiment_history()
if not hist:
    st.info(
        f"Журнал sentiment пока пуст ({SENTIMENT_HISTORY_PATH}). Каждый "
        "refresh-цикл sidecar'а добавляет сюда per-ticker snapshot."
    )
else:
    hdf = pd.DataFrame(hist)
    hdf["logged_at"] = pd.to_datetime(hdf["logged_at"])

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown(metric_card(
            title="Snapshots в журнале",
            value=str(len(hist)),
            subtitle=f"тикеров: {hdf['ticker'].nunique()}",
            border_color=ACCENT_CYAN,
        ), unsafe_allow_html=True)
    with sc2:
        first_ts = hdf["logged_at"].min()
        last_ts = hdf["logged_at"].max()
        span_h = (last_ts - first_ts).total_seconds() / 3600 if len(hdf) > 1 else 0
        st.markdown(metric_card(
            title="Период наблюдения",
            value=f"{span_h:.1f}ч",
            subtitle=f"первый: {first_ts.strftime('%Y-%m-%d %H:%M')}",
            border_color=TEXT_MUTED,
        ), unsafe_allow_html=True)
    with sc3:
        bullish_count = (hdf["weighted_label"] == "bullish").sum()
        bearish_count = (hdf["weighted_label"] == "bearish").sum()
        st.markdown(metric_card(
            title="Bullish / Bearish",
            value=f"{bullish_count} / {bearish_count}",
            subtitle="за всю историю",
            border_color=ACCENT_CYAN,
        ), unsafe_allow_html=True)

    # Per-ticker time series
    st.markdown("**Эволюция sentiment по тикерам:**")
    available_tickers = sorted(hdf["ticker"].unique())
    selected = st.multiselect(
        "Тикеры", options=available_tickers, default=available_tickers[:3]
    )
    if selected:
        plot_df = hdf[hdf["ticker"].isin(selected)].copy()
        # Pivot for plotting
        pivot = plot_df.pivot_table(
            index="logged_at", columns="ticker",
            values="weighted_score", aggfunc="last"
        )
        st.line_chart(pivot, use_container_width=True)

    # Raw history table (last 100)
    with st.expander("📋 Сырой журнал (последние 100 записей)"):
        rows = []
        for h in reversed(hist[-100:]):
            tier_summary = ""
            for tier, d in (h.get("by_tier") or {}).items():
                tier_summary += f"{tier}={d.get('mean_sentiment'):+.2f}({d.get('n_tweets')}) "
            rows.append({
                "When": (h.get("logged_at") or "")[:16],
                "Тикер": h.get("ticker"),
                "Score": h.get("weighted_score"),
                "Label": h.get("weighted_label"),
                "Tweets": h.get("n_tweets_total"),
                "By tier": tier_summary,
                "Divergence": (h.get("divergence") or "")[:60],
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                          use_container_width=True)


# ===========================================================================
# Wave-journal — full audit of every analysis the detector produced
# ===========================================================================
st.divider()
section_header("🌊 Wave detector — hit-rate, действия, стабильность")
st.caption(
    "Каждое открытие страницы **Elliott Waves** автоматически пишет snapshot "
    "со всеми ТФ-классификациями и прогнозами сюда. Каждая твоя кнопка "
    "(«Взял setup», «Стоп» и т.д.) тоже пишется в журнал. Эти три блока "
    "превращают журнал в оценку: работает ли детектор, работают ли твои сетапы, "
    "стабильна ли классификация."
)

wj_snapshots = load_snapshots()
wj_actions = load_actions()
wj_transitions = load_transitions()

wj_ac1, wj_ac2, wj_ac3 = st.columns([1, 1, 2])
with wj_ac1:
    if st.button("🔄 Проверить прогнозы snapshot'ов",
                  disabled=(len(wj_snapshots) == 0),
                  key="wj_verify_snaps"):
        with st.spinner("Сверяемся с реальной ценой по каждому ТФ…"):
            sstats = verify_snapshots(_fetch_ohlc, only_open=True)
        st.success(
            f"Snapshot'ов обработано: {sstats['checked']}  ·  "
            f"TF-исходов: ✅ {sstats['tf_won']} / ❌ {sstats['tf_lost']} / "
            f"⏱ {sstats['tf_expired']} / 🟡 {sstats['tf_open']} / "
            f"⊘ {sstats['tf_skipped']}"
        )
        wj_snapshots = load_snapshots()
with wj_ac2:
    if st.button("🔄 Проверить мои действия",
                  disabled=(len(wj_actions) == 0),
                  key="wj_verify_actions"):
        with st.spinner("Считаем результаты по каждому «Взял setup»…"):
            astats = verify_actions(_fetch_ohlc, only_open=True)
        st.success(
            f"Действий проверено: {astats['checked']}  ·  "
            f"✅ {astats['won']} / ❌ {astats['lost']} / "
            f"⏱ {astats['expired']} / 🟡 {astats['open']} / "
            f"⊘ {astats['no_trade']}"
        )
        wj_actions = load_actions()
with wj_ac3:
    st.markdown(
        f'<div style="color:#8B949E; font-size:12px; line-height:1.6;">'
        f'Snapshots: <code>{SNAPSHOTS_PATH}</code> · '
        f'<b>{len(wj_snapshots)}</b><br/>'
        f'Actions: <code>{ACTIONS_PATH}</code> · '
        f'<b>{len(wj_actions)}</b><br/>'
        f'Transitions: <code>{TRANSITIONS_PATH}</code> · '
        f'<b>{len(wj_transitions)}</b></div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Block 1 — Wave Detector Hit-Rate
# ---------------------------------------------------------------------------
st.markdown("#### 🎯 Hit-rate детектора (по snapshot'ам)")

det_stats = compute_detector_stats(wj_snapshots)

dsc1, dsc2, dsc3, dsc4 = st.columns(4)
with dsc1:
    st.markdown(metric_card(
        title="Snapshots всего", value=str(det_stats["total_snapshots"]),
        subtitle="≈ запусков анализа", border_color=ACCENT_CYAN,
    ), unsafe_allow_html=True)
with dsc2:
    st.markdown(metric_card(
        title="TF-исходов", value=str(det_stats["total_tf_outcomes"]),
        subtitle="ТФ × snapshot", border_color=TEXT_MUTED,
    ), unsafe_allow_html=True)

# Aggregate hit-rate across all decisive TF outcomes
total_won = sum(v["won"] for v in det_stats["by_pattern"].values())
total_lost = sum(v["lost"] for v in det_stats["by_pattern"].values())
total_dec = total_won + total_lost
total_rate = (total_won / total_dec) if total_dec else None
with dsc3:
    rate_str = f"{total_rate:.0%}" if total_rate is not None else "—"
    rate_color = (REGIME_COLORS["Low Vol"] if (total_rate or 0) >= 0.55
                  else REGIME_COLORS["High Vol"] if (total_rate or 0) < 0.45
                  else ACCENT_CYAN)
    st.markdown(metric_card(
        title="Hit rate (общий)", value=rate_str,
        subtitle=f"{total_won}W / {total_lost}L",
        border_color=rate_color,
    ), unsafe_allow_html=True)
with dsc4:
    pat_count = len(det_stats["by_pattern"])
    st.markdown(metric_card(
        title="Паттернов в журнале", value=str(pat_count),
        subtitle="разных классификаций", border_color=TEXT_MUTED,
    ), unsafe_allow_html=True)

if det_stats["total_tf_outcomes"] == 0:
    st.info(
        "Пока ни один TF-прогноз не сверён. Нажми «🔄 Проверить прогнозы "
        "snapshot'ов» — для каждого ТФ с actionable forecast мы пройдём "
        "вперёд по реальной цене и решим: target достигнут (win) / "
        "invalidation сработал (loss) / окно истекло (expired)."
    )
else:
    db1, db2 = st.columns(2)
    with db1:
        _render_breakdown("По паттерну", det_stats["by_pattern"])
    with db2:
        _render_breakdown("По таймфрейму", det_stats["by_timeframe"])
    db3, db4 = st.columns(2)
    with db3:
        _render_breakdown("По направлению прогноза", det_stats["by_direction"])
    with db4:
        _render_breakdown("Pattern × TF (топ-10)",
                          dict(list(det_stats["by_pattern_tf"].items())[:10]))

# Recent snapshots browser
with st.expander("📋 Последние snapshot'ы (50)"):
    rows = []
    for s in reversed(wj_snapshots[-50:]):
        per_tf_summary = []
        for tf in s.get("per_tf", []):
            top = tf.get("top") or {}
            outcome = (s.get("per_tf_outcomes") or {}).get(tf.get("name"), {})
            status = outcome.get("status", "—")
            icon = {"win": "✅", "loss": "❌", "expired": "⏱",
                    "open": "🟡", "skipped": "⊘"}.get(status, "—")
            per_tf_summary.append(
                f"{tf.get('name')}={top.get('pattern','?')}{icon}"
            )
        synth = s.get("synthesis") or {}
        rows.append({
            "When":     (s.get("logged_at") or "")[:16],
            "Ticker":   s.get("ticker"),
            "Setup":    (synth.get("setup") or "")[:60],
            "Color":    synth.get("color", "—"),
            "TF summary": "  ".join(per_tf_summary),
            "Verified": (s.get("verified_at") or "")[:16] or "—",
            "id":       s.get("id"),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                      use_container_width=True)

# ---------------------------------------------------------------------------
# Block 2 — Manual Actions journal
# ---------------------------------------------------------------------------
st.markdown("#### ✋ Журнал ручных действий")

act_stats = compute_action_stats(wj_actions)

ab1, ab2, ab3, ab4 = st.columns(4)
with ab1:
    st.markdown(metric_card(
        title="Всего действий", value=str(act_stats["total"]),
        subtitle=(f"setup: {act_stats['take_setup']}  ·  "
                  f"skip: {act_stats['skip']}"),
        border_color=ACCENT_CYAN,
    ), unsafe_allow_html=True)
with ab2:
    hr = act_stats["hit_rate"]
    hr_str = f"{hr:.0%}" if hr is not None else "—"
    hr_color = (REGIME_COLORS["Low Vol"] if (hr or 0) >= 0.55
                 else REGIME_COLORS["High Vol"] if (hr or 0) < 0.45
                 else ACCENT_CYAN)
    st.markdown(metric_card(
        title="Winrate setups", value=hr_str,
        subtitle=f"{act_stats['won']}W / {act_stats['lost']}L",
        border_color=hr_color,
    ), unsafe_allow_html=True)
with ab3:
    avgR = act_stats["avg_r_realized"]
    avgR_str = f"{avgR:+.2f}R" if avgR is not None else "—"
    r_color = (REGIME_COLORS["Low Vol"] if (avgR or 0) > 0
                else REGIME_COLORS["High Vol"] if (avgR or 0) < 0
                else TEXT_MUTED)
    st.markdown(metric_card(
        title="Средний R (по setups)", value=avgR_str,
        subtitle="на одну сделку", border_color=r_color,
    ), unsafe_allow_html=True)
with ab4:
    st.markdown(metric_card(
        title="Открыто / истекло",
        value=f"{act_stats['open']} / {act_stats['expired']}",
        subtitle=f"стопов: {act_stats['stop_hit']}, "
                 f"тейков: {act_stats['take_profit']}",
        border_color=TEXT_MUTED,
    ), unsafe_allow_html=True)

if wj_actions:
    abb1, abb2 = st.columns(2)
    with abb1:
        if act_stats["by_ticker"]:
            _render_breakdown("По тикеру", act_stats["by_ticker"])
    with abb2:
        if act_stats["by_timeframe"]:
            _render_breakdown("По таймфрейму", act_stats["by_timeframe"])

    with st.expander("📋 Все действия (newest first)"):
        rows = []
        for a in reversed(wj_actions[-100:]):
            outcome = a.get("outcome") or "—"
            icon = {"win": "✅", "loss": "❌", "expired": "⏱",
                    "no_trade": "⊘"}.get(outcome, "🟡")
            r_real = a.get("r_realized")
            r_str = f"{r_real:+.2f}R" if isinstance(r_real, (int, float)) else "—"
            rows.append({
                "When":     (a.get("logged_at") or "")[:16],
                "Ticker":   a.get("ticker"),
                "TF":       a.get("timeframe"),
                "Тип":      a.get("action_type"),
                "Dir":      a.get("direction") or "—",
                "Entry":    a.get("entry"),
                "Stop":     a.get("stop_loss"),
                "Target":   a.get("target"),
                "Outcome":  f"{icon} {outcome}",
                "R факт":   r_str,
                "Заметка":  (a.get("notes") or "")[:50],
                "snap_id":  a.get("snapshot_id"),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                          use_container_width=True)
else:
    st.info(
        "Пока нет ни одного действия. На странице **Elliott Waves** под "
        "synthesis-вердиктом теперь есть «Журнал действий» с кнопками "
        "«Взял сетап», «Пропустил», «Закрыл по стопу», «Закрыл в плюс». "
        "Каждое нажатие пишется сюда."
    )

# ---------------------------------------------------------------------------
# Block 3 — Detector Stability
# ---------------------------------------------------------------------------
st.markdown("#### 🔁 Стабильность детектора")
st.caption(
    "Сколько раз классификация на (ticker, TF) переключилась за последние 14 "
    "дней. Большое число = детектор «дёргается» — те же данные периодически "
    "приводят к разному паттерну. Хороший детектор флипает редко."
)

stab_stats = compute_stability_stats(wj_transitions, window_days=14)

sb1, sb2 = st.columns([1, 3])
with sb1:
    st.markdown(metric_card(
        title="Всего переключений (14д)",
        value=str(stab_stats["total_transitions_in_window"]),
        subtitle="по всем тикерам и TF",
        border_color=ACCENT_CYAN,
    ), unsafe_allow_html=True)
with sb2:
    if stab_stats["by_ticker_tf"]:
        st.markdown("**Кто чаще всего переключается:**")
        df_st = pd.DataFrame([
            {"Ticker × TF": k, "Переключений": v}
            for k, v in sorted(stab_stats["by_ticker_tf"].items(),
                               key=lambda kv: -kv[1])
        ])
        st.dataframe(df_st, hide_index=True, use_container_width=True)
    else:
        st.info("Пока недостаточно snapshot'ов чтобы посчитать переключения. "
                "Открой страницу Elliott Waves хотя бы 2 раза для одного тикера.")

if stab_stats["by_pattern_pair"]:
    with st.expander("🔍 Самые частые пары переходов"):
        df_pp = pd.DataFrame([
            {"Переход": k, "Сколько раз": v}
            for k, v in sorted(stab_stats["by_pattern_pair"].items(),
                                key=lambda kv: -kv[1])[:30]
        ])
        st.dataframe(df_pp, hide_index=True, use_container_width=True)


# ===========================================================================
# Phase 46 — History of holistic versions per ticker (v1 → vN)
# ===========================================================================
st.divider()
st.markdown("#### 📊 История версий holistic per тикер")
st.caption(
    "Phase 43-46. Каждый новый holistic-запрос на тикер добавляет версию "
    "(v1, v2, …, vN). При следующем запросе цена за прошедшее время "
    "сверяется с предсказанными target/stop — выставляется verdict "
    "(✓ hit / ✗ stopped / ⏳ pending / ⚠ ambiguous). Это даёт детальную "
    "calibration на уровне отдельной монеты."
)

try:
    from holistic_history import (
        load_history, verify_history, MAX_HISTORY_PER_TICKER,
        _read_history as _hh_read,
    )
    _all_hh = _hh_read()
except Exception as exc:
    _all_hh = {}
    st.warning(f"holistic_history.py не доступен: {exc}")

if not _all_hh:
    st.info(
        "Пока нет истории holistic. Запусти **«🧠 Получить holistic мнение "
        "Claude»** на странице Elliott Waves хотя бы для одного тикера — "
        "после второго запроса появится первая верификация v1."
    )
else:
    # Ticker selector
    _tickers_with_hh = sorted(_all_hh.keys())
    hh_ticker = st.selectbox(
        "Тикер",
        options=_tickers_with_hh,
        index=0,
        key="holistic_hh_ticker",
    )

    # Re-verify on click (uses current OHLC)
    col_a, col_b = st.columns([1, 5])
    with col_a:
        do_reverify = st.button(
            "🔄 Перепроверить",
            help="Запустит verify_history против актуальной OHLC. Pending "
                 "версии могут стать hit/stopped по новым ценам.",
            key="hh_reverify_btn",
        )

    # Build OHLC fetcher reusing _fetch_ohlc (5m bars, most granular)
    def _hh_ohlc_fetcher(since_dt):
        try:
            cutoff = pd.Timestamp(since_dt).tz_localize(None) \
                if pd.Timestamp(since_dt).tz is not None \
                else pd.Timestamp(since_dt)
            df = _fetch_ohlc(hh_ticker, "5m", cutoff)
            return df if not df.empty else None
        except Exception:
            return None

    if do_reverify:
        with st.spinner("Верификация версий против OHLC..."):
            try:
                verify_history(hh_ticker, _hh_ohlc_fetcher,
                                max_n=MAX_HISTORY_PER_TICKER)
                st.success("✅ Верификация выполнена")
                _all_hh = _hh_read()  # reload after writes
            except Exception as e:
                st.error(f"Ошибка верификации: {e}")

    versions = _all_hh.get(hh_ticker) or []

    # Summary stats
    verdicts = {"hit": 0, "stopped": 0, "ambiguous": 0, "pending": 0}
    for v in versions:
        verdict = v.get("verdict", "pending")
        if verdict in verdicts:
            verdicts[verdict] += 1
    resolved = verdicts["hit"] + verdicts["stopped"]
    hit_rate_pct = (verdicts["hit"] / resolved * 100) if resolved > 0 else None

    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.markdown(metric_card(
            title="Всего версий", value=str(len(versions)),
            subtitle=f"максимум {MAX_HISTORY_PER_TICKER}",
            border_color=ACCENT_CYAN,
        ), unsafe_allow_html=True)
    with s2:
        st.markdown(metric_card(
            title="✓ Hit", value=str(verdicts["hit"]),
            subtitle="target достигнут",
            border_color=REGIME_COLORS["Low Vol"],
        ), unsafe_allow_html=True)
    with s3:
        st.markdown(metric_card(
            title="✗ Stopped", value=str(verdicts["stopped"]),
            subtitle="stop hit",
            border_color=REGIME_COLORS["High Vol"],
        ), unsafe_allow_html=True)
    with s4:
        st.markdown(metric_card(
            title="⏳ Pending", value=str(verdicts["pending"]),
            subtitle="ни target ни stop",
            border_color=TEXT_MUTED,
        ), unsafe_allow_html=True)
    with s5:
        hr_value = f"{hit_rate_pct:.0f}%" if hit_rate_pct is not None else "—"
        hr_color = (REGIME_COLORS["Low Vol"]
                     if hit_rate_pct is not None and hit_rate_pct >= 50
                     else REGIME_COLORS["High Vol"]
                     if hit_rate_pct is not None and hit_rate_pct < 33
                     else ACCENT_CYAN)
        st.markdown(metric_card(
            title="Hit rate", value=hr_value,
            subtitle=f"из {resolved} решённых",
            border_color=hr_color,
        ), unsafe_allow_html=True)

    # Versions table
    if versions:
        rows = []
        for v in versions:
            verdict = v.get("verdict", "pending")
            verdict_emoji = {
                "hit": "✓ HIT", "stopped": "✗ STOPPED",
                "ambiguous": "⚠ AMBIGUOUS", "pending": "⏳ PENDING",
            }.get(verdict, verdict)
            saved_at = v.get("saved_at", "")
            short_time = saved_at[:16].replace("T", " ") if saved_at else "—"
            bias = v.get("bias", "?")
            bias_emoji = {"long": "🟢", "short": "🔴",
                          "neutral": "⚪"}.get(bias, "⚪")
            rows.append({
                "Версия":   f"v{v.get('version', '?')}",
                "Дата":     short_time,
                "Bias":     f"{bias_emoji} {bias}",
                "Action":   v.get("action", "?"),
                "Quality":  v.get("setup_quality", "?"),
                "Entry":    v.get("entry"),
                "Stop":     v.get("stop_loss"),
                "Target":   v.get("target"),
                "R/R":      v.get("risk_reward"),
                "Реальность": verdict_emoji,
                "Note":     (v.get("verdict_note") or "")[:60],
                "Extreme":  v.get("extreme_price"),
            })
        df_hh = pd.DataFrame(rows)
        st.dataframe(df_hh, hide_index=True, use_container_width=True)

        # Detail expander
        if len(versions) > 0:
            with st.expander("🔍 Детали последней версии"):
                last_v = versions[-1]
                st.markdown(f"**v{last_v.get('version')} — "
                             f"{last_v.get('saved_at', '')}**")
                if last_v.get("summary"):
                    st.markdown(f"**Summary:** {last_v['summary']}")
                if last_v.get("golden_entry_zone"):
                    st.markdown(f"**Golden entry zone:** "
                                 f"{last_v['golden_entry_zone']}")
                if last_v.get("critical_level"):
                    st.markdown(f"**Critical level:** "
                                 f"{last_v['critical_level']}")
                if last_v.get("verdict_note"):
                    st.markdown(f"**Verdict note:** "
                                 f"{last_v['verdict_note']}")
