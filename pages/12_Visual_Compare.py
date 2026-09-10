"""Visual comparison page: our wave detection vs TradingView screenshot.

Pick a prediction, see side-by-side:
  - left: our Plotly chart with major + minor wave labels (Phase 1-3 patterns)
  - right: TV's own chart screenshot captured at time of prediction

Lets you visually verify whether our pattern detection agrees with what TV's
Pine indicators / community charting would label the same structure.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from design_system import (
    ACCENT_CYAN, BORDER_SUBTLE, REGIME_COLORS, TEXT_MUTED, TEXT_PRIMARY,
    apply_theme, metric_card, section_header,
)
from prediction_log import load_all as load_predictions
from tv_screenshot import capture, screenshot_path, screenshot_exists, SHOTS_DIR


apply_theme()

st.markdown(
    '<div style="display:flex; align-items:baseline; gap:14px;">'
    '<div style="font-size:24px; font-weight:700; letter-spacing:-0.02em;">'
    '🔍 Visual Compare — наш чарт vs TradingView</div>'
    '<div style="color:#8B949E; font-size:13px;">'
    'Сверка нашего волнового детектора с TV-разметкой по конкретному прогнозу.'
    '</div></div>',
    unsafe_allow_html=True,
)
st.divider()


# ---------------------------------------------------------------------------
# Pick a prediction
# ---------------------------------------------------------------------------
preds = load_predictions()
if not preds:
    st.info("Нет прогнозов в журнале. Сделай holistic на странице Elliott Waves.")
    st.stop()

# Newest-first picker
preds_sorted = sorted(preds, key=lambda p: p.get("logged_at", ""), reverse=True)
options = []
for p in preds_sorted[:100]:
    label = (f"{p.get('logged_at','')[:16]} · {p.get('ticker'):<10} "
              f"{p.get('type'):<10} · "
              f"{p.get('action','?'):<6} · "
              f"entry={p.get('entry') or '—'} · "
              f"id={p.get('id','')[:8]}")
    options.append((label, p))

selected_label = st.selectbox(
    "Прогноз для сверки", options=[o[0] for o in options],
)
sel_pred = next((o[1] for o in options if o[0] == selected_label), None)


# ---------------------------------------------------------------------------
# Prediction header
# ---------------------------------------------------------------------------
if sel_pred:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card(
            title="Тикер / TF",
            value=f"{sel_pred.get('ticker')} · {sel_pred.get('timeframe')}",
            subtitle=sel_pred.get("type", ""),
            border_color=ACCENT_CYAN,
        ), unsafe_allow_html=True)
    with c2:
        act = sel_pred.get("action", "—")
        act_color = (REGIME_COLORS["Low Vol"] if act == "long"
                      else REGIME_COLORS["High Vol"] if act == "short"
                      else TEXT_MUTED)
        st.markdown(metric_card(
            title="Действие", value=act.upper(),
            subtitle=f"bias: {sel_pred.get('bias', '—')}",
            border_color=act_color,
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card(
            title="Entry / Stop / Target",
            value=f"{sel_pred.get('entry') or '—'}",
            subtitle=f"S {sel_pred.get('stop_loss') or '—'} · "
                     f"T {sel_pred.get('target') or '—'}",
            border_color=TEXT_MUTED,
        ), unsafe_allow_html=True)
    with c4:
        rr = sel_pred.get("risk_reward")
        rr_str = f"{rr:.2f}" if isinstance(rr, (int, float)) else "—"
        st.markdown(metric_card(
            title="R/R", value=rr_str,
            subtitle=f"outcome: {sel_pred.get('outcome') or 'open'}",
            border_color=ACCENT_CYAN,
        ), unsafe_allow_html=True)

    # ---------------------------------------------------------------------
    # Side-by-side comparison
    # ---------------------------------------------------------------------
    st.divider()

    pid = sel_pred.get("id")
    ticker = sel_pred.get("ticker", "")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 🎯 Наша система — Plotly чарт с метками")
        # Show the prediction summary and per-TF notes
        if sel_pred.get("type") == "holistic":
            st.markdown(f"**Bias:** {sel_pred.get('bias', '—')}  ·  "
                         f"**Setup:** {sel_pred.get('setup_quality', '—')}")
            summary = sel_pred.get("summary", "")
            if summary:
                st.info(summary)
            notes = sel_pred.get("per_tf_notes") or {}
            for tf, note in notes.items():
                st.markdown(f"**{tf}** — {note}")
        else:
            st.markdown(f"**Pattern:** {sel_pred.get('preferred_pattern', '—')}  ·  "
                         f"**Confidence:** {sel_pred.get('confidence', '—')}")
            reasoning = sel_pred.get("reasoning", "")
            if reasoning:
                st.info(reasoning)
        st.caption(
            "Это разметка которую сделал наш Phase 1+2 детектор + Claude. "
            "Полный chart с метками — на странице Elliott Waves."
        )

    with col_right:
        st.markdown("### 📸 TradingView — независимая разметка")
        # Try each TF screenshot. Show LOUDLY when missing so user knows
        # which TF needs capture.
        n_missing = 0
        for tf in ("1d", "4h"):
            shot_id = f"{pid}_{tf}"
            path = screenshot_path(shot_id)
            if os.path.isfile(path):
                st.markdown(f"**{tf}:**")
                st.image(path, use_container_width=True)
            else:
                n_missing += 1
                # Big bright warning so user can't miss it
                st.markdown(
                    f'<div style="padding:10px; border:2px dashed #E36209; '
                    f'border-radius:6px; background:rgba(227,99,9,0.08); '
                    f'margin-bottom:8px;">'
                    f'<div style="color:#E36209; font-weight:700; font-size:14px;">'
                    f'⚠️ {tf.upper()} скриншот ОТСУТСТВУЕТ</div>'
                    f'<div style="color:#8B949E; font-size:11px;">'
                    f'Нажми кнопку «📸 Захватить недостающие» ниже — '
                    f'bridge сделает PNG через TV Desktop за 20-30 сек.'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

        if n_missing > 0:
            st.caption(
                f"💡 Подсказка: для тикера {ticker} ещё нет TV-скриншотов. "
                f"Это бывает когда холистик-прогноз сделан до запуска bridge, "
                f"или bridge был перегружен очередью. Нажми «📸 Захватить "
                f"недостающие» — будет видно через ~30 сек."
            )

        # Capture-on-demand buttons — both modes:
        # «Захватить недостающие» — only TFs without existing PNG
        # «🔄 Пересохранить (force)» — re-capture even if PNG exists
        #
        # Phase 21 — load OHLC + swings so the Plotly fallback in
        # tv_screenshot.capture() can render a chart even when TV Desktop
        # and chart-img.com are both unavailable. Without this, only TV/
        # chart-img worked here and clicks failed when bridge was down.
        def _load_ohlc_for_capture(_ticker: str) -> dict:
            """Run multi_tf.analyze_ticker so we have prices+swings per TF.
            Best-effort; returns empty dict on failure."""
            try:
                from multi_tf import analyze_ticker, DEFAULT_TF_CONFIGS
                results = analyze_ticker(
                    _ticker, tf_configs=DEFAULT_TF_CONFIGS, source="auto",
                )
                return {r.name: r for r in results if r.ok and len(r.prices)}
            except Exception:
                return {}

        cap_col1, cap_col2 = st.columns(2)
        with cap_col1:
            if st.button("📸 Захватить недостающие", key=f"cap_missing_{pid}"):
                with st.spinner("Снимаем TV-чарт (с Plotly fallback)…"):
                    tf_data = _load_ohlc_for_capture(ticker)
                    ok = 0
                    skipped = 0
                    for tf in ("1d", "4h"):
                        if screenshot_exists(f"{pid}_{tf}"):
                            skipped += 1
                            continue
                        _r = tf_data.get(tf)
                        path = capture(
                            f"{pid}_{tf}", ticker, tf,
                            prices=_r.prices if _r else None,
                            major_swings=_r.major_swings if _r else None,
                            minor_swings=_r.minor_swings if _r else None,
                        )
                        if path:
                            ok += 1
                    if ok > 0 or skipped > 0:
                        st.success(f"Захвачено {ok} новых, пропущено {skipped} "
                                    f"(уже есть).")
                        st.rerun()
                    else:
                        # Detailed diagnosis — show what failed where
                        from tv_screenshot import diagnose_capture
                        _diag_tf = "1d" if tf_data.get("1d") else "4h"
                        _r = tf_data.get(_diag_tf)
                        diag = diagnose_capture(
                            ticker, _diag_tf,
                            prices=_r.prices if _r else None,
                            major_swings=_r.major_swings if _r else None,
                            minor_swings=_r.minor_swings if _r else None,
                        )
                        st.error(
                            f"Не удалось захватить ни один TF. Диагностика "
                            f"({_diag_tf}):"
                        )
                        for backend, status in diag.items():
                            ico = "✅" if status.startswith("OK") else "❌"
                            st.code(f"{ico} {backend}: {status}",
                                     language="text")
        with cap_col2:
            if st.button("🔄 Пересохранить (force)", key=f"cap_force_{pid}"):
                with st.spinner("Принудительный re-capture (с Plotly fallback)…"):
                    tf_data = _load_ohlc_for_capture(ticker)
                    ok = 0
                    for tf in ("1d", "4h"):
                        _r = tf_data.get(tf)
                        path = capture(
                            f"{pid}_{tf}", ticker, tf, force=True,
                            prices=_r.prices if _r else None,
                            major_swings=_r.major_swings if _r else None,
                            minor_swings=_r.minor_swings if _r else None,
                        )
                        if path:
                            ok += 1
                    if ok > 0:
                        st.success(f"Пересохранено {ok}/2 скриншотов.")
                        st.rerun()
                    else:
                        from tv_screenshot import diagnose_capture
                        _diag_tf = "1d" if tf_data.get("1d") else "4h"
                        _r = tf_data.get(_diag_tf)
                        diag = diagnose_capture(
                            ticker, _diag_tf,
                            prices=_r.prices if _r else None,
                            major_swings=_r.major_swings if _r else None,
                            minor_swings=_r.minor_swings if _r else None,
                        )
                        st.error(f"Force-refresh не удался. Диагностика ({_diag_tf}):")
                        for backend, status in diag.items():
                            ico = "✅" if status.startswith("OK") else "❌"
                            st.code(f"{ico} {backend}: {status}",
                                     language="text")

# ---------------------------------------------------------------------------
# Gallery — все сохранённые скриншоты с thumbnail-preview
# ---------------------------------------------------------------------------
st.divider()
st.markdown("### 🖼 Все захваченные скриншоты")

if not os.path.isdir(SHOTS_DIR):
    st.info(f"Папка {SHOTS_DIR} не создана. Будет создана при первом захвате.")
else:
    files = sorted(
        [f for f in os.listdir(SHOTS_DIR) if f.endswith(".png")],
        key=lambda f: -os.path.getmtime(os.path.join(SHOTS_DIR, f)),
    )
    if not files:
        st.info("Скриншотов пока нет. Сделай holistic на Elliott Waves — "
                 "автоматически захватятся 1d и 4h.")
    else:
        st.markdown(f"Всего файлов: **{len(files)}**. "
                     "Сверху — самые свежие. Кликни на превью чтобы открыть в полном размере.")
        # 3 columns gallery
        for row_start in range(0, len(files), 3):
            cols = st.columns(3)
            for i, col in enumerate(cols):
                if row_start + i >= len(files):
                    break
                fname = files[row_start + i]
                path = os.path.join(SHOTS_DIR, fname)
                mtime = pd.Timestamp(os.path.getmtime(path), unit='s', tz='UTC')
                size_kb = os.path.getsize(path) / 1024
                with col:
                    st.image(path, caption=fname,
                             use_container_width=True)
                    st.caption(f"⏱ {mtime.strftime('%Y-%m-%d %H:%M')} · "
                                f"💾 {size_kb:.0f}KB")
                    # Match this screenshot to a prediction if possible
                    # filenames: {pid_16}_{tf}.png e.g. faf5bbd3a2a8f53b_4h.png
                    base = fname[:-4]   # strip .png
                    parts = base.rsplit("_", 1)
                    if len(parts) == 2:
                        pid_prefix, tf_suffix = parts
                        matched = next((p for p in preds_sorted
                                         if p.get("id", "").startswith(pid_prefix)), None)
                        if matched:
                            st.caption(
                                f"🎯 {matched.get('ticker')} · {matched.get('action')} · "
                                f"entry {matched.get('entry')} ({tf_suffix})"
                            )
                            # Force-refresh button for THIS thumbnail
                            if st.button("🔄 Обновить", key=f"refresh_{fname}",
                                         use_container_width=True):
                                with st.spinner(f"Пересохраняем {fname}…"):
                                    full_pid = matched["id"]
                                    out = capture(f"{full_pid}_{tf_suffix}",
                                                    matched.get("ticker"),
                                                    tf_suffix, force=True)
                                    if out:
                                        st.success("Готово")
                                        st.rerun()
                                    else:
                                        st.error("Не удалось — проверь bridge")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
with st.expander("🔧 Diagnostics — screenshots folder paths"):
    if not os.path.isdir(SHOTS_DIR):
        st.markdown(f"Папка {SHOTS_DIR} не создана.")
    else:
        st.markdown(f"Папка: `{os.path.abspath(SHOTS_DIR)}`")
        files = sorted(os.listdir(SHOTS_DIR), reverse=True)
        st.markdown(f"Всего файлов: **{len(files)}**")
        for f in files[:30]:
            st.code(f)
