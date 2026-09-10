"""TradingView Ideas dashboard.

Shows per-ticker bias from public TV trade ideas, broken down by author tier
(automatic from follower count), with top setups and divergence signals.

Plus the Influencer leaderboard (Twitter + TV combined track-record).
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from design_system import (
    ACCENT_CYAN, BORDER_SUBTLE, REGIME_COLORS, TEXT_MUTED, TEXT_PRIMARY,
    apply_theme, metric_card, section_header,
)
from tv_ideas_aggregate import (
    load_tv_index, load_tv_history, refresh_from_snapshots, TV_INDEX_PATH,
)
from tv_ideas_scraper import (
    scrape, save_snapshot, SNAPSHOTS_DIR, _apify_token,
)
from influencer_track_record import author_stats, load_all as load_predictions


apply_theme()

st.markdown(
    '<div style="display:flex; align-items:baseline; gap:14px;">'
    '<div style="font-size:24px; font-weight:700; letter-spacing:-0.02em;">'
    '📈 TradingView Ideas</div>'
    '<div style="color:#8B949E; font-size:13px;">'
    'Публичные trade-идеи от TV-авторов, ранжированные по followers.'
    '</div></div>',
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Action bar
# ---------------------------------------------------------------------------
idx = load_tv_index()

ac1, ac2, ac3 = st.columns([1, 1, 2])
with ac1:
    if st.button("📈 Скачать идеи сейчас", type="primary"):
        with st.spinner("Скрапим TradingView Ideas через Apify…"):
            try:
                ideas = scrape(max_per_source=40)
                if ideas:
                    save_snapshot(ideas)
                    refresh_from_snapshots(within_hours=48)
                    st.success(f"Получено {len(ideas)} идей.")
                else:
                    st.warning("Apify вернул 0 идей. Проверь APIFY_API_TOKEN.")
            except Exception as exc:
                st.error(f"Ошибка: {type(exc).__name__}: {exc}")
        st.rerun()

with ac2:
    if st.button("🧪 Mock-данные"):
        ideas = scrape(mock=True)
        if ideas:
            save_snapshot(ideas)
            refresh_from_snapshots()
            st.info(f"Загружено {len(ideas)} mock-идей.")
        st.rerun()

with ac3:
    token = "🟢 настроен" if _apify_token() else "🔴 не настроен"
    st.markdown(
        f'<div style="color:#8B949E; font-size:12px;">'
        f'Последнее обновление: <b>{idx.get("updated_at", "—")[:16]}</b><br/>'
        f'Apify токен: {token}<br/>'
        f'Тикеров: {idx.get("n_tickers", 0)}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Per-ticker TV bias
# ---------------------------------------------------------------------------
section_header("TV Ideas bias по тикерам")

tickers_data = idx.get("tickers", {})
if not tickers_data:
    st.info("Пока нет данных. Нажми «📈 Скачать идеи сейчас» или mock.")
else:
    for ticker in sorted(tickers_data.keys()):
        t = tickers_data[ticker]
        bias = t.get("weighted_bias", 0)
        label = t.get("label", "neutral").upper()
        n_ideas = t.get("n_ideas_total", 0)
        n_long  = t.get("n_long", 0)
        n_short = t.get("n_short", 0)
        color = (REGIME_COLORS["Low Vol"] if bias > 0.2
                  else REGIME_COLORS["High Vol"] if bias < -0.2
                  else TEXT_MUTED)
        emoji = "🟢" if bias > 0.2 else "🔴" if bias < -0.2 else "⚪"
        st.markdown(
            f'<div style="padding:14px 18px; border-left:4px solid {color}; '
            f'background:#161B22; border-radius:8px; margin-bottom:12px;">'
            f'<div style="font-size:11px; color:#8B949E; text-transform:uppercase;">'
            f'{ticker} · {n_ideas} идей за 24ч ({n_long} LONG, {n_short} SHORT)'
            f'</div>'
            f'<div style="font-size:20px; font-weight:700; color:{color};">'
            f'{emoji} {label} · bias {bias:+.2f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Per-tier breakdown
        by_tier = t.get("by_tier", {})
        cols = st.columns(3)
        for i, tier_label in enumerate(("tier_1", "tier_2", "tier_3")):
            with cols[i]:
                d = by_tier.get(tier_label)
                tier_human = {"tier_1": "🎯 Tier-1 (>50k followers)",
                              "tier_2": "📊 Tier-2 (5k-50k)",
                              "tier_3": "👥 Tier-3 (<5k)"}[tier_label]
                if not d:
                    st.markdown(
                        f'<div style="color:#6E7B8B; font-size:12px;">'
                        f'{tier_human}: нет идей</div>',
                        unsafe_allow_html=True,
                    )
                    continue
                tier_color = (REGIME_COLORS["Low Vol"] if d["bias"] > 0.2
                                else REGIME_COLORS["High Vol"] if d["bias"] < -0.2
                                else TEXT_MUTED)
                st.markdown(
                    f'<div style="padding:10px; border:1px solid {tier_color}; '
                    f'border-radius:6px;">'
                    f'<div style="font-size:11px; color:#8B949E;">{tier_human}</div>'
                    f'<div style="font-size:17px; font-weight:700; color:{tier_color};">'
                    f'bias {d["bias"]:+.2f}</div>'
                    f'<div style="font-size:11px; color:#8B949E;">'
                    f'{d["n_long"]} L / {d["n_short"]} S · {d["label"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        if t.get("divergence"):
            st.warning(t["divergence"])

        # Top ideas
        top = t.get("top_ideas", [])
        if top:
            rows = [{
                "Tier": ti["tier"], "Author": f"@{ti['author']}",
                "Followers": ti["followers"], "Direction": ti["direction"],
                "TF": ti["timeframe"], "Likes": ti["likes"],
                "Title": ti["title"], "URL": ti["url"],
            } for ti in top]
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                          use_container_width=True)
        st.divider()


# ---------------------------------------------------------------------------
# Influencer leaderboard (Phase 6.3)
# ---------------------------------------------------------------------------
section_header("🏆 Influencer leaderboard — кто чаще прав")
stats = author_stats()
total_preds = sum(s["n_total"] for s in stats)
verified = sum(s["n_verified"] for s in stats)
lc1, lc2, lc3 = st.columns(3)
with lc1:
    st.markdown(metric_card(
        title="Прогнозов в журнале",
        value=str(total_preds),
        subtitle=f"авторов: {len(stats)}",
        border_color=ACCENT_CYAN,
    ), unsafe_allow_html=True)
with lc2:
    st.markdown(metric_card(
        title="Верифицировано", value=str(verified),
        subtitle=f"открыто: {total_preds - verified}",
        border_color=TEXT_MUTED,
    ), unsafe_allow_html=True)
with lc3:
    decisive = sum(s["won"] + s["lost"] for s in stats)
    won_total = sum(s["won"] for s in stats)
    overall_hr = (won_total / decisive) if decisive else None
    st.markdown(metric_card(
        title="Общий hit-rate",
        value=f"{overall_hr:.0%}" if overall_hr is not None else "—",
        subtitle=f"{won_total} побед / {decisive} разрешено",
        border_color=REGIME_COLORS["Low Vol"] if (overall_hr or 0) > 0.5
                      else REGIME_COLORS["High Vol"],
    ), unsafe_allow_html=True)

if not stats:
    st.info(
        "Накапливается. После нескольких рефреш-циклов sidecar'ов начнут "
        "появляться авторы. Через ~14 дней можно будет видеть hit-rate "
        "каждого инфлюенсера."
    )
else:
    rows = []
    for s in stats[:50]:
        hr_str = f"{s['hit_rate']:.0%}" if s["hit_rate"] is not None else "—"
        rows.append({
            "Author": f"@{s['handle']}",
            "Tier": s["tier"],
            "Source": s["source"],
            "Total preds": s["n_total"],
            "Verified": s["n_verified"],
            "Won": s["won"], "Lost": s["lost"], "Expired": s["expired"],
            "Hit rate": hr_str,
            "Avg MFE": f"{s['avg_mfe']:.1%}",
            "Avg MAE": f"{s['avg_mae']:.1%}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True,
                  use_container_width=True)


# ---------------------------------------------------------------------------
# TV bias history chart
# ---------------------------------------------------------------------------
hist = load_tv_history()
if hist:
    section_header("📊 История TV bias по тикерам")
    hdf = pd.DataFrame(hist)
    hdf["logged_at"] = pd.to_datetime(hdf["logged_at"])
    selected = st.multiselect(
        "Тикеры", options=sorted(hdf["ticker"].unique()),
        default=sorted(hdf["ticker"].unique())[:3],
    )
    if selected:
        plot_df = hdf[hdf["ticker"].isin(selected)]
        pivot = plot_df.pivot_table(
            index="logged_at", columns="ticker",
            values="weighted_bias", aggfunc="last",
        )
        st.line_chart(pivot, use_container_width=True)
