"""Twitter sentiment dashboard.

Shows per-ticker sentiment broken down by tier (market movers / quality
analysts / crowd), with top posts and divergence signals.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from design_system import (
    ACCENT_CYAN, BORDER_SUBTLE, REGIME_COLORS, TEXT_MUTED, TEXT_PRIMARY,
    apply_theme, metric_card, section_header,
)
from influencer_registry import all_accounts, summary as registry_summary
from twitter_sentiment import (
    SENTIMENT_INDEX_PATH, load_sentiment_index, refresh_from_snapshots,
)
from twitter_scraper import (
    SNAPSHOTS_DIR, list_snapshots, scrape, save_snapshot, _apify_token,
)


apply_theme()

st.markdown(
    '<div style="display:flex; align-items:baseline; gap:14px;">'
    '<div style="font-size:24px; font-weight:700; letter-spacing:-0.02em;">'
    '🐦 Twitter Sentiment</div>'
    '<div style="color:#8B949E; font-size:13px;">'
    'Реакция Crypto Twitter в трёх тиерах — market movers, аналитики, толпа.'
    '</div></div>',
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Action bar
# ---------------------------------------------------------------------------
idx = load_sentiment_index()
last_updated = idx.get("updated_at", "—")
n_tickers = idx.get("n_tickers", 0)

ac1, ac2, ac3 = st.columns([1, 1, 2])
with ac1:
    if st.button("🐦 Скачать твиты сейчас", type="primary"):
        with st.spinner("Скрапим Twitter через Apify…"):
            try:
                tweets = scrape(hours_back=24, mock=False)
                if tweets:
                    save_snapshot(tweets)
                    refresh_from_snapshots(within_hours=48)
                    st.success(f"Получено {len(tweets)} твитов, sentiment пересчитан.")
                else:
                    st.warning("Apify вернул 0 твитов. Проверь APIFY_API_TOKEN в .env "
                                "или Apify актор-ID.")
            except Exception as exc:
                st.error(f"Ошибка скрапа: {type(exc).__name__}: {exc}")
        st.rerun()

with ac2:
    if st.button("🧪 Использовать mock-данные"):
        tweets = scrape(mock=True)
        if tweets:
            save_snapshot(tweets)
            refresh_from_snapshots(within_hours=48)
            st.info(f"Загружено {len(tweets)} mock-твитов.")
        st.rerun()

with ac3:
    token_status = "🟢 настроен" if _apify_token() else "🔴 не настроен (.env: APIFY_API_TOKEN)"
    st.markdown(
        f'<div style="color:#8B949E; font-size:12px; line-height:1.6;">'
        f'Последнее обновление: <b>{last_updated[:16] if last_updated else "—"}</b><br/>'
        f'Apify токен: {token_status}<br/>'
        f'Тикеров в индексе: {n_tickers}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Registry overview
# ---------------------------------------------------------------------------
section_header("Реестр инфлюенсеров")
reg = registry_summary()
rc1, rc2, rc3, rc4 = st.columns(4)
with rc1:
    st.markdown(metric_card(
        title="Всего аккаунтов", value=str(reg["total"]),
        subtitle="отслеживаем", border_color=ACCENT_CYAN,
    ), unsafe_allow_html=True)
with rc2:
    st.markdown(metric_card(
        title="Tier 1 (×3)", value=str(reg["by_tier"].get("tier_1", 0)),
        subtitle="market movers", border_color=REGIME_COLORS["High Vol"],
    ), unsafe_allow_html=True)
with rc3:
    st.markdown(metric_card(
        title="Tier 2 (×2)", value=str(reg["by_tier"].get("tier_2", 0)),
        subtitle="quality analysts", border_color=ACCENT_CYAN,
    ), unsafe_allow_html=True)
with rc4:
    st.markdown(metric_card(
        title="Tier 3 (×1)", value=str(reg["by_tier"].get("tier_3", 0)),
        subtitle="crowd / news", border_color=TEXT_MUTED,
    ), unsafe_allow_html=True)

with st.expander("📋 Полный список аккаунтов"):
    rows = []
    for a in all_accounts():
        rows.append({
            "Tier": a.tier,
            "Weight": a.weight,
            "Handle": f"@{a.handle}",
            "Name": a.name,
            "Focus": a.focus,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Per-ticker sentiment
# ---------------------------------------------------------------------------
section_header("Sentiment по тикерам")
tickers_data = idx.get("tickers", {})
if not tickers_data:
    st.info(
        "Пока нет данных. Нажми **🐦 Скачать твиты сейчас** или используй "
        "**🧪 mock-данные** для теста."
    )
    st.stop()

for ticker in sorted(tickers_data.keys()):
    t_data = tickers_data[ticker]
    score = t_data.get("weighted_score", 0)
    label = t_data.get("weighted_label", "neutral").upper()
    n_tweets = t_data.get("n_tweets_total", 0)
    color = (REGIME_COLORS["Low Vol"] if score > 0.15
              else REGIME_COLORS["High Vol"] if score < -0.15
              else TEXT_MUTED)
    emoji = "🟢" if score > 0.15 else "🔴" if score < -0.15 else "⚪"

    st.markdown(
        f'<div style="padding:14px 18px; border-left: 4px solid {color}; '
        f'background:#161B22; border-radius: 8px; margin-bottom: 12px;">'
        f'<div style="font-size:11px; color:#8B949E; text-transform:uppercase; '
        f'letter-spacing:.08em; margin-bottom:6px;">{ticker} · {n_tweets} постов за 24ч</div>'
        f'<div style="font-size:20px; font-weight:700; color:{color};">'
        f'{emoji} {label} · score {score:+.2f}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Per-tier breakdown
    by_tier = t_data.get("by_tier", {})
    tier_cols = st.columns(3)
    for i, tier_label in enumerate(("tier_1", "tier_2", "tier_3")):
        with tier_cols[i]:
            d = by_tier.get(tier_label)
            if not d:
                st.markdown(
                    f'<div style="color:#6E7B8B; font-size:13px; padding:8px;">'
                    f'{tier_label}: нет постов</div>',
                    unsafe_allow_html=True,
                )
                continue
            tier_mean = d["mean_sentiment"]
            tier_color = (REGIME_COLORS["Low Vol"] if tier_mean > 0.15
                            else REGIME_COLORS["High Vol"] if tier_mean < -0.15
                            else TEXT_MUTED)
            tier_human = {"tier_1": "🎯 Tier-1",
                          "tier_2": "📊 Tier-2",
                          "tier_3": "👥 Tier-3"}[tier_label]
            st.markdown(
                f'<div style="padding:10px 12px; border:1px solid {tier_color}; '
                f'border-radius: 6px;">'
                f'<div style="font-size:11px; color:#8B949E;">{tier_human} · '
                f'{d["n_tweets"]} постов</div>'
                f'<div style="font-size:17px; font-weight:700; color:{tier_color};">'
                f'{tier_mean:+.2f}</div>'
                f'<div style="font-size:11px; color:#8B949E;">{d["label"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Divergence callout
    if t_data.get("divergence"):
        st.warning(t_data["divergence"])

    # Top posts table
    all_top = []
    for tier_label, d in by_tier.items():
        for tp in d.get("top_posts", []):
            all_top.append({
                "Tier": tier_label,
                "Author": f"@{tp['handle']}",
                "Sentiment": tp["sentiment"],
                "Label": tp["label"],
                "Text": tp["text"],
                "When": (tp.get("created_at") or "")[:16],
                "URL": tp.get("url", ""),
            })
    if all_top:
        df = pd.DataFrame(all_top).sort_values("Sentiment", key=lambda s: s.abs(),
                                                 ascending=False)
        st.dataframe(df, hide_index=True, use_container_width=True)
    st.divider()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
with st.expander("🔧 Diagnostics — snapshot files"):
    snaps = list_snapshots(within_hours=72)
    st.markdown(f"Snapshots за 72ч: **{len(snaps)}**, путь: `{SNAPSHOTS_DIR}/`")
    for s in snaps[-10:]:
        st.code(s)
