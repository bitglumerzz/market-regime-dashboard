"""Wave Knowledge Base — view & manage expert source material.

Shows every source registered in knowledge_base/sources/, the pattern
reference cards, the trading-rules block, and lets you preview the
exact prompt that gets compiled into Claude's system message.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pandas as pd
import streamlit as st

from design_system import (
    ACCENT_CYAN, REGIME_COLORS, TEXT_MUTED, apply_theme,
    metric_card, section_header,
)
from wave_knowledge import (
    KNOWLEDGE_BASE_DIR, SOURCES_DIR,
    Source, build_knowledge_prompt, load_patterns, load_sources,
    load_trading_rules, screenshot_descriptions, stats, transcript_for,
)


apply_theme()

st.markdown(
    '<div style="display:flex; align-items:baseline; gap:14px;">'
    '<div style="font-size:24px; font-weight:700; letter-spacing:-0.02em;">'
    '📚 Wave Knowledge Base</div>'
    '<div style="color:#8B949E; font-size:13px;">База экспертных материалов: '
    'транскрипты, скрины, паттерны, торговые правила. Подмешивается в '
    'system prompt каждого вызова Claude.</div>'
    '</div>',
    unsafe_allow_html=True,
)
st.divider()


# ---------------------------------------------------------------------------
# Overview metrics
# ---------------------------------------------------------------------------
overview = stats()

mc1, mc2, mc3, mc4 = st.columns(4)
with mc1:
    st.markdown(metric_card(
        title="Источников", value=str(overview["n_sources"]),
        subtitle=", ".join(f"{k}:{v}" for k, v in overview["by_type"].items()) or "—",
        border_color=ACCENT_CYAN,
    ), unsafe_allow_html=True)
with mc2:
    st.markdown(metric_card(
        title="Паттернов", value=str(overview["n_patterns"]),
        subtitle="карточек в patterns.yaml",
        border_color=ACCENT_CYAN,
    ), unsafe_allow_html=True)
with mc3:
    st.markdown(metric_card(
        title="Скриншоты (файлы / описания)",
        value=f"{overview['n_screenshot_files']} / {overview['n_screenshot_annotations']}",
        subtitle="png+jpg / yaml entries",
        border_color=TEXT_MUTED,
    ), unsafe_allow_html=True)
with mc4:
    rules_state = "✓" if overview["has_trading_rules"] else "—"
    st.markdown(metric_card(
        title="Trading rules", value=rules_state,
        subtitle="trading_rules.yaml",
        border_color=REGIME_COLORS["Low Vol"] if overview["has_trading_rules"]
                      else TEXT_MUTED,
    ), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_sources, tab_patterns, tab_rules, tab_prompt = st.tabs([
    "📖 Источники", "🌊 Паттерны", "⚙️ Trading rules", "🤖 Compiled prompt",
])


# =============================================================================
# Tab — Sources
# =============================================================================
with tab_sources:
    sources = load_sources()
    if not sources:
        st.info(
            f"Пока нет ни одного источника. Положи папку в "
            f"`{SOURCES_DIR}/<slug>/` с файлом `source.yaml`. "
            f"Опционально — `transcript.txt` и папка `screenshots/` с "
            f"изображениями + `descriptions.yaml`."
        )
    else:
        labels = [f"{s.title or s.slug} ({s.author or '—'})" for s in sources]
        choice = st.selectbox("Выбери источник", options=range(len(sources)),
                                format_func=lambda i: labels[i])
        s = sources[choice]

        st.markdown(f"### {s.title or s.slug}")
        info_bits = []
        if s.author:    info_bits.append(f"**Автор:** {s.author}")
        if s.publisher: info_bits.append(f"**Издатель:** {s.publisher}")
        if s.year:      info_bits.append(f"**Год:** {s.year}")
        info_bits.append(f"**Тип:** {s.type}")
        info_bits.append(f"**Язык:** {s.language}")
        info_bits.append(f"**Tier:** {s.quality_tier}")
        st.markdown(" · ".join(info_bits))

        if s.url:
            st.markdown(f"[🔗 Источник]({s.url})")

        if s.summary:
            with st.expander("📝 Summary", expanded=True):
                st.markdown(s.summary)

        if s.key_takeaways:
            st.markdown("**Key takeaways:**")
            for kt in s.key_takeaways:
                st.markdown(f"- {kt}")

        if s.recommended_books:
            with st.expander("📚 Рекомендованные книги"):
                for b in s.recommended_books:
                    star = "⭐ " if b.get("must_read") else "  "
                    line = f"{star}**{b.get('title','?')}** — {b.get('author','—')}"
                    if b.get("notes"):
                        line += f"  \n_{b['notes']}_"
                    st.markdown(line)

        # Transcript preview
        txt = transcript_for(s.slug)
        if txt:
            with st.expander(f"📄 Transcript ({len(txt):,} символов)"):
                st.text_area("Содержимое", txt, height=400, key=f"tx_{s.slug}")

        # Screenshots
        descs = screenshot_descriptions(s.slug)
        screenshots_folder = os.path.join(s.path, "screenshots")
        files_present = []
        if os.path.isdir(screenshots_folder):
            files_present = sorted(
                fn for fn in os.listdir(screenshots_folder)
                if fn.lower().endswith((".png", ".jpg", ".jpeg"))
            )

        if descs or files_present:
            section_header("🖼️ Screenshots")
            st.caption(
                f"Файлов в папке: **{len(files_present)}**, "
                f"описаний в yaml: **{len(descs)}**."
            )

            # Re-ingest button — runs vision pipeline
            ic1, ic2 = st.columns([1, 3])
            with ic1:
                if st.button("🔄 Re-ingest screenshots (vision)",
                              key=f"ingest_{s.slug}",
                              help="Прогнать новые/непомеченные изображения "
                                   "через Claude vision и добавить описания. "
                                   "~$0.015 за изображение."):
                    with st.spinner(f"Ingesting {len(files_present)} images…"):
                        try:
                            result = subprocess.run(
                                [sys.executable, "wave_knowledge_ingest.py",
                                 "--source", s.slug],
                                capture_output=True, text=True, timeout=600,
                            )
                            st.code(result.stdout or "(no output)")
                            if result.returncode != 0:
                                st.error(f"Ingest вернул код {result.returncode}")
                                if result.stderr:
                                    st.code(result.stderr, language="text")
                        except Exception as exc:
                            st.error(f"Не удалось запустить: "
                                     f"{type(exc).__name__}: {exc}")

            # Each screenshot — image + description
            for d in descs:
                fn = d.get("file", "")
                with st.expander(f"📸 {fn}"):
                    img_path = os.path.join(screenshots_folder, fn) if fn else ""
                    if img_path and os.path.isfile(img_path):
                        st.image(img_path, use_container_width=True)
                    else:
                        st.caption(f"⚠️ Файл `{fn}` отсутствует в папке. "
                                    "Описание ниже всё равно подмешивается в prompt.")
                    if d.get("patterns"):
                        st.markdown(f"**Паттерны:** "
                                     f"{', '.join(d['patterns'])}")
                    if d.get("timeframe") or d.get("symbol"):
                        st.markdown(f"**Контекст:** {d.get('symbol','?')} · "
                                     f"{d.get('timeframe','?')}")
                    if d.get("description"):
                        st.markdown(f"**Описание:** {d['description']}")
                    if d.get("teaching_point"):
                        st.markdown(f"**Главная мысль:** {d['teaching_point']}")
                    if d.get("ingest_error"):
                        st.warning(f"Ingest error: {d['ingest_error']}")


# =============================================================================
# Tab — Patterns
# =============================================================================
with tab_patterns:
    patterns = load_patterns()
    if not patterns:
        st.info(f"Файл `{os.path.join(KNOWLEDGE_BASE_DIR, 'patterns.yaml')}` "
                "пуст или отсутствует.")
    else:
        # Filter
        groups = sorted({p.get("degree_group", "?") for p in patterns})
        f_group = st.multiselect("Фильтр по группе",
                                  options=groups, default=groups)
        filt = [p for p in patterns if p.get("degree_group") in f_group]

        st.markdown(f"Показано **{len(filt)}** паттернов из {len(patterns)}.")

        for p in filt:
            with st.expander(f"**{p.get('ru_name', '?')}** "
                              f"({p.get('en_name', '')}) — "
                              f"{p.get('waves', '?')}"):
                st.markdown(f"`slug: {p.get('slug')}`")

                if p.get("rules_strict"):
                    st.markdown("**Жёсткие правила:**")
                    for r in p["rules_strict"]:
                        st.markdown(f"- 🚨 {r}")
                if p.get("guidelines_soft"):
                    st.markdown("**Soft guidelines:**")
                    for g in p["guidelines_soft"]:
                        st.markdown(f"- 💡 {g}")
                if p.get("common_extensions"):
                    st.markdown("**Расширения / особенности:**")
                    for e in p["common_extensions"]:
                        st.markdown(f"- ↳ {e}")

                ta = p.get("trading_advice") or {}
                if ta:
                    st.markdown("**Trading advice:**")
                    if ta.get("primary_trade"):
                        st.markdown(f"- ▶ **PRIMARY:** {ta['primary_trade']}")
                    if ta.get("secondary_trade"):
                        st.markdown(f"- ▶ **SECONDARY:** {ta['secondary_trade']}")
                    if ta.get("do_not_trade"):
                        st.markdown(f"- ✗ **НЕ ТОРГОВАТЬ:** "
                                    f"{', '.join(ta['do_not_trade'])}")
                    if ta.get("entry_model"):
                        st.markdown(f"- ⊕ **Entry:** {ta['entry_model']}")
                    if ta.get("stop_placement"):
                        st.markdown(f"- ⊗ **Stop:** {ta['stop_placement']}")
                    if ta.get("target_min"):
                        st.markdown(f"- 🎯 **Target min:** {ta['target_min']}")
                    if ta.get("target_ideal"):
                        st.markdown(f"- 🎯 **Target ideal:** {ta['target_ideal']}")
                    if ta.get("rationale"):
                        st.caption(ta["rationale"])

                if p.get("sources"):
                    st.caption(f"Источники: {', '.join(p['sources'])}")


# =============================================================================
# Tab — Trading rules
# =============================================================================
with tab_rules:
    rules = load_trading_rules()
    if not rules:
        st.info("`trading_rules.yaml` пуст или отсутствует.")
    else:
        st.markdown(f"### Система: {rules.get('system','?')}")
        if rules.get("sources"):
            st.caption(f"Источники: {', '.join(rules['sources'])}")

        if rules.get("tradable_waves"):
            st.markdown("**Торгуемые волны (PRIMARY):**")
            for w in rules["tradable_waves"]:
                st.markdown(f"- ✓ **{w.get('location')}** "
                            f"({w.get('confidence','—')}) — {w.get('why','')}")
        if rules.get("non_tradable_waves"):
            st.markdown("**НЕ торгуемые волны:**")
            for w in rules["non_tradable_waves"]:
                st.markdown(f"- ✗ **{w.get('location')}** — {w.get('why','')}")

        em = rules.get("primary_entry_model") or {}
        if em:
            st.markdown(f"### Entry model: «{em.get('name','?')}»")
            if em.get("description"):
                st.markdown(em["description"])
            if em.get("preconditions"):
                st.markdown("**Preconditions:**")
                for p in em["preconditions"]:
                    st.markdown(f"- {p}")
            st.markdown(f"**Entry:** {em.get('entry','—')}")
            st.markdown(f"**Stop:** {em.get('stop','—')}")
            st.markdown(f"**R/R target:** {em.get('initial_risk_reward_target','—')}")

        sm = rules.get("stop_management") or {}
        if sm:
            st.markdown("### Stop management")
            for r in sm.get("rules", []) or []:
                st.markdown(f"- {r}")
            if sm.get("rationale"):
                st.caption(sm["rationale"])

        if rules.get("common_mistakes"):
            with st.expander("⚠️ Частые ошибки и как их избегать"):
                for m in rules["common_mistakes"]:
                    st.markdown(f"- **{m.get('mistake')}** → {m.get('fix','')}")

        if rules.get("notation"):
            with st.expander("🔤 Волновая нотация"):
                hier = rules["notation"].get("hierarchy") or []
                for h in hier:
                    st.markdown(f"- **{h.get('degree')}** — {h.get('style')}")
                if rules["notation"].get("importance"):
                    st.markdown(rules["notation"]["importance"])


# =============================================================================
# Tab — Compiled prompt (what Claude actually sees)
# =============================================================================
with tab_prompt:
    st.caption(
        "Это точный блок который подмешивается в `system` параметр каждого "
        "вызова Claude (per-TF и holistic). Можно проверить что именно видит "
        "модель и сравнить разные конфигурации."
    )
    pc1, pc2 = st.columns(2)
    with pc1:
        include_rules = st.checkbox("Включить trading rules", value=True)
    with pc2:
        include_sources = st.checkbox("Включить bibliography", value=True)

    focus_input = st.text_input(
        "Фокусировать на паттернах (slugs через запятую, пусто = все)",
        value="",
        help="Например: `impulse_up,leading_diagonal,zigzag_correction`",
    )
    focus = [s.strip() for s in focus_input.split(",") if s.strip()] or None

    prompt = build_knowledge_prompt(
        focus_patterns=focus,
        include_trading_rules=include_rules,
        include_sources_list=include_sources,
        max_chars=20000,  # show full thing on this preview page
    )

    pl1, pl2, pl3 = st.columns(3)
    with pl1:
        st.markdown(metric_card(
            title="Длина prompt",
            value=f"{len(prompt):,} симв.",
            subtitle=f"≈ {len(prompt)//4:,} токенов",
            border_color=ACCENT_CYAN,
        ), unsafe_allow_html=True)
    with pl2:
        # Rough cost estimate at Opus pricing
        approx_tokens = len(prompt) // 4
        cost_per_call = approx_tokens / 1_000_000 * 15.0  # $15/M input @ Opus
        st.markdown(metric_card(
            title="Cost / вызов (Opus)",
            value=f"${cost_per_call:.4f}",
            subtitle="input only",
            border_color=TEXT_MUTED,
        ), unsafe_allow_html=True)
    with pl3:
        n_pat = len(load_patterns()) if focus is None else len(focus)
        st.markdown(metric_card(
            title="Карточек в prompt",
            value=str(n_pat),
            subtitle="паттернов",
            border_color=TEXT_MUTED,
        ), unsafe_allow_html=True)

    st.markdown("**Compiled system prompt:**")
    st.code(prompt, language="markdown")
