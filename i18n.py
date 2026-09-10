"""Lightweight i18n for trading-decision UI strings.

Only the strings that drive a *decision* are translated to Russian. Technical
table headers, debug expanders, and the like stay in English (you read the
numbers, not the labels). This keeps the diff small and the dashboard fast
to scan.

Usage:
    from i18n import t
    st.markdown(t("setup.long_entry_now"))
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Translation dictionary — keep keys stable, only add new languages later.
# ---------------------------------------------------------------------------
_RU: dict[str, str] = {
    # ─── Top-Down Synthesis verdicts ────────────────────────────────────
    "setup.long_entry_now":
        "🎯 ВХОД В ЛОНГ СЕЙЧАС: 1d рост, 4h откат, 15m завершён, 5m подтверждает",
    "setup.long_opportunity":
        "ВОЗМОЖНОСТЬ ЛОНГА: 1d рост, 4h откат, 15m коррекция завершена",
    "setup.close_to_long":
        "Близко к ЛОНГУ: макро согласовано, ждём подтверждения 5m",
    "setup.running_long":
        "Уже в лонге — 1d/4h/15m все вверх",
    "setup.running_long_5m_against":
        "В лонге, но 5m показывает откат — рассмотри частичную фиксацию",
    "setup.wait_15m_long":
        "Ждём завершения коррекции 15m (для лонга)",
    "setup.wait_4h_long":
        "1d вверх, но структура 4h неясная — ждём",

    "setup.short_entry_now":
        "🎯 ВХОД В ШОРТ СЕЙЧАС: 1d падение, 4h отскок, 15m завершён, 5m подтверждает",
    "setup.short_opportunity":
        "ВОЗМОЖНОСТЬ ШОРТА: 1d падение, 4h отскок, 15m коррекция завершена",
    "setup.close_to_short":
        "Близко к ШОРТУ: макро согласовано, ждём подтверждения 5m",
    "setup.running_short":
        "Уже в шорте — 1d/4h/15m все вниз",
    "setup.running_short_5m_against":
        "В шорте, но 5m показывает отскок — рассмотри частичную фиксацию",
    "setup.wait_15m_short":
        "Ждём завершения коррекции 15m (для шорта)",
    "setup.wait_4h_short":
        "1d вниз, но структура 4h неясная — ждём",

    "setup.in_correction":
        "1d в коррекции — плохая среда для трендовой торговли. Ждём чистого импульса.",
    "setup.unclear":
        "Стоим в стороне — картина неопределённая",

    # ─── Card titles / subtitles ────────────────────────────────────────
    "card.macro": "1d МАКРО",
    "card.meso":  "4h МЕЗО",
    "card.micro": "15m МИКРО",
    "card.entry": "5m ВХОД",

    "card.macro.sub":  "Определяет смещение (bias)",
    "card.meso.sub":   "Подтверждает подструктуру",
    "card.micro.sub":  "Мид-структура / сетап",
    "card.entry.sub":  "🎯 Тайминг живого входа",

    "card.trading_setup": "ТОРГОВЫЙ СЕТАП",

    # ─── Detector backtest interpretation ───────────────────────────────
    "backtest.hit_signal_real": "> 60% = сигнал реален",
    "backtest.hit_noise":       "40–60% = шум / переобучение",
    "backtest.hit_backwards":   "< 40% = настройки систематически инвертированы",
    "backtest.hits_misses_amb": "{hits} попаданий / {misses} промахов / {amb} неоднозначных",
    "backtest.cases_no_decisive": "недостаточно решительных движений",

    # ─── Section headers (just the ones with trade meaning) ─────────────
    "section.synthesis":      "Top-Down Синтез",
    "section.per_tf":         "Детали по таймфреймам",
    "section.backtest":       "Бэктест детектора — насколько верны волны?",
    "section.claude_opinion": "Второе мнение Claude AI (опционально)",
    "section.how_to_read":    "Как это читать",

    # ─── Reading guide (the most important trading-decision text) ───────
    "guide.title": "**Top-down процесс (метод Андреева/Эллиотта):**",
    "guide.step_1":
        "1. **1d МАКРО** задаёт *смещение*. Если 1d говорит «impulse up wave 5 of 5» — "
        "ты у конца тренда, осторожнее. Если «impulse up wave 3 of 5» — "
        "ждём продолжения, торгуем по тренду.",
    "guide.step_2":
        "2. **4h МЕЗО** — это *подструктура внутри текущей 1d-волны*. "
        "Бычий 1d + бычий 4h = сильнейший сетап. Бычий 1d + 4h коррекция = "
        "возможность входа на откате.",
    "guide.step_3":
        "3. **15m МИКРО** *задаёт время входа*. Ищем конец 15m-коррекции "
        "внутри 4h-импульса — там входит Андреев.",
    "guide.step_4":
        "4. **5m ВХОД** *подтверждает живой момент*. Если 5m против — ждём.",

    "guide.rules_title": "**Ключевые правила (проверяются автоматически):**",
    "guide.rule_r0":
        "• **R0** — направление: \"импульс вверх\" должен заканчиваться выше начала. "
        "Без R0 боковые колебания считались бы импульсами. **Самое важное правило.**",
    "guide.rule_r1":
        "• **R1** — волна 2 не откатывает больше 100% волны 1.",
    "guide.rule_r2":
        "• **R2** — волна 3 не самая короткая из 1, 3, 5.",
    "guide.rule_r3":
        "• **R3** — волна 4 не заходит в зону волны 1.",
    "guide.score":
        "**Score** = 0–100 по жёстким правилам (R0–R3) + соответствие "
        "Фибоначчи + бонус за размер движения.",
    "guide.alternatives":
        "**Всегда смотри альтернативы** — если топ-кандидат 60, а второй 55, "
        "структура неоднозначная. Ждём.",

    # ─── Claude AI second opinion ───────────────────────────────────────
    "claude.title":          "Спросить Claude о таймфрейме?",
    "claude.btn":            "Получить мнение Claude",
    "claude.agree":          "Соглашается с топ-гипотезой?",
    "claude.preferred":      "Предпочтительная разметка по мнению Claude",
    "claude.confidence":     "Уверенность",
    "claude.reasoning":      "Обоснование:",
    "claude.no_key":         "Ключ API не задан. Создай `.env` файл с ANTHROPIC_API_KEY.",
    "claude.calling":        "Спрашиваю Claude…",
    "claude.action":         "Рекомендуемое действие",
    "claude.entry":          "Точка входа",
    "claude.stop":           "Стоп-лосс",
    "claude.target":         "Цель (тейк-профит)",

    # ─── Status labels ──────────────────────────────────────────────────
    "status.long":  "ЛОНГ",
    "status.short": "ШОРТ",
    "status.wait":  "ЖДЁМ",

    # ─── Forecast (next-move prediction) ────────────────────────────────
    "section.forecast":           "Прогноз следующего движения",
    "forecast.pattern":           "Ожидаемый паттерн",
    "forecast.direction":         "Направление",
    "forecast.confidence":        "Уверенность прогноза",
    "forecast.invalidation":      "Уровень отмены сценария",
    "forecast.duration":          "Ожидаемая длительность",
    "forecast.targets":           "Ценовые цели",
    "forecast.rationale":         "Обоснование",
    "forecast.no_forecast":       "Прогноз недоступен — структура неясная",
    "forecast.duration_bars":     "{lo}–{hi} баров",

    "forecast.pattern.correction_down":  "🔻 Зигзаг вниз (A-B-C, 5-3-5)",
    "forecast.pattern.correction_up":    "🔺 Зигзаг вверх (A-B-C, 5-3-5)",
    "forecast.pattern.impulse_down":     "🔻 Импульс вниз (5 волн)",
    "forecast.pattern.impulse_up":       "🔺 Импульс вверх (5 волн)",
    # Phase-1 patterns
    "forecast.pattern.flat_down":        "🔻 Плоскость вниз (A-B-C, 3-3-5)",
    "forecast.pattern.flat_up":          "🔺 Плоскость вверх (A-B-C, 3-3-5)",
    "forecast.pattern.triangle_down":    "🔻 Треугольник (вниз-thrust, A-B-C-D-E)",
    "forecast.pattern.triangle_up":      "🔺 Треугольник (вверх-thrust, A-B-C-D-E)",
    # Phase-2 patterns
    "forecast.pattern.leading_diagonal_up":   "🔺 Leading diagonal вверх (клин, начало тренда)",
    "forecast.pattern.leading_diagonal_down": "🔻 Leading diagonal вниз (клин, начало тренда)",
    "forecast.pattern.ending_diagonal_up":    "🔺 Ending diagonal вверх (клин разворота)",
    "forecast.pattern.ending_diagonal_down":  "🔻 Ending diagonal вниз (клин разворота)",
    "forecast.pattern.complex_correction_up":   "🔺 WXY коррекция вверх (double three)",
    "forecast.pattern.complex_correction_down": "🔻 WXY коррекция вниз (double three)",
    "forecast.pattern.unknown":          "— неопределённо",
    # Extension badge
    "wave.extended":                     "⚡ РАСШИРЕННЫЙ ИМПУЛЬС",
    "wave.extended.note":                "Волна 3 = {ratio:.2f}× Волны 1 — следующая волна 5 будет ≈ Волне 1",
    # Truncation badge
    "wave.truncated":                    "⚠️ ИМПУЛЬС-ОБРЫВ (truncated fifth)",
    "wave.truncated.note":               "Волна 5 не пробила вершину/дно волны 3 — сильный сигнал разворота",
    # Diagonal badge
    "wave.diagonal.ending":              "🔻 ENDING DIAGONAL — резкий разворот ожидается",
    "wave.diagonal.leading":             "🆕 LEADING DIAGONAL — начало нового тренда (после коррекции)",

    "forecast.dir.up":      "вверх ↑",
    "forecast.dir.down":    "вниз ↓",
    "forecast.dir.unclear": "неясно",
}


def t(key: str, **kwargs) -> str:
    """Translate a key. Falls back to the key itself if not found.

    Supports str.format-style substitutions: `t("backtest.hits_misses_amb",
    hits=10, misses=2, amb=1)`.
    """
    text = _RU.get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, IndexError):
            pass
    return text


# ---------------------------------------------------------------------------
# Reverse mapping for synthesizer hardcoded English → Russian
# multi_tf.top_down_synthesis emits English markers like
# "LONG ENTRY NOW: ...". We post-translate those at render time.
# ---------------------------------------------------------------------------
_VERDICT_MAP: dict[str, str] = {
    "🎯 LONG ENTRY NOW:": "🎯 ВХОД В ЛОНГ СЕЙЧАС:",
    "🎯 SHORT ENTRY NOW:": "🎯 ВХОД В ШОРТ СЕЙЧАС:",
    "LONG opportunity:": "ВОЗМОЖНОСТЬ ЛОНГА:",
    "SHORT opportunity:": "ВОЗМОЖНОСТЬ ШОРТА:",
    "Close to LONG":      "Близко к ЛОНГУ",
    "Close to SHORT":     "Близко к ШОРТУ",
    "Already running long": "Уже в лонге",
    "Already running short": "Уже в шорте",
    "Wait for 15m correction to complete": "Ждём завершения коррекции 15m",
    "Wait for 5m to confirm":               "Ждём подтверждения 5m",
    "wait for 5m to confirm":               "ждём подтверждения 5m",
    "Running long but 5m showing correction": "В лонге, но 5m откатывается",
    "Running short but 5m showing bounce":    "В шорте, но 5m отскакивает",
    "Stand aside — picture unclear":          "Стоим в стороне — картина неопределённая",
    "1d up but 4h structure unclear":         "1d вверх, но 4h структура неясная",
    "1d down but 4h structure unclear":       "1d вниз, но 4h структура неясная",
    "1d is in a correction":                  "1d в коррекции",
    "wait":                                   "ждём",
    "partial profit?":                        "частичная фиксация?",
    "trade with it":                          "торгуем с ним",
    "be cautious":                            "будь осторожен",
    "macro aligned":                          "макро согласовано",
    "1d up, 4h pull-back, 15m done, 5m confirms":
        "1d вверх, 4h откат, 15m завершён, 5m подтверждает",
    "1d down, 4h bounce, 15m done, 5m confirms":
        "1d вниз, 4h отскок, 15m завершён, 5m подтверждает",
    "1d up, 4h pull-back, 15m correction completed":
        "1d вверх, 4h откат, 15m коррекция завершена",
    "1d down, 4h bounce, 15m correction completed":
        "1d вниз, 4h отскок, 15m коррекция завершена",
    "1d/4h/15m all up":  "1d/4h/15m все вверх",
    "1d/4h/15m all down": "1d/4h/15m все вниз",
    "historically a poor environment for trend-following":
        "исторически плохая среда для трендовой торговли",
    "Wait for the next clean impulse":
        "Ждём следующий чистый импульс",
}


def translate_verdict(text: str) -> str:
    """Post-process an English verdict produced by top_down_synthesis."""
    out = text
    for en, ru in _VERDICT_MAP.items():
        out = out.replace(en, ru)
    return out
