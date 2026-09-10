"""Translate detector's English Elliott-wave descriptions to Russian.

The wave detector (elliott.py) emits English `current_position` strings
like "completed regular up flat — next likely strong impulse in opposite
direction" or "contracting triangle complete — expect DOWN thrust to 2208".

This module provides ``translate_detector_text(text)`` which runs an
ordered series of regex/literal replacements to produce Russian output,
keeping numbers, prices, and slugs untouched.

Used in two places:
  * pages/4_Elliott_Waves.py — Журнал действий transitions display
  * wave_journal.py — when formatting transitions for display

Intentionally simple and additive — if a new English phrase appears,
add a row here without touching elliott.py.
"""
from __future__ import annotations

import re


# Ordered: longer/more-specific phrases first to avoid partial replacements.
_REPLACEMENTS: list[tuple[str, str]] = [
    # Full phrases — these are exact patterns used by elliott.py
    ("market likely in subsequent correction now",
     "рынок вероятно в последующей коррекции"),
    ("next likely strong impulse in opposite direction",
     "далее ожидается сильный импульс в противоположную сторону"),
    ("strong impulse in opposite direction expected",
     "ожидается сильный импульс в противоположную сторону"),
    ("completed expanded down flat",
     "завершена расширенная плоская коррекция вниз"),
    ("completed expanded up flat",
     "завершена расширенная плоская коррекция вверх"),
    ("completed running down flat",
     "завершена running плоская коррекция вниз"),
    ("completed running up flat",
     "завершена running плоская коррекция вверх"),
    ("completed regular up flat",
     "завершена обычная плоская коррекция вверх"),
    ("completed regular down flat",
     "завершена обычная плоская коррекция вниз"),
    ("contracting triangle complete",
     "сжимающийся треугольник завершён"),
    ("expanding triangle complete",
     "расширяющийся треугольник завершён"),
    ("barrier triangle complete",
     "barrier треугольник завершён"),
    # "(impulse up completed 5 swing(s) ago — market likely in ..."
    ("impulse up completed",  "импульс вверх завершён"),
    ("impulse down completed", "импульс вниз завершён"),
    ("impulse up completed 1 swing(s) ago", "импульс вверх завершён 1 свинг назад"),
    ("expect DOWN thrust to",  "ожидается выброс ВНИЗ к"),
    ("expect UP thrust to",    "ожидается выброс ВВЕРХ к"),
    ("expect thrust to",       "ожидается выброс к"),
    ("post-flat thrust",       "выброс после плоской"),
    ("post-triangle thrust",   "выброс после треугольника"),
    ("not enough swings for",  "недостаточно свингов для"),
    ("with violations",        "с нарушениями"),
    # Wave-direction shorthand
    ("hypothesis",            "гипотеза"),

    # Words that often appear within phrases
    (" swing(s) ago",         " свинг(ов) назад"),
    (" swings ago",           " свингов назад"),
    (" swing ago",            " свинг назад"),
    ("completed ",            "завершён "),
    ("expected ",             "ожидается "),
    ("expect ",               "ожидается "),
    ("complete ",             "завершён "),
    ("active ",               "активный "),
    # Phase 38 — extra technical terms that still slip through in
    # Claude's holistic narrative even with the prompt rule
    ("expanded flat",         "расширенная плоская"),
    ("regular flat",          "обычная плоская"),
    ("alternation rule",      "правило чередования (Прехтер)"),
    ("Alternation rule",      "Правило чередования (Прехтер)"),
    ("Multi-TF confluence",   "Сводка по нескольким ТФ"),
    ("Multi-TF",              "По нескольким ТФ"),
    ("multi-TF",              "по нескольким ТФ"),
    ("Cross-TF",              "По таймфреймам"),
    ("cross-TF",              "по таймфреймам"),
    ("Cancel:",               "Отмена:"),
    ("Cancel ",               "Отмена "),
    ("cancel ",               "отмена "),
    ("Re-check",              "Перепроверка"),
    ("re-check",              "перепроверка"),
    ("structure incomplete",  "структура незавершена"),
    ("structure complete",    "структура завершена"),
    ("stoploss",              "стоп"),
    ("stop loss",             "стоп"),
    ("breakdown",             "пробой вниз"),
    ("breakout",              "пробой"),
    ("bounce",                "отскок"),
    ("retrace",               "ретрейс"),
    ("retracement",           "ретрейс"),
    ("invalidation",          "инвалидация"),
    ("valid",                 "валидный"),
    ("invalid",               "невалидный"),
    ("setup",                 "сетап"),
    ("entry zone",            "зона входа"),
    ("target",                "цель"),
    # Pavelko terminology
    ("sweet spot",            "оптимальная зона"),
    ("sweet-spot",            "оптимальная зона"),
    ("playbook",              "стратегия"),
    ("thrust",                "выброс"),
    ("post-flat",             "после плоской"),
    ("post-triangle",         "после треугольника"),
    # Prechter terminology
    ("ending diagonal",       "конечная диагональ"),
    ("leading diagonal",      "ведущая диагональ"),
    ("truncated",             "укороченная"),
    ("truncation",            "укорочение"),
    ("channeling",            "построение канала"),
    # General trading
    ("smart money",           "smart money (институционалы)"),
    ("follower",              "последователь"),
    ("leader",                "лидер"),
    ("contrarian",            "контр-трендовый"),
    ("divergent",             "расхождение"),
    ("aligned",               "выровнен"),
    ("alignment",             "выравнивание"),
    ("confidence",            "уверенность"),
    ("confluence",            "confluence (сходимость)"),
    # Phase 38 part 2 — case variants and missed terms
    ("Expanded flat",         "Расширенная плоская"),
    ("Regular flat",          "Обычная плоская"),
    ("Running flat",          "Running плоская"),  # term kept latin
    ("Triangle",              "Треугольник"),
    ("Active impulse",        "Активный импульс"),
    ("active impulse",        "активный импульс"),
    ("Risk:",                 "Риск:"),
    ("rule-engine",           "детектор-правил"),
    ("incomplete",            "незавершён"),
    ("swings",                "свинги"),
    ("Support",               "Поддержка"),
    (" support",              " поддержка"),
    ("Resistance",            "Сопротивление"),
    (" resistance",           " сопротивление"),
    ("hit",                   "достиг"),
    ("active",                "активный"),
    ("Active",                "Активный"),
    ("WAIT",                  "ЖДЁМ"),
    ("wait,",                 "ждём,"),
    ("WAIT,",                 "ЖДЁМ,"),
    ("Wait ",                 "Ждём "),
    ("equity",                "капитал"),
    ("equality",              "равенство"),
    ("structure",             "структура"),
    ("Structure",             "Структура"),
    # Phase 38 part 3 — common remaining English phrases
    ("Breakout direction defines trade",
                              "направление пробоя определит сделку"),
    ("breakout direction",    "направление пробоя"),
    ("no-man's-land",         "ничейная зона"),
    ("no man's land",         "ничейная зона"),
    ("truncate here",         "укоротиться здесь"),
    ("truncate",              "укоротиться"),
    ("extend to",             "продлиться к"),
    ("extend",                "продлиться"),
    ("clarity",               "ясность"),
    ("retest",                "ретест"),
    ("pivot",                 "разворотная точка"),
    ("range",                 "диапазон"),
    ("momentum",              "моментум"),
    ("last line of defense",  "последняя линия защиты"),
    ("BULLISH",               "БЫЧИЙ"),
    ("BEARISH",               "МЕДВЕЖИЙ"),
    ("Bullish",               "Бычий"),
    ("Bearish",               "Медвежий"),
    ("bullish",               "бычий"),
    ("bearish",               "медвежий"),
    ("continuation pattern",  "паттерн продолжения"),
    ("continuation",          "продолжение"),
    ("reversal zone",         "зона разворота"),
    ("reversal",              "разворот"),
    ("ETF",                   "ETF"),  # keep
    ("integration",           "интеграция"),
    ("HIT",                   "ДОСТИГНУТ"),

    # Pattern names (used in transitions like "flat_up → impulse_down")
    # Keep the slug itself, but provide a Russian gloss for display.
    # These are conservative — only the bare lowercase form (not slugs).
]


# Pattern slug → Russian gloss (used in transitions display when showing
# the slug name)
PATTERN_SLUG_RU: dict[str, str] = {
    "impulse_up":            "импульс вверх",
    "impulse_down":          "импульс вниз",
    "correction_up":         "коррекция вверх",
    "correction_down":       "коррекция вниз",
    "complex_correction_up":   "сложная коррекция вверх (WXY)",
    "complex_correction_down": "сложная коррекция вниз (WXY)",
    "flat_up":               "плоская вверх",
    "flat_down":             "плоская вниз",
    "triangle_up":           "треугольник вверх",
    "triangle_down":         "треугольник вниз",
    "leading_diagonal_up":   "ведущая диагональ вверх",
    "leading_diagonal_down": "ведущая диагональ вниз",
    "ending_diagonal_up":    "конечная диагональ вверх",
    "ending_diagonal_down":  "конечная диагональ вниз",
    "zigzag_up":             "зигзаг вверх",
    "zigzag_down":           "зигзаг вниз",
    "truncated_fifth_up":    "укороченная 5-я вверх",
    "truncated_fifth_down":  "укороченная 5-я вниз",
}


def translate_detector_text(text: str) -> str:
    """Run all replacements on a single line. Numbers and slugs untouched."""
    if not text:
        return text
    out = text
    for en, ru in _REPLACEMENTS:
        out = out.replace(en, ru)
    return out


def translate_pattern_slug(slug: str) -> str:
    """Return Russian gloss for a pattern slug, or the slug itself."""
    if not slug:
        return slug
    return PATTERN_SLUG_RU.get(slug.lower(), slug)


def translate_transition_message(text: str) -> str:
    """Translate a full transition narrative.

    Input example:
      "flat_up (completed regular up flat — next likely strong impulse
       in opposite direction — completed 1 swing(s) ago) → impulse_down
       (impulse down completed 1 swing(s) ago — market likely in
       subsequent correction now)"

    Output:
      "плоская вверх (завершена обычная плоская коррекция вверх —
       далее ожидается сильный импульс в противоположную сторону —
       завершён 1 свинг(ов) назад) → импульс вниз (импульс вниз
       завершён 1 свинг(ов) назад — рынок вероятно в последующей
       коррекции)"
    """
    if not text:
        return text
    # First translate the descriptive phrases
    out = translate_detector_text(text)
    # Then replace pattern slugs (whole-word matches) with Russian glosses
    for slug, ru in sorted(PATTERN_SLUG_RU.items(), key=lambda x: -len(x[0])):
        # Match slug at word boundary (avoid replacing part of longer slug)
        out = re.sub(rf"\b{re.escape(slug)}\b", ru, out)
    return out
