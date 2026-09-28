"""Тексты кнопок и форматирование."""
from __future__ import annotations

from datetime import date, datetime

BTN_BOOK = "💅 Записаться"
BTN_MY = "📅 Мои записи"
BTN_PRICES = "💰 Услуги и цены"
BTN_CONTACTS = "📍 Контакты"
BTN_ASK = "💬 Вопрос администратору"

WEEKDAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля",
          "августа", "сентября", "октября", "ноября", "декабря"]


def short_day(d: date) -> str:
    return f"{WEEKDAYS[d.weekday()]} {d:%d.%m}"


def human_dt(dt: datetime) -> str:
    return f"{WEEKDAYS[dt.weekday()]}, {dt.day} {MONTHS[dt.month - 1]} в {dt:%H:%M}"


def price(value) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):,.0f} ₽".replace(",", " ")
    except (TypeError, ValueError):
        return str(value)


def duration(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    return " ".join(p for p in (f"{h} ч" if h else "", f"{m} мин" if m else "") if p)
