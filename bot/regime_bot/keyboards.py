"""Инлайн-клавиатуры."""
from __future__ import annotations

from aiogram.types import InlineKeyboardButton as B, InlineKeyboardMarkup as KB

from .config import Settings


def main_menu(settings: Settings) -> KB:
    support = settings.support.lstrip("@")
    return KB(inline_keyboard=[
        [B(text="🤖 Как это работает", callback_data="how"), B(text="📈 Пример сигнала", callback_data="example")],
        [B(text="💎 Тарифы и доступ", callback_data="plans")],
        [B(text="📊 Статистика", callback_data="stats"), B(text="👤 Мой доступ", callback_data="access")],
        [B(text="❓ Вопросы", callback_data="faq"), B(text="🎁 Пригласить друга", callback_data="ref")],
        [B(text="💬 Поддержка", url=f"https://t.me/{support}")],
    ])


def back(to: str = "menu", extra: list[list[B]] | None = None) -> KB:
    return KB(inline_keyboard=[*(extra or []), [B(text="← Назад", callback_data=to)]])


def to_plans() -> list[list[B]]:
    return [[B(text="💎 Оформить доступ", callback_data="plans")]]


def risk_kb(settings: Settings) -> KB:
    rows = [[B(text="✅ Понимаю риски и принимаю", callback_data="accept")]]
    if settings.terms_url:
        rows.append([B(text="📄 Условия полностью", url=settings.terms_url)])
    rows.append([B(text="← Назад", callback_data="menu")])
    return KB(inline_keyboard=rows)


def plans_kb(settings: Settings) -> KB:
    rows = [[B(text=p.button, callback_data=f"buy:{p.code}")] for p in settings.plans]
    rows.append([B(text="← Назад", callback_data="menu")])
    return KB(inline_keyboard=rows)


def pay_link_kb(url: str, stars: int) -> KB:
    return KB(inline_keyboard=[[B(text=f"Оформить подписку · {stars} ⭐/мес", url=url)], [B(text="← Тарифы", callback_data="plans")]])


def access_kb(active: bool, auto_renew: bool, has_channel: bool) -> KB:
    rows = []
    if active and has_channel:
        rows.append([B(text="🔐 Ссылка в закрытый канал", callback_data="invite")])
    if active:
        rows.append([B(text="📬 Последние сигналы", callback_data="signals")])
    rows.append([B(text="➕ Продлить" if active else "💎 Оформить доступ", callback_data="plans")])
    if auto_renew:
        rows.append([B(text="Отключить автопродление", callback_data="cancel_renew")])
    rows.append([B(text="← Назад", callback_data="menu")])
    return KB(inline_keyboard=rows)


def renew_kb() -> KB:
    return KB(inline_keyboard=[[B(text="💎 Продлить доступ", callback_data="plans")]])
