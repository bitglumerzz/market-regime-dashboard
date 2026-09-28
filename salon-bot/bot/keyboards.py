from __future__ import annotations

from datetime import date, datetime, timedelta

from aiogram.filters.callback_data import CallbackData
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

from .backends import Master, Service
from .texts import BTN_ASK, BTN_BOOK, BTN_CONTACTS, BTN_MY, BTN_PRICES, duration, price, short_day


class BookCb(CallbackData, prefix="bk"):
    step: str  # svc | mst | day | tm | ok | back | stop
    value: str = ""


class CancelCb(CallbackData, prefix="cn"):
    action: str  # ask | yes | no
    appointment_id: str


def main_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=BTN_BOOK)],
            [KeyboardButton(text=BTN_MY), KeyboardButton(text=BTN_PRICES)],
            [KeyboardButton(text=BTN_CONTACTS), KeyboardButton(text=BTN_ASK)],
        ],
        resize_keyboard=True,
    )


def phone_request() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="📱 Отправить мой номер", request_contact=True)]],
        resize_keyboard=True,
        one_time_keyboard=True,
    )


def _nav(kb: InlineKeyboardBuilder, back: bool = True) -> None:
    buttons = []
    if back:
        buttons.append(InlineKeyboardButton(text="⬅️ Назад", callback_data=BookCb(step="back").pack()))
    buttons.append(InlineKeyboardButton(text="✖️ Отмена", callback_data=BookCb(step="stop").pack()))
    kb.row(*buttons)


def services_kb(services: list[Service]) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    for s in services:
        label = " · ".join(p for p in (s.title, price(s.price), duration(s.duration)) if p)
        kb.button(text=label, callback_data=BookCb(step="svc", value=s.id))
    kb.adjust(1)
    _nav(kb, back=False)
    return kb.as_markup()


def masters_kb(masters: list[Master]) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="✨ Любой свободный мастер", callback_data=BookCb(step="mst", value="any"))
    for m in masters:
        kb.button(text=m.name, callback_data=BookCb(step="mst", value=m.id))
    kb.adjust(1)
    _nav(kb)
    return kb.as_markup()


def days_kb(today: date, days_ahead: int) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    for i in range(days_ahead):
        d = today + timedelta(days=i)
        label = "Сегодня" if i == 0 else "Завтра" if i == 1 else short_day(d)
        kb.button(text=label, callback_data=BookCb(step="day", value=d.isoformat()))
    kb.adjust(3)
    _nav(kb)
    return kb.as_markup()


def times_kb(slots: list[datetime]) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    for dt in slots:
        kb.button(text=dt.strftime("%H:%M"), callback_data=BookCb(step="tm", value=dt.strftime("%H%M")))
    kb.adjust(4)
    _nav(kb)
    return kb.as_markup()


def confirm_kb() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="✅ Подтвердить запись", callback_data=BookCb(step="ok"))
    kb.adjust(1)
    _nav(kb)
    return kb.as_markup()


def cancel_ask_kb(appointment_id: str) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="Отменить запись", callback_data=CancelCb(action="ask", appointment_id=appointment_id))
    return kb.as_markup()


def cancel_confirm_kb(appointment_id: str) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="Да, отменить", callback_data=CancelCb(action="yes", appointment_id=appointment_id))
    kb.button(text="Нет, оставить", callback_data=CancelCb(action="no", appointment_id=appointment_id))
    kb.adjust(2)
    return kb.as_markup()
