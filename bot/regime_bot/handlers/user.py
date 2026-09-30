"""Пользовательская часть: /start, меню, информация, мой доступ, сигналы, статистика."""
from __future__ import annotations

import time

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardMarkup, Message

from .. import keyboards as kb
from .. import texts
from ..access import parse_start_arg
from ..config import Settings
from ..db import Database
from ..formatting import EXAMPLE_SIGNAL, fmt_date, render_history_line, render_signal

router = Router(name="user")


async def show(target: Message | CallbackQuery, text: str, markup: InlineKeyboardMarkup | None = None) -> None:
    """В колбэке — редактируем сообщение с меню, иначе отправляем новое."""
    if isinstance(target, CallbackQuery):
        await target.answer()
        try:
            await target.message.edit_text(text, reply_markup=markup, disable_web_page_preview=True)
            return
        except TelegramBadRequest:
            target = target.message
    await target.answer(text, reply_markup=markup, disable_web_page_preview=True)


def remember(db: Database, user, arg: str | None = None) -> None:
    source, ref = parse_start_arg(arg)
    db.upsert_user(user.id, user.username, user.first_name, source=source, referred_by=ref)


@router.message(CommandStart())
async def start(message: Message, command: CommandObject, db: Database, settings: Settings) -> None:
    remember(db, message.from_user, command.args)
    await message.answer(texts.WELCOME.format(brand=texts.BRAND), reply_markup=kb.main_menu(settings))


@router.message(Command("menu"))
@router.callback_query(F.data == "menu")
async def menu(event: Message | CallbackQuery, db: Database, settings: Settings) -> None:
    remember(db, event.from_user)
    await show(event, texts.WELCOME.format(brand=texts.BRAND), kb.main_menu(settings))


@router.callback_query(F.data == "how")
async def how(cb: CallbackQuery) -> None:
    await show(cb, texts.HOW, kb.back("menu", kb.to_plans()))


@router.callback_query(F.data == "example")
async def example(cb: CallbackQuery) -> None:
    await show(cb, texts.EXAMPLE.format(signal=render_signal(EXAMPLE_SIGNAL)), kb.back("menu", kb.to_plans()))


@router.message(Command("help"))
@router.callback_query(F.data == "faq")
async def faq(event: Message | CallbackQuery) -> None:
    await show(event, texts.FAQ, kb.back("menu", kb.to_plans()))


@router.message(Command("terms"))
async def terms(message: Message, settings: Settings) -> None:
    link = f"\n\nПолная версия: {settings.terms_url}" if settings.terms_url else ""
    await message.answer(texts.TERMS.format(support=settings.support, terms_link=link), disable_web_page_preview=True)


@router.message(Command("paysupport"))
async def paysupport(message: Message, settings: Settings) -> None:
    await message.answer(texts.PAYSUPPORT.format(support=settings.support))


@router.message(Command("stats"))
@router.callback_query(F.data == "stats")
async def stats(event: Message | CallbackQuery, db: Database, settings: Settings) -> None:
    # /stats у админа перехватывает админ-роутер (он подключён раньше)
    s = db.signal_stats()
    if not s["closed"]:
        await show(event, texts.STATS_EMPTY, kb.back("menu", kb.to_plans()))
        return
    last = "\n".join(render_history_line(r) for r in db.recent_signals(8, only_closed=True))
    await show(event, texts.STATS.format(last=last, **s), kb.back("menu", kb.to_plans()))


@router.message(Command("access"))
@router.callback_query(F.data == "access")
async def my_access(event: Message | CallbackQuery, db: Database, settings: Settings) -> None:
    remember(db, event.from_user)
    acc = db.get_access(event.from_user.id)
    if acc is None:
        await show(event, texts.ACCESS_NONE, kb.access_kb(False, False, False))
        return
    plan = settings.plan(acc.plan)
    if acc.active():
        text = texts.ACCESS_ACTIVE.format(plan=plan.title if plan else acc.plan, until=fmt_date(acc.ends_at),
                                          days=acc.days_left(), renew="включено 🔄" if acc.auto_renew else "нет")
    else:
        text = texts.ACCESS_EXPIRED.format(until=fmt_date(acc.ends_at))
    await show(event, text, kb.access_kb(acc.active(), acc.auto_renew and acc.active(), bool(settings.channel_id)))


@router.message(Command("signals"))
@router.callback_query(F.data == "signals")
async def last_signals(event: Message | CallbackQuery, db: Database) -> None:
    acc = db.get_access(event.from_user.id)
    if not acc or not acc.active():
        await show(event, texts.NO_ACCESS_SIGNALS, kb.back("menu", kb.to_plans()))
        return
    rows = db.recent_signals(3)
    if not rows:
        await show(event, "Сигналов пока не было — как только модель найдёт сетап, он придёт сюда.", kb.back("access"))
        return
    if isinstance(event, CallbackQuery):
        await event.answer()
        event = event.message
    await event.answer(texts.LAST_SIGNALS)
    for r in reversed(rows):
        await event.answer(render_signal(r, number=r["id"]))


@router.callback_query(F.data == "invite")
async def invite(cb: CallbackQuery, bot: Bot, db: Database, settings: Settings) -> None:
    acc = db.get_access(cb.from_user.id)
    if not settings.channel_id or not acc or not acc.active():
        await cb.answer("Нет активного доступа", show_alert=True)
        return
    link = await bot.create_chat_invite_link(settings.channel_id, name=f"u{cb.from_user.id}", member_limit=1,
                                             expire_date=int(time.time()) + 86400)
    await cb.answer()
    await cb.message.answer(f"{texts.CHANNEL_INVITE}\n{link.invite_link}")


@router.message(Command("ref"))
@router.callback_query(F.data == "ref")
async def referral(event: Message | CallbackQuery, bot: Bot, db: Database, settings: Settings) -> None:
    remember(db, event.from_user)
    me = await bot.me()
    link = f"https://t.me/{me.username}?start=ref_{event.from_user.id}"
    await show(event, texts.REF.format(days=settings.referral_bonus_days, link=link), kb.back())
