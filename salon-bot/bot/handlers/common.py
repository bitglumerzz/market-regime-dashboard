"""Меню клиента: приветствие, мои записи, отмена, цены, контакты, вопрос администратору."""
from __future__ import annotations

import html
from datetime import datetime

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, Message

from .. import keyboards as kb
from ..backends import BookingBackend, BookingError
from ..config import Settings
from ..notify import notify_admins
from ..storage import Storage
from ..texts import BTN_ASK, BTN_CONTACTS, BTN_MY, BTN_PRICES, duration, human_dt, price

router = Router(name="common")


class AskAdmin(StatesGroup):
    text = State()


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext, settings: Settings):
    await state.clear()
    await message.answer(
        f"Здравствуйте, {html.escape(message.from_user.first_name or '')}! 👋\n"
        f"Я цифровой администратор «{html.escape(settings.salon_name)}».\n\n"
        "Помогу записаться на наращивание ресниц, покажу свободное время, "
        "напомню о визите и отвечу на вопросы. Выберите действие в меню 👇",
        reply_markup=kb.main_menu(),
    )


@router.message(Command("menu"))
async def cmd_menu(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("Главное меню 👇", reply_markup=kb.main_menu())


@router.message(F.text == BTN_PRICES)
async def prices(message: Message, backend: BookingBackend):
    services = await backend.get_services()
    lines = ["<b>Услуги и цены</b>", ""]
    for s in services:
        tail = " · ".join(p for p in (price(s.price), duration(s.duration)) if p)
        lines.append(f"• {html.escape(s.title)}" + (f" — {tail}" if tail else ""))
    await message.answer("\n".join(lines))


@router.message(F.text == BTN_CONTACTS)
async def contacts(message: Message, settings: Settings):
    lines = [f"<b>{html.escape(settings.salon_name)}</b>"]
    if settings.salon_address:
        lines.append(f"📍 {html.escape(settings.salon_address)}")
    if settings.salon_phone:
        lines.append(f"📞 {html.escape(settings.salon_phone)}")
    if settings.dikidi_booking_url:
        lines.append(f"🌐 Онлайн-запись DIKIDI: {settings.dikidi_booking_url}")
    await message.answer("\n".join(lines))


# --- мои записи и отмена ---

@router.message(F.text == BTN_MY)
async def my_bookings(message: Message, db: Storage, settings: Settings):
    items = db.upcoming_for_user(message.from_user.id, datetime.now(settings.tz))
    if not items:
        await message.answer("У вас пока нет предстоящих записей. Нажмите «💅 Записаться» 🙂")
        return
    for b in items:
        await message.answer(
            f"🗓 {human_dt(b.start)}\n💅 {html.escape(b.service_title)}\n👩‍🎨 {html.escape(b.master_name)}",
            reply_markup=kb.cancel_ask_kb(b.appointment_id),
        )


@router.callback_query(kb.CancelCb.filter(F.action == "ask"))
async def cancel_ask(cb: CallbackQuery, callback_data: kb.CancelCb):
    await cb.message.edit_reply_markup(reply_markup=kb.cancel_confirm_kb(callback_data.appointment_id))
    await cb.answer("Точно отменить?")


@router.callback_query(kb.CancelCb.filter(F.action == "no"))
async def cancel_no(cb: CallbackQuery, callback_data: kb.CancelCb):
    await cb.message.edit_reply_markup(reply_markup=kb.cancel_ask_kb(callback_data.appointment_id))
    await cb.answer("Запись сохранена 👍")


@router.callback_query(kb.CancelCb.filter(F.action == "yes"))
async def cancel_yes(cb: CallbackQuery, callback_data: kb.CancelCb, backend: BookingBackend,
                     db: Storage, settings: Settings, bot: Bot):
    booking = db.get(callback_data.appointment_id)
    if not booking or booking.user_id != cb.from_user.id:
        await cb.answer("Запись не найдена", show_alert=True)
        return
    try:
        await backend.cancel(booking.appointment_id)
    except BookingError:
        await cb.answer("Не получилось отменить, напишите администратору", show_alert=True)
        return
    db.mark_cancelled(booking.appointment_id)
    await cb.message.edit_text(f"❌ Запись на {human_dt(booking.start)} отменена.")
    await cb.answer()
    client = db.get_client(cb.from_user.id)
    who = f"{client[0]}, {client[1]}" if client else str(cb.from_user.id)
    await notify_admins(
        bot, settings,
        f"❌ <b>Клиент отменил запись</b>\n{html.escape(booking.service_title)} · "
        f"{html.escape(booking.master_name)}\n{human_dt(booking.start)}\nКлиент: {html.escape(who)}",
    )


# --- вопрос администратору ---

@router.message(F.text == BTN_ASK)
async def ask_admin(message: Message, state: FSMContext):
    await state.set_state(AskAdmin.text)
    await message.answer("Напишите ваш вопрос одним сообщением — администратор ответит здесь же.")


@router.message(AskAdmin.text, F.text)
async def ask_admin_text(message: Message, state: FSMContext, settings: Settings, bot: Bot):
    await state.clear()
    user = message.from_user
    await notify_admins(
        bot, settings,
        f"✉️ Вопрос от {html.escape(user.full_name)} "
        f"(@{user.username or '—'}) #u{user.id}\n\n{html.escape(message.text)}\n\n"
        "<i>Ответьте на это сообщение (reply), чтобы клиент получил ответ.</i>",
    )
    await message.answer("Спасибо! Передала вопрос администратору 💛", reply_markup=kb.main_menu())
