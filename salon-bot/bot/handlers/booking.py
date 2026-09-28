"""Запись клиента: услуга → мастер → день → время → имя/телефон → подтверждение."""
from __future__ import annotations

import html
import re
from datetime import date, datetime, timedelta

from aiogram import Bot, F, Router
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, Message

from .. import keyboards as kb
from ..backends import BookingBackend, BookingError, Service
from ..config import Settings
from ..notify import notify_admins
from ..storage import Storage
from ..texts import BTN_BOOK, duration, human_dt, price

router = Router(name="booking")


class Booking(StatesGroup):
    service = State()
    master = State()
    day = State()
    time = State()
    name = State()
    phone = State()
    confirm = State()


def normalize_phone(raw: str) -> str | None:
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 11 and digits[0] in "78":
        return "+7" + digits[1:]
    if len(digits) == 10 and digits[0] == "9":
        return "+7" + digits
    if 10 <= len(digits) <= 15 and raw.strip().startswith("+"):
        return "+" + digits
    return None


def _service(data: dict) -> Service:
    return Service(**data["service"])


# --- шаги (показ экранов) ------------------------------------------------------

async def show_services(target: Message | CallbackQuery, state: FSMContext, backend: BookingBackend):
    services = await backend.get_services()
    await state.set_state(Booking.service)
    await state.update_data(services={s.id: s.__dict__ for s in services})
    text = "Выберите услугу 👇"
    if isinstance(target, CallbackQuery):
        await target.message.edit_text(text, reply_markup=kb.services_kb(services))
    else:
        await target.answer(text, reply_markup=kb.services_kb(services))


async def show_masters(cb: CallbackQuery, state: FSMContext, backend: BookingBackend, settings: Settings):
    data = await state.get_data()
    masters = await backend.get_masters(data["service"]["id"])
    if not masters:
        await cb.answer("Нет мастеров для этой услуги", show_alert=True)
        return
    await state.update_data(masters={m.id: m.name for m in masters})
    if len(masters) == 1:  # один мастер — шаг выбора не нужен
        await state.update_data(master_id=masters[0].id, single_master=True)
        await show_days(cb, state, settings)
        return
    await state.update_data(single_master=False)
    await state.set_state(Booking.master)
    await cb.message.edit_text("Выберите мастера:", reply_markup=kb.masters_kb(masters))


async def show_days(cb: CallbackQuery, state: FSMContext, settings: Settings):
    data = await state.get_data()
    await state.set_state(Booking.day)
    s = data["service"]
    master = "любой свободный" if data["master_id"] == "any" else data["masters"][data["master_id"]]
    today = datetime.now(settings.tz).date()
    await cb.message.edit_text(
        f"<b>{html.escape(s['title'])}</b>\nМастер: {html.escape(master)}\n\nВыберите день:",
        reply_markup=kb.days_kb(today, settings.days_ahead),
    )


async def collect_slots(
    backend: BookingBackend, settings: Settings, service: Service, master_ids: list[str], day: date
) -> dict[str, str]:
    """Свободное время {"HHMM": master_id}; при «любом мастере» объединяет окна всех мастеров."""
    earliest = datetime.now(settings.tz) + timedelta(minutes=settings.min_lead_minutes)
    slots: dict[str, str] = {}
    for master_id in master_ids:
        for dt in await backend.get_free_slots(service, master_id, day):
            if dt >= earliest:
                slots.setdefault(dt.strftime("%H%M"), master_id)
    return dict(sorted(slots.items()))


async def show_times(cb: CallbackQuery, state: FSMContext, backend: BookingBackend, settings: Settings, day: date):
    data = await state.get_data()
    service = _service(data)
    master_ids = list(data["masters"]) if data["master_id"] == "any" else [data["master_id"]]
    try:
        slots = await collect_slots(backend, settings, service, master_ids, day)
    except Exception:
        await cb.answer("Не получилось загрузить расписание, попробуйте ещё раз", show_alert=True)
        return
    if not slots:
        await cb.answer("На этот день свободного времени нет — выберите другой 🙏", show_alert=True)
        return
    await state.update_data(day=day.isoformat(), slots=slots)
    await state.set_state(Booking.time)
    times = [datetime.combine(day, datetime.strptime(k, "%H%M").time()) for k in slots]
    await cb.message.edit_text(f"Свободное время на {day:%d.%m}:", reply_markup=kb.times_kb(times))


async def ask_confirm(target: Message, state: FSMContext):
    data = await state.get_data()
    s = data["service"]
    start = datetime.fromisoformat(data["start"])
    lines = [
        "Проверьте, пожалуйста, запись:",
        "",
        f"💅 <b>{html.escape(s['title'])}</b>",
        f"👩‍🎨 Мастер: {html.escape(data['masters'][data['chosen_master']])}",
        f"🗓 {human_dt(start)}",
        f"⏱ {duration(s['duration'])}" + (f" · {price(s['price'])}" if s.get("price") else ""),
        f"🙋‍♀️ {html.escape(data['name'])}, {data['phone']}",
    ]
    await state.set_state(Booking.confirm)
    await target.answer("\n".join(lines), reply_markup=kb.confirm_kb())


# --- хендлеры ----------------------------------------------------------------------

@router.message(F.text == BTN_BOOK)
async def start_booking(message: Message, state: FSMContext, backend: BookingBackend):
    await state.clear()
    try:
        await show_services(message, state, backend)
    except Exception:
        await message.answer("Не удалось получить список услуг. Попробуйте чуть позже или напишите администратору.")
        raise


@router.callback_query(kb.BookCb.filter(F.step == "stop"))
async def stop(cb: CallbackQuery, state: FSMContext):
    await state.clear()
    await cb.message.edit_text("Запись отменена. Возвращайтесь, когда будет удобно 💛")


@router.callback_query(kb.BookCb.filter(F.step == "svc"), Booking.service)
async def pick_service(cb: CallbackQuery, callback_data: kb.BookCb, state: FSMContext,
                       backend: BookingBackend, settings: Settings):
    data = await state.get_data()
    await state.update_data(service=data["services"][callback_data.value])
    await show_masters(cb, state, backend, settings)
    await cb.answer()


@router.callback_query(kb.BookCb.filter(F.step == "mst"), Booking.master)
async def pick_master(cb: CallbackQuery, callback_data: kb.BookCb, state: FSMContext, settings: Settings):
    await state.update_data(master_id=callback_data.value)
    await show_days(cb, state, settings)
    await cb.answer()


@router.callback_query(kb.BookCb.filter(F.step == "day"), Booking.day)
async def pick_day(cb: CallbackQuery, callback_data: kb.BookCb, state: FSMContext,
                   backend: BookingBackend, settings: Settings):
    await show_times(cb, state, backend, settings, date.fromisoformat(callback_data.value))
    await cb.answer()


@router.callback_query(kb.BookCb.filter(F.step == "tm"), Booking.time)
async def pick_time(cb: CallbackQuery, callback_data: kb.BookCb, state: FSMContext,
                    settings: Settings, db: Storage):
    data = await state.get_data()
    day = date.fromisoformat(data["day"])
    start = datetime.combine(day, datetime.strptime(callback_data.value, "%H%M").time(), tzinfo=settings.tz)
    await state.update_data(start=start.isoformat(), chosen_master=data["slots"][callback_data.value])
    await cb.message.edit_text(f"Вы выбрали {human_dt(start)} ✨")
    await cb.answer()

    known = db.get_client(cb.from_user.id)
    if known:
        await state.update_data(name=known[0], phone=known[1])
        await ask_confirm(cb.message, state)
    else:
        await state.set_state(Booking.name)
        await cb.message.answer("Как к вам обращаться? Напишите имя:")


@router.message(Booking.name, F.text)
async def enter_name(message: Message, state: FSMContext):
    name = message.text.strip()[:60]
    if len(name) < 2:
        await message.answer("Напишите, пожалуйста, имя 🙂")
        return
    await state.update_data(name=name)
    await state.set_state(Booking.phone)
    await message.answer(
        "Оставьте номер телефона — нажмите кнопку ниже или напишите его в формате +7…",
        reply_markup=kb.phone_request(),
    )


@router.message(Booking.phone, F.contact | F.text)
async def enter_phone(message: Message, state: FSMContext):
    raw = message.contact.phone_number if message.contact else message.text
    phone = normalize_phone(raw or "")
    if not phone:
        await message.answer("Не похоже на номер телефона. Пример: +7 999 123-45-67")
        return
    await state.update_data(phone=phone)
    await message.answer("Спасибо!", reply_markup=kb.main_menu())
    await ask_confirm(message, state)


@router.callback_query(kb.BookCb.filter(F.step == "ok"), Booking.confirm)
async def confirm(cb: CallbackQuery, state: FSMContext, backend: BookingBackend,
                  settings: Settings, db: Storage, bot: Bot):
    data = await state.get_data()
    service = _service(data)
    start = datetime.fromisoformat(data["start"])
    master_id = data["chosen_master"]
    await cb.answer("Записываю…")
    try:
        appt = await backend.book(
            service, master_id, start, data["name"], data["phone"],
            comment=f"Запись через Telegram-бота (@{cb.from_user.username or cb.from_user.id})",
        )
    except BookingError as exc:
        await state.set_state(Booking.day)
        await cb.message.edit_text(
            f"😔 Не получилось записать: {html.escape(str(exc))}\nВыберите другое время:",
            reply_markup=kb.days_kb(datetime.now(settings.tz).date(), settings.days_ahead),
        )
        return

    master_name = data["masters"][master_id]
    db.save_client(cb.from_user.id, data["name"], data["phone"])
    db.add_booking(appt.id, cb.from_user.id, start, service.title, master_name)
    await state.clear()
    await cb.message.edit_text(
        "🎉 Вы записаны!\n\n"
        f"💅 {html.escape(service.title)}\n👩‍🎨 {html.escape(master_name)}\n🗓 {human_dt(start)}\n\n"
        + (f"📍 {html.escape(settings.salon_address)}\n" if settings.salon_address else "")
        + "Напомню о визите за день и за 2 часа. Отменить запись можно в «📅 Мои записи»."
    )
    await notify_admins(
        bot, settings,
        "🆕 <b>Новая запись из Telegram</b>\n"
        f"{html.escape(service.title)} · {html.escape(master_name)}\n{human_dt(start)}\n"
        f"Клиент: {html.escape(data['name'])}, {data['phone']}",
    )


@router.callback_query(kb.BookCb.filter(F.step == "back"))
async def back(cb: CallbackQuery, state: FSMContext, backend: BookingBackend, settings: Settings):
    current = await state.get_state()
    data = await state.get_data()
    if current == Booking.master.state:
        await show_services(cb, state, backend)
    elif current == Booking.day.state:
        if data.get("single_master"):
            await show_services(cb, state, backend)
        else:
            await show_masters(cb, state, backend, settings)
    elif current in (Booking.time.state, Booking.confirm.state):
        await show_days(cb, state, settings)
    await cb.answer()
