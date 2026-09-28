"""Команды администратора: расписание из DIKIDI и ответы клиентам."""
from __future__ import annotations

import html
import re
from datetime import date, datetime, timedelta

from aiogram import Bot, F, Router
from aiogram.filters import Command, Filter
from aiogram.types import Message

from ..backends import BookingBackend
from ..config import Settings
from ..texts import human_dt


class IsAdmin(Filter):
    async def __call__(self, message: Message, settings: Settings) -> bool:
        return message.from_user is not None and message.from_user.id in settings.admin_ids


router = Router(name="admin")
# Роутер срабатывает только для админов, остальные сообщения идут дальше к клиентским хендлерам
router.message.filter(IsAdmin())


async def _send_schedule(message: Message, backend: BookingBackend, date_from: date, date_to: date, title: str):
    appts = await backend.get_appointments(date_from, date_to)
    if not appts:
        await message.answer(f"{title}: записей нет")
        return
    lines = [f"<b>{title}</b> — {len(appts)} зап."]
    current_day = None
    for a in appts:
        if a.start.date() != current_day:
            current_day = a.start.date()
            lines.append(f"\n<b>{human_dt(a.start).split(' в ')[0]}</b>")
        client = ", ".join(p for p in (a.client_name, a.client_phone) if p)
        lines.append(
            f"{a.start:%H:%M} {html.escape(a.service_title)} · {html.escape(a.master_name)}"
            + (f" — {html.escape(client)}" if client else "")
        )
    await message.answer("\n".join(lines))


@router.message(Command("today"))
async def today(message: Message, backend: BookingBackend, settings: Settings):
    d = datetime.now(settings.tz).date()
    await _send_schedule(message, backend, d, d, "Сегодня")


@router.message(Command("tomorrow"))
async def tomorrow(message: Message, backend: BookingBackend, settings: Settings):
    d = datetime.now(settings.tz).date() + timedelta(days=1)
    await _send_schedule(message, backend, d, d, "Завтра")


@router.message(Command("week"))
async def week(message: Message, backend: BookingBackend, settings: Settings):
    d = datetime.now(settings.tz).date()
    await _send_schedule(message, backend, d, d + timedelta(days=6), "Ближайшие 7 дней")


@router.message(Command("admin"))
async def admin_help(message: Message):
    await message.answer(
        "Команды администратора:\n"
        "/today — записи на сегодня\n/tomorrow — на завтра\n/week — на 7 дней\n\n"
        "Чтобы ответить клиенту, ответьте (reply) на пересланный ботом вопрос."
    )


@router.message(F.reply_to_message.text.regexp(r"#u(\d+)", mode="search").as_("match"), F.text)
async def reply_to_client(message: Message, match: re.Match, bot: Bot):
    await bot.send_message(int(match.group(1)), f"💬 Ответ администратора:\n\n{html.escape(message.text)}")
    await message.reply("Ответ отправлен ✅")
