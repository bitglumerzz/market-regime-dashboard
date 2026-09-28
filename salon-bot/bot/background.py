"""Фоновые задачи: напоминания клиентам и синхронизация с DIKIDI.

Синхронизация: если администратор отменил или перенёс запись в DIKIDI,
бот обновит свою базу и сообщит об этом клиенту в Telegram.
"""
from __future__ import annotations

import asyncio
import html
import logging
from datetime import datetime, timedelta

from aiogram import Bot

from .backends import BookingBackend
from .config import Settings
from .storage import Storage
from .texts import human_dt

log = logging.getLogger(__name__)


async def send_reminders(bot: Bot, storage: Storage, settings: Settings) -> None:
    now = datetime.now(settings.tz)
    for b in storage.all_upcoming(now):
        left = b.start - now
        kind = None
        if left <= timedelta(hours=2) and not b.reminded_2h:
            kind, text = "2h", f"⏰ Ждём вас сегодня в {b.start:%H:%M}!"
        elif timedelta(hours=2) < left <= timedelta(hours=24) and not b.reminded_24h:
            kind, text = "24h", f"🔔 Напоминаем: вы записаны на {human_dt(b.start)}."
        if not kind:
            continue
        text += (f"\n💅 {html.escape(b.service_title)} · {html.escape(b.master_name)}\n"
                 "Если планы изменились — отмените запись в «📅 Мои записи».")
        try:
            await bot.send_message(b.user_id, text)
        except Exception:
            log.exception("Не удалось отправить напоминание %s", b.appointment_id)
        storage.mark_reminded(b.appointment_id, kind)


async def sync_with_backend(bot: Bot, backend: BookingBackend, storage: Storage, settings: Settings) -> None:
    now = datetime.now(settings.tz)
    ours = storage.all_upcoming(now)
    if not ours:
        return
    last_day = max(b.start for b in ours).date()
    # если запрос упадёт — исключение пробросится и ничего не будет помечено отменённым
    remote = {a.id: a for a in await backend.get_appointments(now.date(), last_day + timedelta(days=1))}
    for b in ours:
        appt = remote.get(b.appointment_id)
        if appt is None:
            storage.mark_cancelled(b.appointment_id)
            message = (f"ℹ️ Ваша запись на {human_dt(b.start)} ({html.escape(b.service_title)}) была отменена "
                       "салоном. Чтобы подобрать другое время, нажмите «💅 Записаться» или напишите администратору.")
        elif appt.start != b.start:
            storage.reschedule(b.appointment_id, appt.start)
            message = (f"ℹ️ Ваша запись ({html.escape(b.service_title)}) перенесена: "
                       f"{human_dt(b.start)} → <b>{human_dt(appt.start)}</b>.")
        else:
            continue
        try:
            await bot.send_message(b.user_id, message)
        except Exception:
            log.exception("Не удалось уведомить клиента %s", b.user_id)


async def background_loop(bot: Bot, backend: BookingBackend, storage: Storage, settings: Settings) -> None:
    last_sync = datetime.min.replace(tzinfo=settings.tz)
    while True:
        try:
            now = datetime.now(settings.tz)
            if (now - last_sync).total_seconds() >= settings.sync_interval:
                await sync_with_backend(bot, backend, storage, settings)
                last_sync = now
            await send_reminders(bot, storage, settings)
        except Exception:
            log.exception("Ошибка фоновой задачи")
        await asyncio.sleep(60)
