"""Фоновые задачи: напоминания о продлении и закрытие доступа по окончании срока."""
from __future__ import annotations

import asyncio
import logging
import time

from . import texts
from .config import Settings
from .db import Database
from .formatting import fmt_date
from .keyboards import renew_kb

log = logging.getLogger("regime_bot.scheduler")


async def run_once(bot, db: Database, settings: Settings, broadcaster, now: int | None = None) -> dict[str, int]:
    now = now or int(time.time())
    reminded = expired = 0
    for tg_id, ends, bit in db.due_reminders(now):
        text = (texts.REMIND_1D if bit == 2 else texts.REMIND_3D).format(until=fmt_date(ends))
        await broadcaster.send(tg_id, text, reply_markup=renew_kb())
        db.mark_reminded(tg_id, bit)
        reminded += 1
    for tg_id in db.expired_not_kicked(now):
        if settings.channel_id:
            try:                                                    # «кик» = бан + разбан: человек сможет вернуться по новой ссылке
                await bot.ban_chat_member(settings.channel_id, tg_id, until_date=now + 60)
                await bot.unban_chat_member(settings.channel_id, tg_id, only_if_banned=True)
            except Exception as e:                                 # не в канале / нет прав — не критично
                log.info("kick %s: %s", tg_id, e)
        await broadcaster.send(tg_id, texts.EXPIRED, reply_markup=renew_kb())
        db.mark_kicked(tg_id)
        expired += 1
    return {"reminded": reminded, "expired": expired}


async def loop(bot, db: Database, settings: Settings, broadcaster, every: int = 600) -> None:
    while True:
        try:
            res = await run_once(bot, db, settings, broadcaster)
            if any(res.values()):
                log.info("scheduler: %s", res)
        except Exception:
            log.exception("scheduler tick failed")
        await asyncio.sleep(every)
