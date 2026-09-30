"""Рассылка с ограничением скорости Telegram (~30 сообщений/с всего) и обработкой блокировок."""
from __future__ import annotations

import asyncio
import logging
import time

from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError, TelegramRetryAfter
from aiogram.types import ReplyParameters

from .db import Database

log = logging.getLogger("regime_bot.broadcast")


class Broadcaster:
    def __init__(self, bot, db: Database, per_second: float = 25.0):
        self.bot, self.db = bot, db
        self.interval = 1.0 / per_second
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def _tick(self) -> None:
        async with self._lock:
            wait = self._last + self.interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()

    async def send(self, chat_id: int, text: str, reply_to: int | None = None, **kw) -> int | None:
        """Отправить сообщение. Возвращает message_id или None (заблокировал бота / ошибка)."""
        if reply_to:
            kw["reply_parameters"] = ReplyParameters(message_id=reply_to, allow_sending_without_reply=True)
        for attempt in range(3):
            await self._tick()
            try:
                msg = await self.bot.send_message(chat_id, text, **kw)
                return msg.message_id
            except TelegramRetryAfter as e:
                await asyncio.sleep(e.retry_after + 0.5)
            except TelegramForbiddenError:
                if chat_id > 0:
                    self.db.set_blocked(chat_id, True)
                return None
            except TelegramBadRequest as e:
                log.warning("send to %s failed: %s", chat_id, e)
                return None
        return None

    async def copy(self, chat_id: int, from_chat: int, message_id: int) -> bool:
        await self._tick()
        try:
            await self.bot.copy_message(chat_id, from_chat, message_id)
            return True
        except TelegramForbiddenError:
            self.db.set_blocked(chat_id, True)
        except (TelegramBadRequest, TelegramRetryAfter) as e:
            log.warning("copy to %s failed: %s", chat_id, e)
        return False
