from __future__ import annotations

import logging

from aiogram import Bot

from .config import Settings

log = logging.getLogger(__name__)


async def notify_admins(bot: Bot, settings: Settings, text: str) -> None:
    """Сообщение в админский чат (или всем админам, если чат не задан)."""
    targets = [settings.admin_chat_id] if settings.admin_chat_id else list(settings.admin_ids)
    for chat_id in targets:
        try:
            await bot.send_message(chat_id, text)
        except Exception:
            log.exception("Не удалось отправить уведомление в %s", chat_id)
