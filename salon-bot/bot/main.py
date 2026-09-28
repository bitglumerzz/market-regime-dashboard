"""Точка входа: python -m bot.main"""
from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand

from .backends import create_backend
from .background import background_loop
from .config import load_settings
from .handlers import admin, booking, common
from .storage import Storage


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    backend = create_backend(settings)
    db = Storage(settings.db_path)

    bot = Bot(settings.bot_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    # settings/backend/db попадают в хендлеры по имени аргумента
    dp = Dispatcher(storage=MemoryStorage(), settings=settings, backend=backend, db=db)
    dp.include_routers(admin.router, common.router, booking.router)

    await bot.set_my_commands([
        BotCommand(command="start", description="Главное меню"),
        BotCommand(command="menu", description="Показать меню"),
    ])
    task = asyncio.create_task(background_loop(bot, backend, db, settings))
    try:
        await dp.start_polling(bot)
    finally:
        task.cancel()
        await backend.close()
        db.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
