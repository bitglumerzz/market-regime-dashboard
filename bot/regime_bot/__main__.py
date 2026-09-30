"""Запуск: python -m regime_bot

MODE=polling  — для разработки и простого сервера (бот сам опрашивает Telegram)
MODE=webhook  — для продакшена: Telegram шлёт апдейты на WEBHOOK_BASE/tg/webhook
В обоих режимах на API_PORT поднимается HTTP API для сигналов локальной модели.
"""
from __future__ import annotations

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.types import BotCommand
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web

from . import handlers, scheduler
from .api import make_api
from .broadcast import Broadcaster
from .config import Settings, load_dotenv
from .db import Database
from .signals import SignalService

COMMANDS = [
    BotCommand(command="start", description="Главное меню"),
    BotCommand(command="plans", description="Тарифы и оплата"),
    BotCommand(command="access", description="Мой доступ"),
    BotCommand(command="signals", description="Последние сигналы"),
    BotCommand(command="stats", description="Статистика сигналов"),
    BotCommand(command="ref", description="Пригласить друга"),
    BotCommand(command="help", description="Вопросы и ответы"),
    BotCommand(command="terms", description="Условия использования"),
    BotCommand(command="paysupport", description="Поддержка по оплатам"),
    BotCommand(command="delete_me", description="Удалить мои данные"),
]


def build(settings: Settings) -> tuple[Bot, Dispatcher, web.Application]:
    db = Database(settings.db_path)
    bot = Bot(settings.bot_token, default=DefaultBotProperties(parse_mode="HTML"))
    broadcaster = Broadcaster(bot, db)
    service = SignalService(db, broadcaster, settings.channel_id)
    dp = Dispatcher(db=db, settings=settings, broadcaster=broadcaster, signals=service)
    dp.include_router(handlers.setup())

    if settings.mode == "webhook" and not settings.signal_secret:
        raise RuntimeError("SIGNAL_API_SECRET обязателен")
    app = make_api(service, settings.signal_secret)

    async def on_startup(bot: Bot) -> None:
        await bot.set_my_commands(COMMANDS)
        await bot.set_my_description("ИИ-советник для трейдинга: сигналы с входом, стопом и целями, режим рынка и честная статистика. "
                                     "Не является инвестиционной рекомендацией.")
        await bot.set_my_short_description("ИИ-сигналы для трейдинга · режим рынка · честная статистика")
        if settings.mode == "webhook":
            await bot.set_webhook(f"{settings.webhook_base}/tg/webhook", secret_token=settings.webhook_secret or None,
                                  allowed_updates=dp.resolve_used_update_types(), drop_pending_updates=False)
        asyncio.get_running_loop().create_task(scheduler.loop(bot, db, settings, broadcaster))

    dp.startup.register(on_startup)
    return bot, dp, app


async def run_polling(settings: Settings) -> None:
    bot, dp, app = build(settings)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, settings.api_host, settings.api_port).start()
    logging.info("signal API on %s:%s", settings.api_host, settings.api_port)
    await bot.delete_webhook(drop_pending_updates=False)
    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await runner.cleanup()


def run_webhook(settings: Settings) -> None:
    bot, dp, app = build(settings)
    SimpleRequestHandler(dispatcher=dp, bot=bot, secret_token=settings.webhook_secret or None).register(app, path="/tg/webhook")
    setup_application(app, dp, bot=bot)
    web.run_app(app, host=settings.api_host, port=settings.api_port)


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    settings = Settings.from_env()
    if not settings.signal_secret:
        logging.warning("SIGNAL_API_SECRET пуст — API сигналов будет отклонять все запросы")
    if settings.mode == "webhook":
        run_webhook(settings)
    else:
        asyncio.run(run_polling(settings))


if __name__ == "__main__":
    main()
