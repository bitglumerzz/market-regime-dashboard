"""Настройки бота из переменных окружения (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _int_list(value: str) -> tuple[int, ...]:
    return tuple(int(x) for x in value.replace(" ", "").split(",") if x)


def _time(value: str) -> time:
    h, m = value.split(":")
    return time(int(h), int(m))


@dataclass(frozen=True)
class Settings:
    bot_token: str
    admin_ids: tuple[int, ...]
    admin_chat_id: int | None
    backend: str  # "dikidi" | "local"
    tz: ZoneInfo
    salon_name: str
    salon_address: str
    salon_phone: str
    dikidi_booking_url: str
    days_ahead: int
    min_lead_minutes: int
    slot_step: int
    work_start: time
    work_end: time
    sync_interval: int
    db_path: Path
    catalog_path: Path


def load_settings() -> Settings:
    admin_chat = os.getenv("ADMIN_CHAT_ID", "").strip()
    admin_ids = _int_list(os.getenv("ADMIN_IDS", ""))
    return Settings(
        bot_token=os.environ["BOT_TOKEN"],
        admin_ids=admin_ids,
        admin_chat_id=int(admin_chat) if admin_chat else (admin_ids[0] if admin_ids else None),
        backend=os.getenv("BACKEND", "dikidi").lower(),
        tz=ZoneInfo(os.getenv("TIMEZONE", "Europe/Moscow")),
        salon_name=os.getenv("SALON_NAME", "Студия наращивания ресниц"),
        salon_address=os.getenv("SALON_ADDRESS", ""),
        salon_phone=os.getenv("SALON_PHONE", ""),
        dikidi_booking_url=os.getenv("DIKIDI_BOOKING_URL", ""),
        days_ahead=int(os.getenv("BOOKING_DAYS_AHEAD", "14")),
        min_lead_minutes=int(os.getenv("MIN_LEAD_MINUTES", "60")),
        slot_step=int(os.getenv("SLOT_STEP_MINUTES", "30")),
        work_start=_time(os.getenv("WORK_START", "10:00")),
        work_end=_time(os.getenv("WORK_END", "20:00")),
        sync_interval=int(os.getenv("SYNC_INTERVAL_SECONDS", "300")),
        db_path=BASE_DIR / os.getenv("DB_PATH", "data/bot.sqlite3"),
        catalog_path=BASE_DIR / os.getenv("CATALOG_PATH", "catalog.json"),
    )
