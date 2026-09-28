import asyncio
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from bot.backends.base import BookingError
from bot.backends.local import LocalBackend, compute_free_slots
from bot.config import BASE_DIR, Settings
from bot.handlers.booking import normalize_phone
from bot.storage import Storage

TZ = ZoneInfo("Europe/Moscow")


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        bot_token="x", admin_ids=(1,), admin_chat_id=1, backend="local", tz=TZ,
        salon_name="Test", salon_address="", salon_phone="", dikidi_booking_url="",
        days_ahead=14, min_lead_minutes=60, slot_step=30,
        work_start=time(10), work_end=time(14), sync_interval=300,
        db_path=tmp_path / "db.sqlite3", catalog_path=BASE_DIR / "catalog.json",
    )


def test_free_slots_skip_busy():
    day = date(2026, 10, 1)
    busy = [(datetime(2026, 10, 1, 11, 0, tzinfo=TZ), datetime(2026, 10, 1, 12, 0, tzinfo=TZ))]
    slots = compute_free_slots(day, time(10), time(14), 30, 60, busy, TZ)
    assert [s.strftime("%H:%M") for s in slots] == ["10:00", "12:00", "12:30", "13:00"]


@pytest.mark.parametrize("raw,expected", [
    ("8 (999) 123-45-67", "+79991234567"),
    ("+7 999 123 45 67", "+79991234567"),
    ("9991234567", "+79991234567"),
    ("+375291234567", "+375291234567"),
    ("123", None),
])
def test_normalize_phone(raw, expected):
    assert normalize_phone(raw) == expected


def test_local_backend_booking_and_cancel(tmp_path):
    async def scenario():
        backend = LocalBackend(make_settings(tmp_path))
        services = await backend.get_services()
        service = next(s for s in services if s.id == "classic")  # 120 минут
        day = date(2026, 10, 1)
        slots = await backend.get_free_slots(service, "1", day)
        assert slots[0] == datetime(2026, 10, 1, 10, 0, tzinfo=TZ)

        appt = await backend.book(service, "1", slots[0], "Ира", "+79990000000")
        with pytest.raises(BookingError):
            await backend.book(service, "1", slots[0], "Оля", "+79990000001")
        left = await backend.get_free_slots(service, "1", day)
        assert [s.strftime("%H:%M") for s in left] == ["12:00"]
        assert [a.id for a in await backend.get_appointments(day, day)] == [appt.id]

        await backend.cancel(appt.id)
        assert await backend.get_appointments(day, day) == []
        await backend.close()

    asyncio.run(scenario())


def test_master_skills_filter(tmp_path):
    backend = LocalBackend(make_settings(tmp_path))
    names = [m.name for m in asyncio.run(backend.get_masters("hollywood"))]
    assert names == ["Анна"]


def test_storage_upcoming_and_reschedule(tmp_path):
    db = Storage(tmp_path / "s.sqlite3")
    now = datetime(2026, 10, 1, 9, 0, tzinfo=TZ)
    db.add_booking("a1", 42, now + timedelta(hours=3), "Классика", "Анна")
    db.add_booking("a2", 42, now - timedelta(hours=3), "Классика", "Анна")
    assert [b.appointment_id for b in db.upcoming_for_user(42, now)] == ["a1"]
    db.mark_reminded("a1", "24h")
    db.reschedule("a1", now + timedelta(days=1))
    row = db.get("a1")
    assert row.start == now + timedelta(days=1) and not row.reminded_24h
    db.mark_cancelled("a1")
    assert db.upcoming_for_user(42, now) == []
