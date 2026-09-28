"""Локальная система записи (SQLite) — для тестирования бота без DIKIDI.

Услуги и мастера берутся из catalog.json, рабочие часы — из .env.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from datetime import date, datetime, time, timedelta

from ..config import Settings
from .base import Appointment, BookingBackend, BookingError, Master, Service


def compute_free_slots(
    day: date,
    work_start: time,
    work_end: time,
    step: int,
    duration: int,
    busy: list[tuple[datetime, datetime]],
    tz,
) -> list[datetime]:
    """Начала сеансов длиной duration, не пересекающиеся с занятыми интервалами."""
    cur = datetime.combine(day, work_start, tzinfo=tz)
    end_of_day = datetime.combine(day, work_end, tzinfo=tz)
    length = timedelta(minutes=duration)
    slots: list[datetime] = []
    while cur + length <= end_of_day:
        if all(cur + length <= b_start or cur >= b_end for b_start, b_end in busy):
            slots.append(cur)
        cur += timedelta(minutes=step)
    return slots


class LocalBackend(BookingBackend):
    def __init__(self, settings: Settings):
        self._s = settings
        catalog = json.loads(settings.catalog_path.read_text(encoding="utf-8"))
        self._services = [
            Service(str(x["id"]), x["title"], int(x["duration"]), x.get("price"))
            for x in catalog["services"]
        ]
        self._masters = [Master(str(x["id"]), x["name"]) for x in catalog["masters"]]
        # мастер -> список id услуг (если не указано — делает все)
        self._skills = {str(x["id"]): [str(s) for s in x.get("services", [])] for x in catalog["masters"]}
        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(settings.db_path, check_same_thread=False)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS local_appointments (
                id TEXT PRIMARY KEY, master_id TEXT, service_id TEXT,
                start TEXT, finish TEXT, client_name TEXT, client_phone TEXT,
                comment TEXT, cancelled INTEGER DEFAULT 0)"""
        )
        self._db.commit()
        self._lock = asyncio.Lock()

    async def get_services(self) -> list[Service]:
        return list(self._services)

    async def get_masters(self, service_id: str) -> list[Master]:
        return [m for m in self._masters if not self._skills[m.id] or service_id in self._skills[m.id]]

    def _busy(self, master_id: str, day: date) -> list[tuple[datetime, datetime]]:
        rows = self._db.execute(
            "SELECT start, finish FROM local_appointments "
            "WHERE master_id = ? AND cancelled = 0 AND substr(start, 1, 10) = ?",
            (master_id, day.isoformat()),
        ).fetchall()
        return [(datetime.fromisoformat(a), datetime.fromisoformat(b)) for a, b in rows]

    async def get_free_slots(self, service: Service, master_id: str, day: date) -> list[datetime]:
        return compute_free_slots(
            day, self._s.work_start, self._s.work_end, self._s.slot_step,
            service.duration, self._busy(master_id, day), self._s.tz,
        )

    async def book(self, service, master_id, start, client_name, client_phone, comment=""):
        async with self._lock:
            if start not in await self.get_free_slots(service, master_id, start.date()):
                raise BookingError("Это время уже занято")
            appt_id = uuid.uuid4().hex[:12]
            finish = start + timedelta(minutes=service.duration)
            self._db.execute(
                "INSERT INTO local_appointments (id, master_id, service_id, start, finish, "
                "client_name, client_phone, comment) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (appt_id, master_id, service.id, start.isoformat(), finish.isoformat(),
                 client_name, client_phone, comment),
            )
            self._db.commit()
        master = next(m for m in self._masters if m.id == master_id)
        return Appointment(appt_id, start, service.title, master.name, client_name, client_phone)

    async def cancel(self, appointment_id: str) -> None:
        self._db.execute("UPDATE local_appointments SET cancelled = 1 WHERE id = ?", (appointment_id,))
        self._db.commit()

    async def get_appointments(self, date_from: date, date_to: date) -> list[Appointment]:
        rows = self._db.execute(
            "SELECT id, start, service_id, master_id, client_name, client_phone "
            "FROM local_appointments WHERE cancelled = 0 "
            "AND substr(start, 1, 10) BETWEEN ? AND ? ORDER BY start",
            (date_from.isoformat(), date_to.isoformat()),
        ).fetchall()
        services = {s.id: s.title for s in self._services}
        masters = {m.id: m.name for m in self._masters}
        return [
            Appointment(r[0], datetime.fromisoformat(r[1]), services.get(r[2], r[2]),
                        masters.get(r[3], r[3]), r[4], r[5])
            for r in rows
        ]

    async def close(self) -> None:
        self._db.close()
