"""Локальные данные бота: связь «клиент Telegram ↔ запись в DIKIDI» и напоминания.

Сами записи живут в DIKIDI; здесь хранится только то, что нужно боту,
чтобы показывать клиенту его записи и присылать напоминания.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class BookingRow:
    appointment_id: str
    user_id: int
    start: datetime
    service_title: str
    master_name: str
    reminded_24h: bool
    reminded_2h: bool


class Storage:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS clients (
                user_id INTEGER PRIMARY KEY, name TEXT, phone TEXT);
            CREATE TABLE IF NOT EXISTS bookings (
                appointment_id TEXT PRIMARY KEY, user_id INTEGER, start TEXT,
                service_title TEXT, master_name TEXT,
                reminded_24h INTEGER DEFAULT 0, reminded_2h INTEGER DEFAULT 0,
                cancelled INTEGER DEFAULT 0);
            """
        )
        self._db.commit()

    # --- клиенты ---
    def get_client(self, user_id: int) -> tuple[str, str] | None:
        row = self._db.execute("SELECT name, phone FROM clients WHERE user_id = ?", (user_id,)).fetchone()
        return (row[0], row[1]) if row else None

    def save_client(self, user_id: int, name: str, phone: str) -> None:
        self._db.execute(
            "INSERT INTO clients (user_id, name, phone) VALUES (?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET name = excluded.name, phone = excluded.phone",
            (user_id, name, phone),
        )
        self._db.commit()

    # --- записи ---
    def add_booking(self, appointment_id: str, user_id: int, start: datetime,
                    service_title: str, master_name: str) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO bookings (appointment_id, user_id, start, service_title, master_name) "
            "VALUES (?, ?, ?, ?, ?)",
            (appointment_id, user_id, start.isoformat(), service_title, master_name),
        )
        self._db.commit()

    def _rows(self, where: str, params: tuple) -> list[BookingRow]:
        rows = self._db.execute(
            "SELECT appointment_id, user_id, start, service_title, master_name, reminded_24h, reminded_2h "
            f"FROM bookings WHERE cancelled = 0 AND {where} ORDER BY start",
            params,
        ).fetchall()
        return [BookingRow(r[0], r[1], datetime.fromisoformat(r[2]), r[3], r[4], bool(r[5]), bool(r[6]))
                for r in rows]

    def upcoming_for_user(self, user_id: int, now: datetime) -> list[BookingRow]:
        return [b for b in self._rows("user_id = ?", (user_id,)) if b.start > now]

    def all_upcoming(self, now: datetime) -> list[BookingRow]:
        return [b for b in self._rows("1 = 1", ()) if b.start > now]

    def get(self, appointment_id: str) -> BookingRow | None:
        rows = self._rows("appointment_id = ?", (appointment_id,))
        return rows[0] if rows else None

    def mark_cancelled(self, appointment_id: str) -> None:
        self._db.execute("UPDATE bookings SET cancelled = 1 WHERE appointment_id = ?", (appointment_id,))
        self._db.commit()

    def reschedule(self, appointment_id: str, start: datetime) -> None:
        self._db.execute(
            "UPDATE bookings SET start = ?, reminded_24h = 0, reminded_2h = 0 WHERE appointment_id = ?",
            (start.isoformat(), appointment_id),
        )
        self._db.commit()

    def mark_reminded(self, appointment_id: str, kind: str) -> None:
        column = {"24h": "reminded_24h", "2h": "reminded_2h"}[kind]
        self._db.execute(f"UPDATE bookings SET {column} = 1 WHERE appointment_id = ?", (appointment_id,))
        self._db.commit()

    def close(self) -> None:
        self._db.close()
