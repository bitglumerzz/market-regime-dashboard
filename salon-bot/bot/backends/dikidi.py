"""Синхронизация с DIKIDI Business.

У DIKIDI нет публичного открытого API для записи, поэтому используется
неофициальный SDK `dikidi-api-client` (https://pypi.org/project/dikidi-api-client/),
который работает от имени аккаунта владельца/администратора салона
(DIKIDI_PHONE / DIKIDI_PASSWORD / DIKIDI_COMPANY в .env).

Все записи создаются прямо в журнале DIKIDI, поэтому администратор видит их
в приложении DIKIDI Business, а бот — видит записи, созданные в DIKIDI вручную
или через онлайн-запись DIKIDI (свободные окна всегда берутся из DIKIDI).

ВАЖНО: SDK неофициальный. Перед запуском выполните `python -m bot.check_dikidi` —
скрипт покажет, что реально возвращает ваш аккаунт. Если названия полей
отличаются, поправьте их только в этом файле (функции _attr ниже уже пробуют
несколько распространённых вариантов).
"""
from __future__ import annotations

import asyncio
import threading
from datetime import date, datetime
from typing import Any

from ..config import Settings
from .base import Appointment, BookingBackend, BookingError, Master, Service


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    """Достаёт первое непустое поле из объекта SDK или dict."""
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if value not in (None, ""):
            return value
    return default


class DikidiBackend(BookingBackend):
    def __init__(self, settings: Settings):
        try:
            from dikidi import DikidiAPI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Установите SDK: pip install dikidi-api-client") from exc
        self._s = settings
        # DikidiAPI сам читает DIKIDI_PHONE / DIKIDI_PASSWORD / DIKIDI_COMPANY из окружения
        self._api = DikidiAPI()
        self._api.__enter__()
        # SDK синхронный и хранит одну сессию — сериализуем обращения к нему
        self._lock = threading.Lock()
        self._services: dict[str, Service] = {}
        self._masters: dict[str, str] = {}

    async def _call(self, fn, *args, **kwargs):
        def run():
            with self._lock:
                return fn(*args, **kwargs)

        return await asyncio.to_thread(run)

    # --- каталог -------------------------------------------------------------

    async def get_services(self) -> list[Service]:
        raw = await self._call(self._api.catalog.get_services)
        services = [
            Service(
                id=str(_attr(s, "id", "service_id")),
                title=str(_attr(s, "title", "name", default="Услуга")),
                duration=int(_attr(s, "duration", "time", "duration_minutes", default=60)),
                price=_attr(s, "price", "cost", "price_min"),
            )
            for s in raw
        ]
        self._services = {s.id: s for s in services}
        return services

    async def _all_masters(self) -> list[Master]:
        raw = await self._call(self._api.catalog.get_masters)
        masters = [Master(str(_attr(m, "id", "master_id")), str(_attr(m, "name", "username", default="Мастер")))
                   for m in raw]
        self._masters = {m.id: m.name for m in masters}
        return masters

    async def get_masters(self, service_id: str) -> list[Master]:
        masters = await self._all_masters()
        today = datetime.now(self._s.tz).date().isoformat()
        try:
            raw = await self._call(self._api.catalog.get_available_masters_for_service, service_id, today)
            allowed = {str(_attr(m, "id", "master_id", default=m)) for m in raw}
        except Exception:
            allowed = set()
        filtered = [m for m in masters if m.id in allowed]
        return filtered or masters

    # --- окна и записи --------------------------------------------------------

    async def get_free_slots(self, service: Service, master_id: str, day: date) -> list[datetime]:
        times = await self._call(
            self._api.appointments.get_free_slots,
            master_id=master_id,
            date=day.isoformat(),
            duration_minutes=service.duration,
            step_minutes=self._s.slot_step,
        )
        result = []
        for t in times:  # ['10:00', '10:30', ...]
            h, m = str(t)[:5].split(":")
            result.append(datetime(day.year, day.month, day.day, int(h), int(m), tzinfo=self._s.tz))
        return result

    async def book(self, service, master_id, start, client_name, client_phone, comment=""):
        from dikidi import BookingRequest

        # Перепроверяем окно прямо перед записью: его могли занять в DIKIDI
        if start not in await self.get_free_slots(service, master_id, start.date()):
            raise BookingError("Это время только что заняли")
        request = BookingRequest(
            master_id=master_id,
            service_id=service.id,
            date=start.date().isoformat(),
            time=start.strftime("%H:%M"),
            client_name=client_name,
            client_phone=client_phone,
            comment=comment,
        )
        try:
            record = await self._call(self._api.appointments.book, request)
        except Exception as exc:
            raise BookingError(f"DIKIDI отклонил запись: {exc}") from exc
        appt_id = _attr(record, "id", "appointment_id", "record_id", default=record)
        return Appointment(
            id=str(appt_id),
            start=start,
            service_title=service.title,
            master_name=self._masters.get(master_id, ""),
            client_name=client_name,
            client_phone=client_phone,
        )

    async def cancel(self, appointment_id: str) -> None:
        try:
            await self._call(self._api.appointments.remove, appointment_id)
        except Exception as exc:
            raise BookingError(f"Не удалось отменить запись в DIKIDI: {exc}") from exc

    async def get_appointments(self, date_from: date, date_to: date) -> list[Appointment]:
        records = await self._call(self._api.appointments.get_records, date_from.isoformat(), date_to.isoformat())
        result = []
        for r in records:
            start = self._parse_start(r)
            if start is None:
                continue
            result.append(
                Appointment(
                    id=str(_attr(r, "id", "appointment_id", "record_id")),
                    start=start,
                    service_title=str(_attr(r, "service_title", "service_name", "service", default="")),
                    master_name=str(_attr(r, "master_name", "master", default="")),
                    client_name=str(_attr(r, "client_name", "client", default="")),
                    client_phone=str(_attr(r, "client_phone", "phone", default="")),
                )
            )
        return sorted(result, key=lambda a: a.start)

    def _parse_start(self, r: Any) -> datetime | None:
        value = _attr(r, "start", "datetime", "time", "date_time")
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=self._s.tz)
        if not value:
            return None
        text = str(value).replace("T", " ")
        if len(text) <= 5 and _attr(r, "date"):  # отдельно дата и время
            text = f"{_attr(r, 'date')} {text}"
        try:
            return datetime.fromisoformat(text[:16]).replace(tzinfo=self._s.tz)
        except ValueError:
            return None

    async def close(self) -> None:
        await asyncio.to_thread(self._api.__exit__, None, None, None)
