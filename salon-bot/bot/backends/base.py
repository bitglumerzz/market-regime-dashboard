"""Общий интерфейс системы записи.

Бот работает только через этот интерфейс, поэтому источник правды
(DIKIDI или локальная база для тестов) подключается одной настройкой BACKEND.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class Service:
    id: str
    title: str
    duration: int  # минуты
    price: float | None = None


@dataclass(frozen=True)
class Master:
    id: str
    name: str


@dataclass(frozen=True)
class Appointment:
    id: str
    start: datetime
    service_title: str
    master_name: str
    client_name: str = ""
    client_phone: str = ""


class BookingError(Exception):
    """Не удалось создать/отменить запись (слот занят, ошибка API и т.п.)."""


class BookingBackend(abc.ABC):
    @abc.abstractmethod
    async def get_services(self) -> list[Service]: ...

    @abc.abstractmethod
    async def get_masters(self, service_id: str) -> list[Master]: ...

    @abc.abstractmethod
    async def get_free_slots(self, service: Service, master_id: str, day: date) -> list[datetime]:
        """Свободные начала сеанса (aware datetime в часовом поясе салона)."""

    @abc.abstractmethod
    async def book(
        self,
        service: Service,
        master_id: str,
        start: datetime,
        client_name: str,
        client_phone: str,
        comment: str = "",
    ) -> Appointment: ...

    @abc.abstractmethod
    async def cancel(self, appointment_id: str) -> None: ...

    @abc.abstractmethod
    async def get_appointments(self, date_from: date, date_to: date) -> list[Appointment]:
        """Все активные записи салона за период (включительно)."""

    async def close(self) -> None:
        return None
