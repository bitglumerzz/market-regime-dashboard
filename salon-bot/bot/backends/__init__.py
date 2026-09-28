from __future__ import annotations

from ..config import Settings
from .base import Appointment, BookingBackend, BookingError, Master, Service


def create_backend(settings: Settings) -> BookingBackend:
    if settings.backend == "dikidi":
        from .dikidi import DikidiBackend

        return DikidiBackend(settings)
    if settings.backend == "local":
        from .local import LocalBackend

        return LocalBackend(settings)
    raise ValueError(f"Неизвестный BACKEND: {settings.backend!r} (ожидается dikidi или local)")


__all__ = [
    "Appointment",
    "BookingBackend",
    "BookingError",
    "Master",
    "Service",
    "create_backend",
]
