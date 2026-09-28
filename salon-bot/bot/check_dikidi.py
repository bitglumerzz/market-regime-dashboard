"""Проверка подключения к DIKIDI перед запуском бота.

    python -m bot.check_dikidi

Покажет услуги, мастеров, свободные окна на завтра и записи на неделю —
то же самое, что будет видеть бот. Ничего не создаёт и не меняет.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from .backends import create_backend
from .config import load_settings


async def main() -> None:
    settings = load_settings()
    backend = create_backend(settings)
    try:
        services = await backend.get_services()
        print(f"Услуги ({len(services)}):")
        for s in services:
            print(f"  [{s.id}] {s.title} — {s.duration} мин, {s.price}")
        if not services:
            return
        service = services[0]
        masters = await backend.get_masters(service.id)
        print(f"\nМастера для «{service.title}» ({len(masters)}):")
        for m in masters:
            print(f"  [{m.id}] {m.name}")
        tomorrow = datetime.now(settings.tz).date() + timedelta(days=1)
        for m in masters:
            slots = await backend.get_free_slots(service, m.id, tomorrow)
            print(f"\nСвободно у {m.name} на {tomorrow}: {', '.join(f'{s:%H:%M}' for s in slots) or 'нет'}")
        today = datetime.now(settings.tz).date()
        appts = await backend.get_appointments(today, today + timedelta(days=6))
        print(f"\nЗаписи на 7 дней ({len(appts)}):")
        for a in appts:
            print(f"  [{a.id}] {a.start:%d.%m %H:%M} {a.service_title} · {a.master_name} — {a.client_name} {a.client_phone}")
    finally:
        await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
