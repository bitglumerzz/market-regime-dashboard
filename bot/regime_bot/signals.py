"""Приём сигналов от локальной модели: проверка подписи, валидация, рассылка подписчикам."""
from __future__ import annotations

import hashlib
import hmac
import logging
import time
from typing import Any

from .db import Database
from .formatting import render_close, render_signal

log = logging.getLogger("regime_bot.signals")

MAX_SKEW = 300          # подпись действительна 5 минут — защита от повтора перехваченного запроса


def sign(secret: str, timestamp: str, body: bytes) -> str:
    return hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256).hexdigest()


def verify(secret: str, timestamp: str | None, signature: str | None, body: bytes, now: int | None = None) -> bool:
    if not secret or not timestamp or not signature:
        return False
    try:
        ts = int(timestamp)
    except ValueError:
        return False
    if abs((now or int(time.time())) - ts) > MAX_SKEW:
        return False
    return hmac.compare_digest(sign(secret, timestamp, body), signature)


class SignalError(ValueError):
    pass


def validate_signal(d: dict[str, Any]) -> dict[str, Any]:
    """Проверить и нормализовать сигнал. Логика цен: у LONG стоп ниже входа, цели выше; у SHORT наоборот."""
    def need(key: str) -> Any:
        if d.get(key) in (None, ""):
            raise SignalError(f"нет поля {key}")
        return d[key]

    ext_id = str(need("ext_id"))[:64]
    asset = str(need("asset")).upper()[:24]
    side = str(need("side")).upper()
    if side not in ("LONG", "SHORT"):
        raise SignalError("side должен быть LONG или SHORT")
    try:
        entry, stop = float(need("entry")), float(need("stop"))
        targets = [float(x) for x in need("targets")]
    except (TypeError, ValueError):
        raise SignalError("entry, stop и targets должны быть числами")
    if not 1 <= len(targets) <= 5:
        raise SignalError("targets: от 1 до 5 целей")
    if min([entry, stop, *targets]) <= 0:
        raise SignalError("цены должны быть положительными")
    if side == "LONG" and not (stop < entry and all(t > entry for t in targets)):
        raise SignalError("LONG: стоп должен быть ниже входа, цели — выше")
    if side == "SHORT" and not (stop > entry and all(t < entry for t in targets)):
        raise SignalError("SHORT: стоп должен быть выше входа, цели — ниже")
    conf = d.get("confidence")
    if conf is not None:
        conf = float(conf)
        if conf > 1:
            conf /= 100
        if not 0 <= conf <= 1:
            raise SignalError("confidence: 0–1 или 0–100")
    clip = lambda k, n: (str(d[k])[:n] if d.get(k) not in (None, "") else None)
    return {"ext_id": ext_id, "asset": asset, "side": side, "entry": entry, "stop": stop, "targets": targets,
            "confidence": conf, "timeframe": clip("timeframe", 12), "regime": clip("regime", 60),
            "horizon": clip("horizon", 40), "rationale": clip("rationale", 700)}


CLOSE_STATUSES = ("tp", "sl", "closed", "cancelled")


class SignalService:
    """Сохраняет сигнал и рассылает его всем активным подписчикам (и в канал, если он задан)."""

    def __init__(self, db: Database, broadcaster, channel_id: int | None = None):
        self.db, self.bc, self.channel_id = db, broadcaster, channel_id

    async def publish(self, raw: dict[str, Any], now: int | None = None) -> dict[str, Any]:
        s = validate_signal(raw)
        row, created = self.db.add_signal(s, now=now)
        if not created:
            return {"status": "duplicate", "id": row["id"]}
        text = render_signal(row, number=row["id"])
        subs = self.db.active_subscribers(now)
        sent = 0
        for tg_id in subs:
            mid = await self.bc.send(tg_id, text)
            if mid is not None:
                self.db.add_delivery(row["id"], tg_id, mid, now=now)
                sent += 1
        if self.channel_id:
            await self.bc.send(self.channel_id, text)
        log.info("signal %s delivered to %s/%s", row["ext_id"], sent, len(subs))
        return {"status": "created", "id": row["id"], "delivered": sent, "subscribers": len(subs)}

    async def close(self, ext_id: str, status: str, result_pct: float | None, now: int | None = None) -> dict[str, Any]:
        if status not in CLOSE_STATUSES:
            raise SignalError(f"status: одно из {', '.join(CLOSE_STATUSES)}")
        if result_pct is not None:
            result_pct = float(result_pct)
        row = self.db.get_signal(ext_id)
        if row is None:
            raise SignalError("сигнал не найден")
        if not self.db.close_signal(ext_id, status, result_pct, now=now):
            return {"status": "already_closed", "id": row["id"]}
        row = self.db.get_signal(ext_id)
        text = render_close(row, number=row["id"])
        for tg_id, mid in self.db.delivered_to(row["id"]):             # итог — тем, кто получил сам сигнал, ответом на него
            await self.bc.send(tg_id, text, reply_to=mid)
        if self.channel_id:
            await self.bc.send(self.channel_id, text)
        return {"status": "closed", "id": row["id"]}
