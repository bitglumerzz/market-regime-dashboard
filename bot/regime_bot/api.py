"""HTTP API для локальной модели: сюда Mac присылает сигналы и их итоги.

POST /api/v1/signals              — новый сигнал (JSON, см. client/push_signal.py)
POST /api/v1/signals/{ext_id}/close — закрыть сигнал: {"status": "tp|sl|closed|cancelled", "result_pct": 5.2}
GET  /healthz                     — проверка живости (без подписи)

Каждый POST подписывается HMAC-SHA256: заголовки X-Regime-Timestamp и X-Regime-Signature,
подпись = hex(hmac(secret, timestamp + "." + тело)). Без правильной подписи — 401.
"""
from __future__ import annotations

import json
import logging

from aiohttp import web

from .signals import SignalError, SignalService, verify

log = logging.getLogger("regime_bot.api")


def make_api(service: SignalService, secret: str, app: web.Application | None = None) -> web.Application:
    app = app or web.Application(client_max_size=64 * 1024)

    async def read_signed(request: web.Request) -> dict:
        body = await request.read()
        if not verify(secret, request.headers.get("X-Regime-Timestamp"), request.headers.get("X-Regime-Signature"), body):
            raise web.HTTPUnauthorized(text=json.dumps({"error": "bad signature"}), content_type="application/json")
        try:
            data = json.loads(body)
            if not isinstance(data, dict):
                raise ValueError
            return data
        except ValueError:
            raise web.HTTPBadRequest(text=json.dumps({"error": "invalid json"}), content_type="application/json")

    async def create(request: web.Request) -> web.Response:
        data = await read_signed(request)
        try:
            res = await service.publish(data)
        except SignalError as e:
            return web.json_response({"error": str(e)}, status=422)
        return web.json_response(res, status=201 if res["status"] == "created" else 200)

    async def close(request: web.Request) -> web.Response:
        data = await read_signed(request)
        try:
            res = await service.close(request.match_info["ext_id"], str(data.get("status", "closed")), data.get("result_pct"))
        except SignalError as e:
            return web.json_response({"error": str(e)}, status=422 if "не найден" not in str(e) else 404)
        return web.json_response(res)

    async def health(_: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    app.router.add_post("/api/v1/signals", create)
    app.router.add_post("/api/v1/signals/{ext_id}/close", close)
    app.router.add_get("/healthz", health)
    return app
