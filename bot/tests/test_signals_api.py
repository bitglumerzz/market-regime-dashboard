import json
import time

import pytest
from aiohttp.test_utils import TestClient, TestServer

from push_signal import push_signal
from regime_bot.api import make_api
from regime_bot.formatting import render_signal
from regime_bot.scheduler import run_once
from regime_bot.signals import SignalError, SignalService, sign, validate_signal, verify

from .conftest import NOW, SIGNAL


def test_validate_normalizes_and_rejects():
    s = validate_signal(SIGNAL)
    assert s["asset"] == "BTC/USDT" and s["side"] == "LONG" and s["confidence"] == 0.78
    for bad in ({"stop": 65000}, {"targets": [60000]}, {"side": "BUY"}, {"targets": []}, {"entry": "x"}, {"confidence": 150}):
        with pytest.raises(SignalError):
            validate_signal({**SIGNAL, **bad})
    short = validate_signal({**SIGNAL, "side": "SHORT", "stop": 65000, "targets": [63000]})
    assert short["side"] == "SHORT"


def test_hmac():
    body = b'{"a":1}'
    ts = str(NOW)
    sig = sign("k", ts, body)
    assert verify("k", ts, sig, body, now=NOW + 10)
    assert not verify("k", ts, sig, body + b" ", now=NOW)
    assert not verify("other", ts, sig, body, now=NOW)
    assert not verify("k", ts, sig, body, now=NOW + 301)               # просроченная подпись
    assert not verify("", ts, sig, body, now=NOW)


def test_render_escapes_html():
    text = render_signal(validate_signal(SIGNAL), number=3)
    assert "#3" in text and "64 250" in text and "&lt;b&gt;" in text and "78%" in text and "+5.7%" in text


def _headers(secret, body):
    ts = str(int(time.time()))
    return {"X-Regime-Timestamp": ts, "X-Regime-Signature": sign(secret, ts, body), "Content-Type": "application/json"}


async def test_api_publish_close_flow(db, bc):
    for uid in (10, 11, 12):
        db.upsert_user(uid, None, None)
    db.extend_access(10, "m1", days=30)
    db.extend_access(11, "m1", days=30)
    bc.blocked.add(11)
    service = SignalService(db, bc, channel_id=-100500)
    async with TestClient(TestServer(make_api(service, "s3cret"))) as client:
        body = json.dumps(SIGNAL).encode()
        r = await client.post("/api/v1/signals", data=body, headers=_headers("s3cret", body))
        assert r.status == 201
        res = await r.json()
        assert res["delivered"] == 1 and res["subscribers"] == 2
        assert [m.chat_id for m in bc.sent] == [10, -100500]            # 12 без доступа, 11 заблокировал бота

        r = await client.post("/api/v1/signals", data=body, headers=_headers("s3cret", body))
        assert r.status == 200 and (await r.json())["status"] == "duplicate"
        assert len(bc.sent) == 2                                        # дубль не разослан

        r = await client.post("/api/v1/signals", data=body, headers=_headers("wrong", body))
        assert r.status == 401

        bad = json.dumps({**SIGNAL, "ext_id": "x2", "stop": 70000}).encode()
        r = await client.post("/api/v1/signals", data=bad, headers=_headers("s3cret", bad))
        assert r.status == 422

        cb = json.dumps({"status": "tp", "result_pct": 5.7}).encode()
        r = await client.post("/api/v1/signals/btc-1/close", data=cb, headers=_headers("s3cret", cb))
        assert r.status == 200
        close_msg = bc.sent[2]
        assert close_msg.chat_id == 10 and close_msg.reply_to == bc.sent[0].message_id and "+5.70%" in close_msg.text
        r = await client.post("/api/v1/signals/nope/close", data=cb, headers=_headers("s3cret", cb))
        assert r.status == 404

        assert (await (await client.get("/healthz")).json())["ok"]
    st = db.signal_stats()
    assert st["closed"] == 1 and st["wins"] == 1 and st["avg"] == pytest.approx(5.7)


async def test_python_client_against_live_api(db, bc, unused_tcp_port_factory=None):
    """push_signal.py (stdlib) подписывает запрос так же, как ждёт сервер."""
    import asyncio
    service = SignalService(db, bc)
    server = TestServer(make_api(service, "k"))
    await server.start_server()
    try:
        url = str(server.make_url("")).rstrip("/")
        res = await asyncio.to_thread(push_signal, {**SIGNAL, "ext_id": "cli-1"}, url=url, secret="k")
        assert res["status"] == "created"
    finally:
        await server.close()


class FakeBot:
    def __init__(self):
        self.calls = []

    async def ban_chat_member(self, *a, **k):
        self.calls.append(("ban", a))

    async def unban_chat_member(self, *a, **k):
        self.calls.append(("unban", a))


async def test_scheduler(db, bc, settings):
    from dataclasses import replace
    settings = replace(settings, channel_id=-1001)
    db.upsert_user(1, None, None, now=NOW)
    db.upsert_user(2, None, None, now=NOW)
    db.extend_access(1, "m1", days=2, now=NOW)
    db.extend_access(2, "m1", days=1, now=NOW - 2 * 86400)            # уже истёк
    bot = FakeBot()
    res = await run_once(bot, db, settings, bc, now=NOW)
    assert res == {"reminded": 1, "expired": 1}
    assert ("ban", (-1001, 2)) in [(c[0], c[1][:2]) for c in bot.calls]
    res = await run_once(bot, db, settings, bc, now=NOW)
    assert res == {"reminded": 0, "expired": 0}                         # повторно не спамим
