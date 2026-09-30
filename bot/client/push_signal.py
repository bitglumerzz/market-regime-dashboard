"""Клиент для локальной модели: отправить сигнал в бота или закрыть его. Только стандартная библиотека Python.

Из кода модели:
    from push_signal import push_signal, close_signal
    push_signal({"ext_id": "btc-4h-2026-10-01-1", "asset": "BTC/USDT", "timeframe": "4H", "side": "LONG",
                 "entry": 64250, "stop": 63100, "targets": [66000, 67900], "confidence": 0.78,
                 "regime": "бычий тренд", "horizon": "1–3 дня", "rationale": "Коротко, почему."})
    close_signal("btc-4h-2026-10-01-1", status="tp", result_pct=5.7)

Из терминала:
    export REGIME_API_URL=https://bot.example.com  REGIME_API_SECRET=...
    python push_signal.py send signal.json
    python push_signal.py close btc-4h-2026-10-01-1 tp 5.7

ext_id — ваш уникальный id сигнала: повторная отправка с тем же ext_id безопасна (бот не разошлёт дубль),
поэтому при сетевой ошибке можно просто повторить запрос.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import time
import urllib.error
import urllib.request


def _post(path: str, payload: dict, url: str | None = None, secret: str | None = None, retries: int = 3) -> dict:
    url = (url or os.environ["REGIME_API_URL"]).rstrip("/") + path
    secret = secret or os.environ["REGIME_API_SECRET"]
    body = json.dumps(payload, ensure_ascii=False).encode()
    for attempt in range(retries):
        ts = str(int(time.time()))
        sig = hmac.new(secret.encode(), ts.encode() + b"." + body, hashlib.sha256).hexdigest()
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Content-Type": "application/json", "X-Regime-Timestamp": ts, "X-Regime-Signature": sig})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            if e.code < 500:                          # 401/422/404 — повтор не поможет
                raise RuntimeError(f"{e.code}: {detail}") from None
            err = f"{e.code}: {detail}"
        except (urllib.error.URLError, TimeoutError) as e:
            err = str(e)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"не удалось отправить после {retries} попыток: {err}")


def push_signal(signal: dict, **kw) -> dict:
    return _post("/api/v1/signals", signal, **kw)


def close_signal(ext_id: str, status: str = "closed", result_pct: float | None = None, **kw) -> dict:
    """status: tp (по цели) | sl (по стопу) | closed (вручную) | cancelled (не сработал, в статистику не идёт)."""
    return _post(f"/api/v1/signals/{ext_id}/close", {"status": status, "result_pct": result_pct}, **kw)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "send":
        print(push_signal(json.load(open(sys.argv[2], encoding="utf-8"))))
    elif len(sys.argv) >= 4 and sys.argv[1] == "close":
        print(close_signal(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else None))
    else:
        print(__doc__)
        sys.exit(1)
