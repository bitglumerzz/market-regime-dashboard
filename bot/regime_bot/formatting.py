"""Оформление сигналов и дат для сообщений Telegram (HTML)."""
from __future__ import annotations

import html
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from . import texts

MSK = timezone(timedelta(hours=3))


def fmt_date(ts: int) -> str:
    return datetime.fromtimestamp(ts, MSK).strftime("%d.%m.%Y %H:%M МСК")


def fmt_price(x: float) -> str:
    digits = 0 if x >= 1000 else 2 if x >= 1 else 6
    s = f"{x:,.{digits}f}".replace(",", " ")
    return s


def pct(a: float, b: float) -> float:
    return (b - a) / a * 100 if a else 0.0


def _get(s: Mapping[str, Any], key: str) -> Any:
    try:
        return s[key]
    except (KeyError, IndexError):
        return None


def render_signal(s: Mapping[str, Any], number: int | None = None, disclaimer: bool = True) -> str:
    """Сообщение с сигналом. s — строка БД или словарь с полями сигнала."""
    targets = s["targets"]
    if isinstance(targets, str):
        targets = json.loads(targets)
    side = str(s["side"]).upper()
    icon = "🟢" if side == "LONG" else "🔴"
    entry, stop = float(s["entry"]), float(s["stop"])
    head = f"🤖 <b>СИГНАЛ ИИ{f' · #{number}' if number else ''}</b>"
    tf = _get(s, "timeframe")
    lines = [
        head,
        f"<b>{html.escape(str(s['asset']))}</b>{f' · {html.escape(str(tf))}' if tf else ''} · <b>{side}</b> {icon}",
        "",
        f"Вход: <b>{fmt_price(entry)}</b>",
        f"Стоп: {fmt_price(stop)} ({pct(entry, stop):+.1f}%)",
        "Цели: " + " · ".join(f"{fmt_price(float(t))} ({pct(entry, float(t)):+.1f}%)" for t in targets),
    ]
    conf = _get(s, "confidence")
    if conf is not None:
        conf = float(conf) * (100 if float(conf) <= 1 else 1)
        lines.append(f"Уверенность модели: <b>{conf:.0f}%</b>")
    for key, label in (("regime", "Режим рынка"), ("horizon", "Горизонт")):
        v = _get(s, key)
        if v:
            lines.append(f"{label}: {html.escape(str(v))}")
    why = _get(s, "rationale")
    if why:
        lines += ["", f"💬 {html.escape(str(why))}"]
    if disclaimer:
        lines += ["", texts.DISCLAIMER_SHORT]
    return "\n".join(lines)


STATUS_TEXT = {"tp": "✅ по цели", "sl": "🛑 по стопу", "closed": "☑️ закрыт", "cancelled": "↩️ отменён"}


def render_close(s: Mapping[str, Any], number: int | None = None) -> str:
    res = _get(s, "result_pct")
    status = STATUS_TEXT.get(str(s["status"]), "☑️ закрыт")
    head = f"<b>Сигнал{f' #{number}' if number else ''} · {html.escape(str(s['asset']))} {str(s['side']).upper()}</b> — {status}"
    if res is None:
        return head
    icon = "📈" if res > 0 else "📉"
    return f"{head}\nРезультат: {icon} <b>{res:+.2f}%</b> от цены входа"


def render_history_line(s: Mapping[str, Any]) -> str:
    res = _get(s, "result_pct")
    mark = "✅" if (res or 0) > 0 else "❌"
    return f"{mark} {html.escape(str(s['asset']))} {str(s['side']).upper()} · {res:+.2f}%" if res is not None else f"☑️ {s['asset']}"


EXAMPLE_SIGNAL = {
    "asset": "BTC/USDT", "timeframe": "4H", "side": "LONG", "entry": 64250, "stop": 63100,
    "targets": [66000, 67900], "confidence": 0.78, "regime": "бычий тренд", "horizon": "1–3 дня",
    "rationale": "Режим сменился на трендовый, коррекция завершилась у поддержки, объёмы растут.",
}
