"""SQLite-хранилище: пользователи, доступ, платежи, сигналы, доставки.

Время везде — целые секунды Unix (UTC). Запросы синхронные: их мало и они быстрые,
а WAL позволяет читать параллельно с записью.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DAY = 86400

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    tg_id          INTEGER PRIMARY KEY,
    username       TEXT,
    first_name     TEXT,
    source         TEXT,                 -- метка рекламной кампании из /start src_xxx
    referred_by    INTEGER,
    terms_at       INTEGER,              -- когда принял предупреждение о рисках
    blocked        INTEGER NOT NULL DEFAULT 0,
    created_at     INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS access (
    tg_id          INTEGER PRIMARY KEY REFERENCES users(tg_id),
    plan           TEXT NOT NULL,
    ends_at        INTEGER NOT NULL,
    sub_charge_id  TEXT,                 -- charge_id Stars-подписки с автопродлением
    auto_renew     INTEGER NOT NULL DEFAULT 0,
    reminded       INTEGER NOT NULL DEFAULT 0,   -- битовая маска: 1 = за 3 дня, 2 = за 1 день
    kicked         INTEGER NOT NULL DEFAULT 0,
    updated_at     INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS payments (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    tg_id          INTEGER NOT NULL,
    plan           TEXT NOT NULL,
    amount         INTEGER NOT NULL,
    currency       TEXT NOT NULL,
    charge_id      TEXT NOT NULL UNIQUE, -- telegram_payment_charge_id — защита от двойного зачёта
    recurring      INTEGER NOT NULL DEFAULT 0,
    status         TEXT NOT NULL DEFAULT 'paid',   -- paid | refunded
    created_at     INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS signals (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ext_id         TEXT NOT NULL UNIQUE, -- id сигнала в локальной модели (идемпотентность)
    asset          TEXT NOT NULL,
    timeframe      TEXT,
    side           TEXT NOT NULL,
    entry          REAL NOT NULL,
    stop           REAL NOT NULL,
    targets        TEXT NOT NULL,        -- JSON-массив
    confidence     REAL,
    regime         TEXT,
    horizon        TEXT,
    rationale      TEXT,
    status         TEXT NOT NULL DEFAULT 'open',   -- open | tp | sl | closed | cancelled
    result_pct     REAL,
    created_at     INTEGER NOT NULL,
    closed_at      INTEGER
);
CREATE TABLE IF NOT EXISTS deliveries (
    signal_id      INTEGER NOT NULL,
    tg_id          INTEGER NOT NULL,
    message_id     INTEGER,
    sent_at        INTEGER NOT NULL,
    PRIMARY KEY (signal_id, tg_id)
);
CREATE INDEX IF NOT EXISTS idx_access_ends ON access(ends_at);
CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status, closed_at);
"""


@dataclass
class Access:
    tg_id: int
    plan: str
    ends_at: int
    auto_renew: bool
    sub_charge_id: str | None

    def active(self, now: int | None = None) -> bool:
        return self.ends_at > (now if now is not None else int(time.time()))

    def days_left(self, now: int | None = None) -> int:
        now = now if now is not None else int(time.time())
        return max(0, -(-(self.ends_at - now) // DAY))


class Database:
    def __init__(self, path: str | Path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def _one(self, sql: str, *args: Any) -> sqlite3.Row | None:
        return self.conn.execute(sql, args).fetchone()

    def _all(self, sql: str, *args: Any) -> list[sqlite3.Row]:
        return self.conn.execute(sql, args).fetchall()

    # ------------------------------------------------------------------ пользователи
    def upsert_user(self, tg_id: int, username: str | None, first_name: str | None,
                    source: str | None = None, referred_by: int | None = None, now: int | None = None) -> bool:
        """Создать или обновить пользователя. Возвращает True, если пользователь новый.
        Источник и реферер записываются только при первом входе."""
        now = now or int(time.time())
        if referred_by == tg_id:
            referred_by = None
        cur = self.conn.execute(
            "INSERT INTO users(tg_id, username, first_name, source, referred_by, created_at) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(tg_id) DO NOTHING", (tg_id, username, first_name, source, referred_by, now))
        if cur.rowcount == 0:
            self.conn.execute("UPDATE users SET username=?, first_name=?, blocked=0 WHERE tg_id=?", (username, first_name, tg_id))
            return False
        return True

    def get_user(self, tg_id: int) -> sqlite3.Row | None:
        return self._one("SELECT * FROM users WHERE tg_id=?", tg_id)

    def accept_terms(self, tg_id: int, now: int | None = None) -> None:
        self.conn.execute("UPDATE users SET terms_at=? WHERE tg_id=? AND terms_at IS NULL", (now or int(time.time()), tg_id))

    def terms_accepted(self, tg_id: int) -> bool:
        row = self._one("SELECT terms_at FROM users WHERE tg_id=?", tg_id)
        return bool(row and row["terms_at"])

    def set_blocked(self, tg_id: int, blocked: bool = True) -> None:
        self.conn.execute("UPDATE users SET blocked=? WHERE tg_id=?", (int(blocked), tg_id))

    def all_user_ids(self) -> list[int]:
        return [r[0] for r in self._all("SELECT tg_id FROM users WHERE blocked=0")]

    # ------------------------------------------------------------------ доступ
    def get_access(self, tg_id: int) -> Access | None:
        r = self._one("SELECT * FROM access WHERE tg_id=?", tg_id)
        return Access(r["tg_id"], r["plan"], r["ends_at"], bool(r["auto_renew"]), r["sub_charge_id"]) if r else None

    def extend_access(self, tg_id: int, plan: str, days: int = 0, until: int | None = None,
                      sub_charge_id: str | None = None, auto_renew: bool | None = None, now: int | None = None) -> int:
        """Продлить доступ на days дней от max(сейчас, текущий конец) или до момента until (что позже).
        Возвращает новый момент окончания."""
        now = now or int(time.time())
        cur = self.get_access(tg_id)
        base = max(now, cur.ends_at) if cur else now
        ends = base + days * DAY
        if until is not None:
            ends = max(ends, until)
        renew = int(auto_renew) if auto_renew is not None else int(cur.auto_renew) if cur else 0
        charge = sub_charge_id or (cur.sub_charge_id if cur else None)
        self.conn.execute(
            "INSERT INTO access(tg_id, plan, ends_at, sub_charge_id, auto_renew, reminded, kicked, updated_at) "
            "VALUES (?,?,?,?,?,0,0,?) ON CONFLICT(tg_id) DO UPDATE SET plan=excluded.plan, ends_at=excluded.ends_at, "
            "sub_charge_id=excluded.sub_charge_id, auto_renew=excluded.auto_renew, reminded=0, kicked=0, updated_at=excluded.updated_at",
            (tg_id, plan, ends, charge, renew, now))
        return ends

    def set_auto_renew(self, tg_id: int, on: bool) -> None:
        self.conn.execute("UPDATE access SET auto_renew=? WHERE tg_id=?", (int(on), tg_id))

    def revoke_access(self, tg_id: int, now: int | None = None) -> None:
        now = now or int(time.time())
        self.conn.execute("UPDATE access SET ends_at=?, auto_renew=0, updated_at=? WHERE tg_id=?", (now - 1, now, tg_id))

    def active_subscribers(self, now: int | None = None) -> list[int]:
        now = now or int(time.time())
        return [r[0] for r in self._all(
            "SELECT a.tg_id FROM access a JOIN users u USING(tg_id) WHERE a.ends_at > ? AND u.blocked = 0", now)]

    def due_reminders(self, now: int | None = None) -> list[tuple[int, int, int]]:
        """(tg_id, ends_at, бит напоминания) для тех, кому пора напомнить: за 3 дня и за 1 день.
        Подписчикам с автопродлением не напоминаем."""
        now = now or int(time.time())
        out = []
        for r in self._all("SELECT tg_id, ends_at, reminded FROM access WHERE ends_at > ? AND auto_renew = 0", now):
            left = r["ends_at"] - now
            if left <= DAY and not r["reminded"] & 2:
                out.append((r["tg_id"], r["ends_at"], 2))
            elif left <= 3 * DAY and not r["reminded"] & 1:
                out.append((r["tg_id"], r["ends_at"], 1))
        return out

    def mark_reminded(self, tg_id: int, bit: int) -> None:
        self.conn.execute("UPDATE access SET reminded = reminded | ? | CASE WHEN ?=2 THEN 1 ELSE 0 END WHERE tg_id=?", (bit, bit, tg_id))

    def expired_not_kicked(self, now: int | None = None) -> list[int]:
        now = now or int(time.time())
        return [r[0] for r in self._all("SELECT tg_id FROM access WHERE ends_at <= ? AND kicked = 0", now)]

    def mark_kicked(self, tg_id: int) -> None:
        self.conn.execute("UPDATE access SET kicked=1 WHERE tg_id=?", (tg_id,))

    # ------------------------------------------------------------------ платежи
    def record_payment(self, tg_id: int, plan: str, amount: int, currency: str, charge_id: str,
                       recurring: bool = False, now: int | None = None) -> bool:
        """Записать платёж. False — такой charge_id уже был (повторная доставка апдейта)."""
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO payments(tg_id, plan, amount, currency, charge_id, recurring, created_at) VALUES (?,?,?,?,?,?,?)",
            (tg_id, plan, amount, currency, charge_id, int(recurring), now or int(time.time())))
        return cur.rowcount == 1

    def paid_count(self, tg_id: int) -> int:
        return self._one("SELECT COUNT(*) FROM payments WHERE tg_id=? AND status='paid'", tg_id)[0]

    def get_payment(self, charge_id: str) -> sqlite3.Row | None:
        return self._one("SELECT * FROM payments WHERE charge_id=?", charge_id)

    def last_payment(self, tg_id: int) -> sqlite3.Row | None:
        return self._one("SELECT * FROM payments WHERE tg_id=? ORDER BY id DESC LIMIT 1", tg_id)

    def mark_refunded(self, charge_id: str) -> None:
        self.conn.execute("UPDATE payments SET status='refunded' WHERE charge_id=?", (charge_id,))

    # ------------------------------------------------------------------ сигналы
    def add_signal(self, s: dict[str, Any], now: int | None = None) -> tuple[sqlite3.Row, bool]:
        """Добавить сигнал. (строка, True) — новый; (строка, False) — такой ext_id уже был."""
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO signals(ext_id, asset, timeframe, side, entry, stop, targets, confidence, regime, horizon, rationale, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (s["ext_id"], s["asset"], s.get("timeframe"), s["side"], s["entry"], s["stop"], json.dumps(s["targets"]),
             s.get("confidence"), s.get("regime"), s.get("horizon"), s.get("rationale"), now or int(time.time())))
        return self._one("SELECT * FROM signals WHERE ext_id=?", s["ext_id"]), cur.rowcount == 1

    def get_signal(self, ext_id: str) -> sqlite3.Row | None:
        return self._one("SELECT * FROM signals WHERE ext_id=?", ext_id)

    def close_signal(self, ext_id: str, status: str, result_pct: float | None, now: int | None = None) -> bool:
        cur = self.conn.execute(
            "UPDATE signals SET status=?, result_pct=?, closed_at=? WHERE ext_id=? AND status='open'",
            (status, result_pct, now or int(time.time()), ext_id))
        return cur.rowcount == 1

    def recent_signals(self, limit: int = 5, only_closed: bool = False) -> list[sqlite3.Row]:
        where = "WHERE status != 'open' AND status != 'cancelled'" if only_closed else ""
        order = "closed_at" if only_closed else "id"
        return self._all(f"SELECT * FROM signals {where} ORDER BY {order} DESC LIMIT ?", limit)

    def signal_stats(self, since: int | None = None) -> dict[str, float]:
        """Честная статистика по закрытым сигналам (отменённые не считаются)."""
        rows = self._all("SELECT result_pct FROM signals WHERE status IN ('tp','sl','closed') AND result_pct IS NOT NULL AND closed_at >= ?",
                         since or 0)
        res = [r[0] for r in rows]
        wins = [x for x in res if x > 0]
        losses = [x for x in res if x <= 0]
        return {
            "closed": len(res),
            "wins": len(wins),
            "winrate": 100 * len(wins) / len(res) if res else 0.0,
            "avg": sum(res) / len(res) if res else 0.0,
            "sum": sum(res),
            "avg_win": sum(wins) / len(wins) if wins else 0.0,
            "avg_loss": sum(losses) / len(losses) if losses else 0.0,
            "open": self._one("SELECT COUNT(*) FROM signals WHERE status='open'")[0],
        }

    def add_delivery(self, signal_id: int, tg_id: int, message_id: int | None, now: int | None = None) -> None:
        self.conn.execute("INSERT OR IGNORE INTO deliveries(signal_id, tg_id, message_id, sent_at) VALUES (?,?,?,?)",
                          (signal_id, tg_id, message_id, now or int(time.time())))

    def delivered_to(self, signal_id: int) -> list[tuple[int, int | None]]:
        return [(r[0], r[1]) for r in self._all("SELECT tg_id, message_id FROM deliveries WHERE signal_id=?", signal_id)]

    # ------------------------------------------------------------------ отчёты
    def admin_stats(self, now: int | None = None) -> dict[str, Any]:
        now = now or int(time.time())
        q = lambda sql, *a: self._one(sql, *a)[0]
        return {
            "users": q("SELECT COUNT(*) FROM users"),
            "new_7d": q("SELECT COUNT(*) FROM users WHERE created_at > ?", now - 7 * DAY),
            "terms": q("SELECT COUNT(*) FROM users WHERE terms_at IS NOT NULL"),
            "blocked": q("SELECT COUNT(*) FROM users WHERE blocked=1"),
            "active": q("SELECT COUNT(*) FROM access WHERE ends_at > ?", now),
            "auto_renew": q("SELECT COUNT(*) FROM access WHERE ends_at > ? AND auto_renew=1", now),
            "stars_30d": q("SELECT COALESCE(SUM(amount),0) FROM payments WHERE status='paid' AND currency='XTR' AND created_at > ?", now - 30 * DAY),
            "stars_total": q("SELECT COALESCE(SUM(amount),0) FROM payments WHERE status='paid' AND currency='XTR'"),
            "sources": [(r[0] or "—", r[1], r[2]) for r in self._all(
                "SELECT u.source, COUNT(DISTINCT u.tg_id), COUNT(DISTINCT p.tg_id) FROM users u "
                "LEFT JOIN payments p ON p.tg_id = u.tg_id AND p.status='paid' GROUP BY u.source ORDER BY 2 DESC LIMIT 10")],
        }

    def iter_rows(self, sql: str, *args: Any) -> Iterable[sqlite3.Row]:
        return self._all(sql, *args)
