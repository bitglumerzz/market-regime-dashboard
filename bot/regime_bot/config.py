"""Настройки бота из переменных окружения (или файла .env рядом с запуском)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv(path: str | os.PathLike = ".env") -> None:
    """Минимальный загрузчик .env: KEY=VALUE, без перезаписи уже заданных переменных."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Plan:
    code: str
    title: str
    days: int
    stars: int
    recurring: bool = False          # Stars-подписка с автопродлением (Telegram поддерживает только 30 дней)
    badge: str = ""

    @property
    def button(self) -> str:
        tail = " · автопродление" if self.recurring else ""
        badge = f" {self.badge}" if self.badge else ""
        return f"{self.title} — {self.stars} ⭐{tail}{badge}"


DEFAULT_PLANS = (
    Plan("m1", "1 месяц", 30, 990, recurring=True),
    Plan("m3", "3 месяца", 90, 2490, badge="−16%"),
    Plan("y1", "12 месяцев", 365, 7990, badge="−33%"),
)

STARS_SUBSCRIPTION_PERIOD = 30 * 24 * 3600   # единственный период, который принимает Telegram


def _plans_from_env(raw: str | None) -> tuple[Plan, ...]:
    if not raw:
        return DEFAULT_PLANS
    plans = tuple(Plan(**item) for item in json.loads(raw))
    for p in plans:
        if p.recurring and p.days != 30:
            raise ValueError(f"План {p.code}: автопродление в Stars возможно только на 30 дней")
    return plans


def _ids(raw: str | None) -> frozenset[int]:
    return frozenset(int(x) for x in (raw or "").replace(" ", "").split(",") if x)


@dataclass(frozen=True)
class Settings:
    bot_token: str
    admin_ids: frozenset[int] = frozenset()
    db_path: str = "data/regime_bot.sqlite3"
    signal_secret: str = ""                  # общий секрет HMAC для локальной модели
    api_host: str = "0.0.0.0"
    api_port: int = 8081
    mode: str = "polling"                    # polling | webhook
    webhook_base: str = ""                   # https://bot.example.com  (для mode=webhook)
    webhook_secret: str = ""
    channel_id: int | None = None            # приватный канал с сигналами (необязательно)
    support: str = "@regime_support"
    terms_url: str = ""
    referral_bonus_days: int = 7
    grace_hours: int = 24                    # запас после окончания Stars-подписки, пока Telegram продлевает
    plans: tuple[Plan, ...] = field(default=DEFAULT_PLANS)

    def plan(self, code: str) -> Plan | None:
        return next((p for p in self.plans if p.code == code), None)

    def is_admin(self, tg_id: int) -> bool:
        return tg_id in self.admin_ids

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ.get
        token = e("BOT_TOKEN", "")
        if not token:
            raise RuntimeError("BOT_TOKEN не задан — скопируйте .env.example в .env и вставьте токен от @BotFather")
        channel = e("CHANNEL_ID")
        return cls(
            bot_token=token,
            admin_ids=_ids(e("ADMIN_IDS")),
            db_path=e("DB_PATH", cls.db_path),
            signal_secret=e("SIGNAL_API_SECRET", ""),
            api_host=e("API_HOST", cls.api_host),
            api_port=int(e("API_PORT", str(cls.api_port))),
            mode=e("MODE", cls.mode),
            webhook_base=e("WEBHOOK_BASE", "").rstrip("/"),
            webhook_secret=e("WEBHOOK_SECRET", ""),
            channel_id=int(channel) if channel else None,
            support=e("SUPPORT", cls.support),
            terms_url=e("TERMS_URL", ""),
            referral_bonus_days=int(e("REFERRAL_BONUS_DAYS", str(cls.referral_bonus_days))),
            grace_hours=int(e("GRACE_HOURS", str(cls.grace_hours))),
            plans=_plans_from_env(e("PLANS_JSON")),
        )
