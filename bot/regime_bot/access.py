"""Логика оплаты и доступа — без Telegram, чтобы её можно было тестировать напрямую."""
from __future__ import annotations

import time
from dataclasses import dataclass

from .config import Plan, Settings
from .db import Database


def make_payload(plan: Plan, tg_id: int) -> str:
    return f"plan:{plan.code}:{tg_id}"


def parse_payload(payload: str) -> tuple[str, int] | None:
    parts = payload.split(":")
    if len(parts) != 3 or parts[0] != "plan":
        return None
    try:
        return parts[1], int(parts[2])
    except ValueError:
        return None


def validate_checkout(settings: Settings, payload: str, amount: int, currency: str, from_id: int) -> str | None:
    """Проверка перед списанием (pre_checkout_query). None — всё в порядке, иначе текст ошибки для пользователя."""
    parsed = parse_payload(payload)
    if not parsed:
        return "Не удалось распознать заказ. Откройте тарифы заново."
    code, tg_id = parsed
    plan = settings.plan(code)
    if plan is None:
        return "Этот тариф больше не доступен. Откройте тарифы заново."
    if tg_id != from_id:
        return "Этот счёт выставлен другому пользователю."
    if currency != "XTR" or amount != plan.stars:
        return "Цена тарифа изменилась. Откройте тарифы заново."
    return None


@dataclass
class PaymentResult:
    applied: bool                   # False — повтор уже учтённого платежа
    plan: Plan | None
    ends_at: int = 0
    renewal: bool = False           # автоматическое продление Stars-подписки
    referrer: int | None = None     # кому начислен реферальный бонус
    referrer_until: int = 0


def apply_payment(db: Database, settings: Settings, tg_id: int, payload: str, amount: int, currency: str,
                  charge_id: str, *, is_recurring: bool = False, is_first_recurring: bool = False,
                  subscription_expiration: int | None = None, now: int | None = None) -> PaymentResult:
    """Зачесть успешный платёж: записать его (идемпотентно), продлить доступ, начислить бонус рефереру."""
    now = now or int(time.time())
    parsed = parse_payload(payload)
    plan = settings.plan(parsed[0]) if parsed else None
    if plan is None:                                     # тариф удалили после оплаты — даём 30 дней, чтобы не обидеть
        plan = Plan(parsed[0] if parsed else "unknown", "Доступ", 30, amount)
    first_payment = db.paid_count(tg_id) == 0
    if not db.record_payment(tg_id, plan.code, amount, currency, charge_id, recurring=is_recurring, now=now):
        acc = db.get_access(tg_id)
        return PaymentResult(False, plan, acc.ends_at if acc else 0)

    if is_recurring and subscription_expiration:
        # Stars-подписка: Telegram сам знает дату следующего списания; добавляем запас на время продления
        ends = db.extend_access(tg_id, plan.code, until=subscription_expiration + settings.grace_hours * 3600,
                                sub_charge_id=charge_id if is_first_recurring or not db.get_access(tg_id) else None,
                                auto_renew=True, now=now)
    else:
        ends = db.extend_access(tg_id, plan.code, days=plan.days, now=now)   # разовая покупка не трогает автопродление

    result = PaymentResult(True, plan, ends, renewal=is_recurring and not is_first_recurring)
    user = db.get_user(tg_id)
    if first_payment and user and user["referred_by"] and settings.referral_bonus_days > 0:
        ref = user["referred_by"]
        if db.get_user(ref):
            acc = db.get_access(ref)
            result.referrer = ref
            result.referrer_until = db.extend_access(ref, acc.plan if acc else "ref", days=settings.referral_bonus_days, now=now)
    return result


def parse_start_arg(arg: str | None) -> tuple[str | None, int | None]:
    """/start ref_123 → (None, 123); /start src_reels → ('reels', None)."""
    if not arg:
        return None, None
    if arg.startswith("ref_"):
        try:
            return None, int(arg[4:])
        except ValueError:
            return None, None
    if arg.startswith("src_"):
        return arg[4:][:32], None
    return arg[:32], None
