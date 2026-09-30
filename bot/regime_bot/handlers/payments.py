"""Оплата в Telegram Stars: тарифы → предупреждение о рисках → счёт → проверка → зачисление."""
from __future__ import annotations

import logging
import time

from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, LabeledPrice, Message, PreCheckoutQuery

from .. import keyboards as kb
from .. import texts
from ..access import apply_payment, make_payload, validate_checkout
from ..config import STARS_SUBSCRIPTION_PERIOD, Settings
from ..db import Database
from ..formatting import fmt_date
from .user import remember, show

log = logging.getLogger("regime_bot.payments")
router = Router(name="payments")


@router.message(Command("plans"))
@router.callback_query(F.data == "plans")
async def plans(event: Message | CallbackQuery, db: Database, settings: Settings) -> None:
    remember(db, event.from_user)
    if not db.terms_accepted(event.from_user.id):              # сначала — честное предупреждение о рисках
        await show(event, texts.RISK, kb.risk_kb(settings))
        return
    line = texts.CHANNEL_LINE if settings.channel_id else ""
    await show(event, texts.PLANS.format(channel_line=line), kb.plans_kb(settings))


@router.callback_query(F.data == "accept")
async def accept(cb: CallbackQuery, db: Database, settings: Settings) -> None:
    db.accept_terms(cb.from_user.id)
    await plans(cb, db, settings)


@router.callback_query(F.data.startswith("buy:"))
async def buy(cb: CallbackQuery, bot: Bot, db: Database, settings: Settings) -> None:
    plan = settings.plan(cb.data.split(":", 1)[1])
    if plan is None or not db.terms_accepted(cb.from_user.id):
        await plans(cb, db, settings)
        return
    title = f"{texts.BRAND} · {plan.title}"
    desc = texts.INVOICE_DESC.format(brand=texts.BRAND, title=plan.title)
    payload = make_payload(plan, cb.from_user.id)
    prices = [LabeledPrice(label=plan.title, amount=plan.stars)]
    await cb.answer()
    if plan.recurring:
        # подписка Stars с автосписанием раз в 30 дней — только через ссылку на счёт
        url = await bot.create_invoice_link(title=title, description=desc, payload=payload, currency="XTR",
                                            prices=prices, subscription_period=STARS_SUBSCRIPTION_PERIOD)
        await cb.message.answer(texts.SUB_LINK.format(stars=plan.stars), reply_markup=kb.pay_link_kb(url, plan.stars))
    else:
        await bot.send_invoice(cb.from_user.id, title=title, description=desc, payload=payload, currency="XTR", prices=prices)


@router.pre_checkout_query()
async def pre_checkout(q: PreCheckoutQuery, settings: Settings) -> None:
    error = validate_checkout(settings, q.invoice_payload, q.total_amount, q.currency, q.from_user.id)
    await q.answer(ok=error is None, error_message=error)


@router.message(F.successful_payment)
async def paid(message: Message, bot: Bot, db: Database, settings: Settings, broadcaster) -> None:
    sp = message.successful_payment
    remember(db, message.from_user)
    res = apply_payment(db, settings, message.from_user.id, sp.invoice_payload, sp.total_amount, sp.currency,
                        sp.telegram_payment_charge_id, is_recurring=bool(sp.is_recurring),
                        is_first_recurring=bool(sp.is_first_recurring), subscription_expiration=sp.subscription_expiration_date)
    log.info("payment %s from %s: %s", sp.telegram_payment_charge_id, message.from_user.id, res)
    if not res.applied:
        return
    if res.renewal:
        await message.answer(texts.RENEWED.format(until=fmt_date(res.ends_at)))
    else:
        renew = "\nАвтопродление: включено 🔄" if sp.is_recurring else ""
        await message.answer(texts.PAID.format(plan=res.plan.title, until=fmt_date(res.ends_at), renew=renew),
                             reply_markup=kb.access_kb(True, bool(sp.is_recurring), bool(settings.channel_id)))
        if settings.channel_id:
            link = await bot.create_chat_invite_link(settings.channel_id, name=f"u{message.from_user.id}", member_limit=1,
                                                     expire_date=int(time.time()) + 86400)
            await message.answer(f"{texts.CHANNEL_INVITE}\n{link.invite_link}")
    if res.referrer:
        await broadcaster.send(res.referrer, texts.REF_BONUS.format(days=settings.referral_bonus_days, until=fmt_date(res.referrer_until)))
    for admin in settings.admin_ids:
        await broadcaster.send(admin, f"💰 +{sp.total_amount} ⭐ · {res.plan.title} · "
                                      f"{message.from_user.full_name} ({message.from_user.id}){' · продление' if res.renewal else ''}")


@router.callback_query(F.data == "cancel_renew")
async def cancel_renew(cb: CallbackQuery, bot: Bot, db: Database) -> None:
    acc = db.get_access(cb.from_user.id)
    if not acc or not acc.auto_renew:
        await cb.answer("Автопродление уже отключено", show_alert=True)
        return
    if acc.sub_charge_id:
        await bot.edit_user_star_subscription(cb.from_user.id, acc.sub_charge_id, is_canceled=True)
    db.set_auto_renew(cb.from_user.id, False)
    await show(cb, texts.RENEW_CANCELLED.format(until=fmt_date(acc.ends_at)), kb.back("access"))
