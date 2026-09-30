"""Сквозной тест: настоящие апдейты Telegram проходят через Dispatcher и обработчики,
вместо сети — сессия, которая записывает вызовы Bot API и отдаёт правдоподобные ответы."""
from datetime import datetime

import pytest
from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.methods import (AnswerCallbackQuery, AnswerPreCheckoutQuery, CreateInvoiceLink, EditMessageText, GetMe,
                             SendInvoice, SendMessage)
from aiogram.types import (CallbackQuery, Chat, Message, PreCheckoutQuery, SuccessfulPayment, Update, User)

from regime_bot import handlers
from regime_bot.access import make_payload

from .conftest import FakeBroadcaster

UID, ADMIN = 555, 1


class RecSession(BaseSession):
    def __init__(self):
        super().__init__()
        self.calls = []

    async def make_request(self, bot, method, timeout=None):
        self.calls.append(method)
        chat = Chat(id=getattr(method, "chat_id", UID) or UID, type="private")
        if isinstance(method, (SendMessage, EditMessageText, SendInvoice)):
            return Message(message_id=len(self.calls), date=datetime.now(), chat=chat, text=getattr(method, "text", ""))
        if isinstance(method, CreateInvoiceLink):
            return "https://t.me/$sub_invoice"
        if isinstance(method, GetMe):
            return User(id=42, is_bot=True, first_name="Regime", username="regime_ai_bot")
        return True

    async def stream_content(self, *a, **k):  # pragma: no cover
        yield b""

    async def close(self):
        pass

    def of(self, kind):
        return [c for c in self.calls if isinstance(c, kind)]


@pytest.fixture
def env(db, settings):
    session = RecSession()
    bot = Bot("123:abc", session=session)
    bc = FakeBroadcaster()
    dp = Dispatcher(db=db, settings=settings, broadcaster=bc, signals=None)
    dp.include_router(handlers.setup())
    return bot, dp, session, bc


def user(uid=UID):
    return User(id=uid, is_bot=False, first_name="Иван", username="ivan")


def msg(text, uid=UID, **extra):
    return Update(update_id=1, message=Message(message_id=1, date=datetime.now(), chat=Chat(id=uid, type="private"),
                                               from_user=user(uid), text=text, **extra))


def cb(data, uid=UID):
    m = Message(message_id=7, date=datetime.now(), chat=Chat(id=uid, type="private"), text="menu")
    return Update(update_id=2, callback_query=CallbackQuery(id="q", from_user=user(uid), chat_instance="c", data=data, message=m))


async def test_full_purchase_flow(env, db, settings):
    bot, dp, s, bc = env
    await dp.feed_update(bot, msg("/start src_reels"))
    assert db.get_user(UID)["source"] == "reels"
    assert "REGIME AI" in s.of(SendMessage)[-1].text

    await dp.feed_update(bot, cb("plans"))                         # без согласия — сначала риски
    assert "Прежде чем оформить" in s.of(EditMessageText)[-1].text
    await dp.feed_update(bot, cb("accept"))
    assert db.terms_accepted(UID) and "Тарифы" in s.of(EditMessageText)[-1].text

    await dp.feed_update(bot, cb("buy:m3"))
    inv = s.of(SendInvoice)[-1]
    assert inv.currency == "XTR" and inv.prices[0].amount == settings.plan("m3").stars

    await dp.feed_update(bot, cb("buy:m1"))                        # автопродление — ссылка на подписку
    link = s.of(CreateInvoiceLink)[-1]
    assert link.subscription_period == 30 * 86400

    payload = make_payload(settings.plan("m3"), UID)
    pre = Update(update_id=3, pre_checkout_query=PreCheckoutQuery(id="p", from_user=user(), currency="XTR",
                                                                  total_amount=settings.plan("m3").stars, invoice_payload=payload))
    await dp.feed_update(bot, pre)
    assert s.of(AnswerPreCheckoutQuery)[-1].ok is True
    tampered = Update(update_id=4, pre_checkout_query=PreCheckoutQuery(id="p2", from_user=user(), currency="XTR",
                                                                       total_amount=1, invoice_payload=payload))
    await dp.feed_update(bot, tampered)
    assert s.of(AnswerPreCheckoutQuery)[-1].ok is False

    sp = SuccessfulPayment(currency="XTR", total_amount=settings.plan("m3").stars, invoice_payload=payload,
                           telegram_payment_charge_id="tg-ch-1", provider_payment_charge_id="")
    await dp.feed_update(bot, msg(None, successful_payment=sp))
    acc = db.get_access(UID)
    assert acc and acc.active() and acc.plan == "m3"
    assert "Оплата прошла" in s.of(SendMessage)[-1].text
    assert any(m.chat_id == ADMIN and "⭐" in m.text for m in bc.sent)   # админу — уведомление о деньгах

    await dp.feed_update(bot, cb("access"))
    assert "активен" in s.of(EditMessageText)[-1].text


async def test_admin_and_public_stats(env, db):
    bot, dp, s, bc = env
    await dp.feed_update(bot, msg("/stats", uid=ADMIN))
    assert "Выручка" in s.of(SendMessage)[-1].text
    await dp.feed_update(bot, msg("/stats", uid=777))
    assert "Статистика сигналов" in s.of(SendMessage)[-1].text and "Выручка" not in s.of(SendMessage)[-1].text
    await dp.feed_update(bot, msg("/grant 777 5", uid=777))                  # не админ — команда игнорируется
    assert db.get_access(777) is None
    await dp.feed_update(bot, msg("/grant 777 5", uid=ADMIN))
    assert db.get_access(777).active()


async def test_signals_locked_without_access_and_referral(env, db):
    bot, dp, s, bc = env
    await dp.feed_update(bot, msg("/signals", uid=900))
    assert "доступны после" in s.of(SendMessage)[-1].text
    await dp.feed_update(bot, cb("ref", uid=900))
    assert "start=ref_900" in s.of(EditMessageText)[-1].text
    await dp.feed_update(bot, msg("/start ref_900", uid=901))
    assert db.get_user(901)["referred_by"] == 900
