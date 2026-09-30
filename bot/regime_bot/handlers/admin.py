"""Админ-команды (только для ADMIN_IDS)."""
from __future__ import annotations

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import Message

from .. import texts
from ..config import Settings
from ..db import Database
from ..formatting import EXAMPLE_SIGNAL, fmt_date, render_signal

router = Router(name="admin")


class IsAdmin:
    def __call__(self, message: Message, settings: Settings) -> bool:
        return bool(message.from_user) and settings.is_admin(message.from_user.id)


router.message.filter(IsAdmin())


@router.message(Command("admin"))
async def admin_help(message: Message) -> None:
    await message.answer(texts.ADMIN_HELP)


@router.message(Command("stats"))
async def stats(message: Message, db: Database) -> None:
    s = db.admin_stats()
    sig = db.signal_stats()
    src = "\n".join(f"  {name}: {users} польз. → {paid} оплат" for name, users, paid in s["sources"]) or "  —"
    await message.answer(
        f"<b>Статистика</b>\nПользователи: {s['users']} (+{s['new_7d']} за 7 дн.), приняли условия: {s['terms']}, заблокировали: {s['blocked']}\n"
        f"Активный доступ: <b>{s['active']}</b> (автопродление: {s['auto_renew']})\n"
        f"Выручка: <b>{s['stars_30d']} ⭐</b> за 30 дн. · всего {s['stars_total']} ⭐\n"
        f"Сигналы: закрыто {sig['closed']}, открыто {sig['open']}, winrate {sig['winrate']:.0f}%, средний {sig['avg']:+.2f}%\n\n"
        f"<b>Источники (src_…)</b>\n{src}")


@router.message(Command("grant"))
async def grant(message: Message, command: CommandObject, db: Database) -> None:
    try:
        tg_id, days = (int(x) for x in (command.args or "").split())
    except ValueError:
        await message.answer("Формат: /grant &lt;tg_id&gt; &lt;дней&gt;")
        return
    if not db.get_user(tg_id):
        db.upsert_user(tg_id, None, None, source="admin")
    ends = db.extend_access(tg_id, "gift", days=days)
    await message.answer(f"Выдано {days} дн. пользователю {tg_id}, доступ до {fmt_date(ends)}")


@router.message(Command("revoke"))
async def revoke(message: Message, command: CommandObject, db: Database) -> None:
    try:
        tg_id = int(command.args or "")
    except ValueError:
        await message.answer("Формат: /revoke &lt;tg_id&gt;")
        return
    db.revoke_access(tg_id)
    await message.answer(f"Доступ {tg_id} отозван (из канала уберёт планировщик в течение 10 минут)")


@router.message(Command("refund"))
async def refund(message: Message, command: CommandObject, bot: Bot, db: Database) -> None:
    charge = (command.args or "").strip()
    pay = db.get_payment(charge) if charge else None
    if not pay:
        await message.answer("Формат: /refund &lt;charge_id&gt; (id платежа есть в уведомлении об оплате и в БД)")
        return
    await bot.refund_star_payment(pay["tg_id"], charge)
    db.mark_refunded(charge)
    db.revoke_access(pay["tg_id"])
    await message.answer(f"Возвращено {pay['amount']} ⭐ пользователю {pay['tg_id']}, доступ отозван")


@router.message(Command("broadcast"))
async def broadcast(message: Message, command: CommandObject, db: Database, broadcaster) -> None:
    target = (command.args or "").strip()
    if target not in ("subs", "all") or not message.reply_to_message:
        await message.answer("Ответьте командой <code>/broadcast subs</code> или <code>/broadcast all</code> на сообщение, которое нужно разослать")
        return
    ids = db.active_subscribers() if target == "subs" else db.all_user_ids()
    await message.answer(f"Рассылка на {len(ids)} получателей началась…")
    ok = 0
    for tg_id in ids:
        ok += await broadcaster.copy(tg_id, message.chat.id, message.reply_to_message.message_id)
    await message.answer(f"Готово: доставлено {ok} из {len(ids)}")


@router.message(Command("testsignal"))
async def test_signal(message: Message) -> None:
    await message.answer(render_signal(EXAMPLE_SIGNAL, number=0))
