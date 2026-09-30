from regime_bot.access import apply_payment, make_payload, parse_start_arg, validate_checkout
from regime_bot.db import DAY

from .conftest import NOW


def test_extend_from_now_then_from_end(db):
    db.upsert_user(5, "u", "U", now=NOW)
    e1 = db.extend_access(5, "m3", days=90, now=NOW)
    assert e1 == NOW + 90 * DAY
    e2 = db.extend_access(5, "m1", days=30, now=NOW + 10)          # продление от конца, а не от «сейчас»
    assert e2 == NOW + 120 * DAY
    e3 = db.extend_access(5, "m1", days=30, now=e2 + 5 * DAY)      # после истечения — от «сейчас»
    assert e3 == e2 + 35 * DAY


def test_checkout_validation(settings):
    m1 = settings.plan("m1")
    ok = make_payload(m1, 42)
    assert validate_checkout(settings, ok, m1.stars, "XTR", 42) is None
    assert "другому" in validate_checkout(settings, ok, m1.stars, "XTR", 43)
    assert "Цена" in validate_checkout(settings, ok, m1.stars - 1, "XTR", 42)
    assert "Цена" in validate_checkout(settings, ok, m1.stars, "USD", 42)
    assert validate_checkout(settings, "plan:zzz:42", 1, "XTR", 42)
    assert validate_checkout(settings, "garbage", 1, "XTR", 42)


def test_one_time_payment_idempotent(db, settings):
    db.upsert_user(7, "a", "A", now=NOW)
    m3 = settings.plan("m3")
    r1 = apply_payment(db, settings, 7, make_payload(m3, 7), m3.stars, "XTR", "ch-1", now=NOW)
    assert r1.applied and r1.ends_at == NOW + 90 * DAY
    r2 = apply_payment(db, settings, 7, make_payload(m3, 7), m3.stars, "XTR", "ch-1", now=NOW + 1)   # повтор апдейта
    assert not r2.applied and db.get_access(7).ends_at == NOW + 90 * DAY
    assert db.admin_stats(now=NOW)["stars_total"] == m3.stars


def test_recurring_subscription_uses_expiration_and_keeps_charge(db, settings):
    db.upsert_user(8, "b", "B", now=NOW)
    m1 = settings.plan("m1")
    exp = NOW + 30 * DAY
    r = apply_payment(db, settings, 8, make_payload(m1, 8), m1.stars, "XTR", "sub-first", is_recurring=True,
                      is_first_recurring=True, subscription_expiration=exp, now=NOW)
    acc = db.get_access(8)
    assert r.applied and not r.renewal
    assert acc.auto_renew and acc.sub_charge_id == "sub-first"
    assert acc.ends_at == exp + settings.grace_hours * 3600
    r2 = apply_payment(db, settings, 8, make_payload(m1, 8), m1.stars, "XTR", "sub-2", is_recurring=True,
                       subscription_expiration=exp + 30 * DAY, now=exp)
    acc = db.get_access(8)
    assert r2.renewal and acc.sub_charge_id == "sub-first"         # отмена идёт по id первой оплаты
    assert acc.ends_at == exp + 30 * DAY + settings.grace_hours * 3600


def test_referral_bonus_only_on_first_payment(db, settings):
    db.upsert_user(100, "ref", "R", now=NOW)
    db.upsert_user(101, "new", "N", referred_by=100, now=NOW)
    m3 = settings.plan("m3")
    r = apply_payment(db, settings, 101, make_payload(m3, 101), m3.stars, "XTR", "c1", now=NOW)
    assert r.referrer == 100 and db.get_access(100).ends_at == NOW + 7 * DAY
    r = apply_payment(db, settings, 101, make_payload(m3, 101), m3.stars, "XTR", "c2", now=NOW)
    assert r.referrer is None


def test_self_referral_ignored_and_source_kept(db):
    db.upsert_user(9, "x", "X", source="reels", referred_by=9, now=NOW)
    db.upsert_user(9, "x2", "X", source="other", now=NOW)
    u = db.get_user(9)
    assert u["referred_by"] is None and u["source"] == "reels" and u["username"] == "x2"


def test_start_args():
    assert parse_start_arg("ref_123") == (None, 123)
    assert parse_start_arg("src_reels_ad1") == ("reels_ad1", None)
    assert parse_start_arg("ref_bad") == (None, None)
    assert parse_start_arg(None) == (None, None)


def test_reminders_and_expiry(db):
    db.upsert_user(1, "a", "A", now=NOW)
    db.extend_access(1, "m1", days=3, now=NOW)
    assert db.due_reminders(NOW) == [(1, NOW + 3 * DAY, 1)]
    db.mark_reminded(1, 1)
    assert db.due_reminders(NOW) == []
    assert db.due_reminders(NOW + 2 * DAY + 1) == [(1, NOW + 3 * DAY, 2)]
    db.mark_reminded(1, 2)
    assert db.due_reminders(NOW + 2 * DAY + 1) == []
    assert db.expired_not_kicked(NOW + 3 * DAY) == [1]
    db.mark_kicked(1)
    assert db.expired_not_kicked(NOW + 4 * DAY) == []
    db.extend_access(1, "m1", days=30, now=NOW + 5 * DAY)          # продление сбрасывает флаги
    assert db.get_access(1).active(NOW + 6 * DAY)
