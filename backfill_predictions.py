"""One-shot backfill of historical predictions from earlier UI sessions.

We don't have a persistent log from before prediction_log.py was wired in.
But we have the trade plans from screenshots — entry / stop / target / RR /
reasoning are all known exactly. The only thing we estimate is the precise
minute timestamp of each prediction.

Run once with:
    python backfill_predictions.py

Each record is tagged `"backfilled": true` so you can tell reconstructed
predictions from live-logged ones.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import prediction_log as PL


# Each entry: (estimated_time, ticker, type, opinion_dict, current_price, extra)
# Times are conservative — earlier than "now" so verifier can check forward bars.
BACKFILL: list[dict] = [

    # ------------------------------------------------------------------
    # Older ETH per-TF predictions (from the very first batch of screenshots)
    # ------------------------------------------------------------------
    {
        "ticker": "ETH-USD",
        "type": "per_tf",
        "timeframe": "4h",
        "logged_at": "2026-05-13T05:00:00+00:00",
        "current_price": 2335.0,
        "action": "short",
        "entry": 2305.0,
        "stop_loss": 2456.0,
        "target": 2215.0,
        "agrees_with_top": True,
        "confidence": "medium",
        "preferred_pattern": "correction_down",
        "reasoning": "4h: rule-engine high confidence correction_down с целями "
                     "2267/2213/2180, инвалидация выше 2454 — основной драйвер шорт-сетапа.",
        "agrees_with_forecast": True,
        "forecast_critique": "",
        "forecast_pattern": "correction_down",
        "forecast_direction": "down",
    },

    {
        "ticker": "ETH-USD",
        "type": "per_tf",
        "timeframe": "1d",
        "logged_at": "2026-05-13T05:05:00+00:00",
        "current_price": 2280.0,
        "action": "wait",
        "entry": None,
        "stop_loss": None,
        "target": None,
        "agrees_with_top": False,
        "confidence": "low",
        "preferred_pattern": "импульс вниз с пика 4240 — развивается волна 5 или коррекция (B) внутри более крупной структуры",
        "reasoning": "Топ-гипотеза 'impulse up' с 1 нарушением выглядит натянуто: общая "
                     "структура с октября — серия снижающихся максимумов (4240→4158→3583→3354→2419), "
                     "типично для нисходящего импульса/зигзага. Пока цена ниже 2419, "
                     "приоритет на продолжение падения; вход против тренда нерационален.",
        "agrees_with_forecast": False,
        "forecast_critique": "",
        "forecast_pattern": None,
        "forecast_direction": None,
    },

    {
        "ticker": "ETH-USD",
        "type": "per_tf",
        "timeframe": "15m",
        "logged_at": "2026-05-13T05:10:00+00:00",
        "current_price": 2306.0,
        "action": "short",
        "entry": 2305.0,
        "stop_loss": 2319.0,
        "target": 2283.0,
        "agrees_with_top": True,
        "confidence": "medium",
        "preferred_pattern": "A-B-C коррекция вверх завершена (от 2259.59)",
        "reasoning": "Структура от лоу 2259.59 — трёхволновка ABC (A=2286.65, B=2274.96, "
                     "C=2306.67), где C≈A по длине. Уровень отмены 2318.20 разумен — выше него "
                     "ломается медвежий сценарий. Цель 2283 реалистична как 50% откат ABC.",
        "agrees_with_forecast": True,
        "forecast_critique": "Цели агрессивные но достижимы при пробое 2259.",
        "forecast_pattern": "correction_down",
        "forecast_direction": "down",
    },

    {
        "ticker": "ETH-USD",
        "type": "per_tf",
        "timeframe": "5m",
        "logged_at": "2026-05-13T05:15:00+00:00",
        "current_price": 2272.0,
        "action": "long",
        "entry": 2260.0,
        "stop_loss": 2213.0,
        "target": 2342.0,
        "agrees_with_top": True,
        "confidence": "medium",
        "preferred_pattern": "A-B-C коррекция вниз завершена на 2258.87",
        "reasoning": "Потенциальное двойное дно на 2258 рядом с предыдущим лоу 2258.24, "
                     "коррекция ABC вниз завершена. Risk/reward благоприятный: риск ~46 пунктов "
                     "до 2213, цель первой волны ~84 пункта (R/R ~1.8). Точка provisional — "
                     "нужно подтверждение разворотом, поэтому стоп обязателен.",
        "agrees_with_forecast": True,
        "forecast_critique": "",
        "forecast_pattern": "impulse_up",
        "forecast_direction": "up",
    },

    # ------------------------------------------------------------------
    # First ETH-USD holistic prediction (R/R 2.57)
    # ------------------------------------------------------------------
    {
        "ticker": "ETH-USD",
        "type": "holistic",
        "logged_at": "2026-05-13T05:30:00+00:00",
        "current_price": 2300.0,
        "bias": "short",
        "setup_quality": "medium",
        "summary": "На 1d структура неясна после завершения импульса вверх и текущей "
                   "коррекции — цена откатила от 2419 к 2253. На 4h rule-engine даёт высокую "
                   "уверенность в коррекции вниз с целями 2267/2213/2180 и инвалидацией 2454. "
                   "На 15m только что сформирован импульс вниз от 2342 с целью 2230 и отменой "
                   "выше 2318. На 5m локально свинг лоу 2258 удерживается, но контр-тренд "
                   "отскок выглядит как B-волна перед продолжением вниз — совокупно картина "
                   "медвежья с потенциалом на 2210-2180.",
        "action": "short",
        "entry": 2306.67,
        "stop_loss": 2343.00,
        "target": 2213.30,
        "risk_reward": 2.57,
        "invalidation_explained": "Закрытие 15m выше 2343 (последний значимый swing high 2342.14) "
                                   "ломает медвежью структуру; пробой 2454 на 4h полностью отменяет "
                                   "сценарий коррекции вниз и возвращает к бычьему импульсу.",
        "per_tf_notes": {
            "1d": "Импульс вверх завершён, рынок в коррекции; структура неоднозначная, но локально под давлением после отката от 2419.",
            "4h": "Rule-engine high-confidence correction_down с целями 2267→2213→2180, инвалидация 2454 — основной драйвер шорт-сетапа.",
            "15m": "Активный impulse_down от 2342, цель 2230, отмена 2318; текущий отскок к 2306 — возможная волна 2/B для входа.",
            "5m": "Локальный лоу 2258 пока держится, но отскок слабый; пробой 2258 откроет путь к 4h целям.",
        },
    },

    # ------------------------------------------------------------------
    # First BTC-USD holistic prediction (R/R 4.40 — exceptional setup)
    # ------------------------------------------------------------------
    {
        "ticker": "BTC-USD",
        "type": "holistic",
        "logged_at": "2026-05-13T05:45:00+00:00",
        "current_price": 82210.0,
        "bias": "short",
        "setup_quality": "medium",
        "summary": "На 1d виден завершённый импульс вниз с февральского максимума 96951 до "
                   "62909, последующий ABC-отскок выдохся около 82500 и рынок начал новый "
                   "разворот вниз — это согласуется с 4h, где также фиксируется завершение "
                   "коррекции вверх и старт импульса down. Младшие TF (15m/5m) показывают "
                   "вялый отскок после падения с 82373 к 79941 и текущий подъём к 81240, что "
                   "типично для волны 2/B перед продолжением снижения. Цена 82210 находится у "
                   "верхней границы отскока, и пока не пробит уровень инвалидации 82785, "
                   "перевес на стороне продавцов с целями в район 75-77k и далее ниже.",
        "action": "short",
        "entry": 81800.0,
        "stop_loss": 82800.0,
        "target": 77400.0,
        "risk_reward": 4.40,
        "invalidation_explained": "Закрытие 4h-свечи выше 82785 (макс. 5m-структуры) и особенно "
                                   "выход за 4h-максимум 82522 с подтверждением — это нарушит структуру "
                                   "нисходящих максимумов и аннулирует сценарий импульса вниз; в этом "
                                   "случае более вероятен возврат к 84-86k.",
        "per_tf_notes": {
            "1d": "Импульс вниз от 96951 к 62909 завершён, текущий рост — коррекция ABC, которая уже теряет силу у 82k; rule-engine ожидает новый импульс вниз с целями 69k и 57k.",
            "4h": "ABC-отскок завершён около 82522, начался разворот вниз; пока цена ниже 82785 структура остаётся медвежьей с целями 69-57k.",
            "15m": "Структура неясная, идёт борьба в диапазоне 79.6-82.4k; текущий подскок к 81240 выглядит как коррекция перед продолжением снижения.",
            "5m": "Чёткий нижний максимум 82048 против 82373, после слома 79941 рынок отскакивает — классическая зона входа в шорт у 81.5-82k с близким стопом.",
        },
    },

    # ------------------------------------------------------------------
    # Later ETH-USD holistic (less attractive R/R 1.12)
    # ------------------------------------------------------------------
    {
        "ticker": "ETH-USD",
        "type": "holistic",
        "logged_at": "2026-05-13T07:30:00+00:00",
        "current_price": 2320.0,
        "bias": "short",
        "setup_quality": "medium",
        "summary": "На 1d структура с февральским обвалом 3026→1826 выглядит как импульс вниз "
                   "с последующим откатом до 2419, и текущее снижение от 2419 похоже на начало "
                   "новой нисходящей волны старшего порядка. На 4h rule-engine уверенно видит "
                   "коррекцию вниз с целями 2267/2213/2180 и инвалидацией выше 2454. На 15m "
                   "только что отработал ABC-up до 2342 и пошёл слом вниз — это согласуется со "
                   "старшими TF. 5m даёт смешанный сигнал (отскок к 2320), но это лишь "
                   "техническая коррекция внутри более крупного down-движения.",
        "action": "short",
        "entry": 2318.0,
        "stop_loss": 2410.0,
        "target": 2215.0,
        "risk_reward": 1.12,
        "invalidation_explained": "Пробой и закрытие 4h выше 2410-2454 отменяет сценарий "
                                   "коррекции вниз: это перекроет последний high 4h и сделает "
                                   "структуру bullish continuation, а не correction down.",
        "per_tf_notes": {
            "1d": "После пика 2419 (17 апреля) сформирован lower high относительно мартовского 2352? Нет — выше, но в контексте обвала с 3026 это лишь коррекционный отскок 38%.",
            "4h": "Rule-engine high confidence: correction_down с целями 2267/2213/2180, инвалидация 2454. Структура от 2410 (6 мая) к 2233 и отскок к 2410 укладывается в A-B, ждём C вниз.",
            "15m": "ABC-up завершён на 2342, локальный импульс вниз к 2259, отскок к 2320 — потенциальная точка входа в шорт с инвалидацией выше 2342.",
            "5m": "Текущий отскок 2258→2320 выглядит как коррекционная волна 2 после impulse down; rule-engine лонг-сценарий имеет всего medium confidence и противоречит старшим TF.",
        },
    },

    # ------------------------------------------------------------------
    # Later BTC-USD holistic — NEUTRAL/WAIT (price moved against the setup)
    # ------------------------------------------------------------------
    {
        "ticker": "BTC-USD",
        "type": "holistic",
        "logged_at": "2026-05-13T08:30:00+00:00",
        "current_price": 82200.0,
        "bias": "neutral",
        "setup_quality": "low",
        "summary": "На 1d после обвала с 124.6k до 62.9k структура указывает на ABC-отскок "
                   "вверх с целями 88-95k, но текущий рост с 66k до 82k уже выглядит как "
                   "зрелая коррекция, и rule-engine на 4h уже сигналит impulse_down с отменой "
                   "выше 75.3k (которая уже нарушена ценой 82.5k — конфликт). 15m даёт "
                   "high-confidence коррекцию вниз с целью 72-74k и отменой выше 79.7k, что "
                   "также уже почти пробито. 5m показывает свежий impulse up в волне 5. "
                   "Картина рассогласованная: макро допускает продолжение вверх к 88-95k, но "
                   "младшие TF перегреты и просят коррекцию — ждём либо отката для лонга, либо "
                   "подтверждения слома структуры.",
        "action": "wait",
        "entry": None,
        "stop_loss": None,
        "target": None,
        "risk_reward": None,
        "invalidation_explained": "Лонг-сценарий ABC вверх отменяется при закрытии 1d ниже 65680. "
                                   "Медвежий сценарий 15m уже под угрозой — отмена выше 79705 "
                                   "фактически достигнута, нужна новая структура.",
        "per_tf_notes": {
            "1d": "Коррекция вверх ABC после impulse down, цели 88-99k, отмена ниже 65.7k. Bias умеренно бычий в рамках большого отскока.",
            "4h": "Rule-engine просит impulse_down с целью 69-57k, но инвалидация 75.3k уже пробита — гипотеза скомпрометирована, скорее продолжение отскока.",
            "15m": "High-confidence correction_down с отменой 79.7k — цена 82.2k уже выше, сценарий де-факто отменён, ждём перерасчёта структуры.",
            "5m": "Импульс вверх в волне 5, локальный перегрев — вход здесь имеет плохой R/R, лучше ждать отката.",
        },
    },

    # ------------------------------------------------------------------
    # HYPE-USD holistic — NEUTRAL/WAIT (conflicting TFs)
    # ------------------------------------------------------------------
    {
        "ticker": "HYPE-USD",
        "type": "holistic",
        "logged_at": "2026-05-13T09:00:00+00:00",
        "current_price": 44.65,
        "bias": "neutral",
        "setup_quality": "low",
        "summary": "На 1d рынок в широком диапазоне 22-58 с признаками коррекции вверх после "
                   "импульса вниз; цена 44.65 уже близко к верху диапазона и зоне сопротивления "
                   "45-48. На 4h rule-engine видит риск разворота вниз (отмена 45.53 совсем "
                   "рядом), хотя структура с 26.23 → 45.30 → 38.89 → 44.69 выглядит как "
                   "потенциальный пятиволновой импульс вверх или ABC. На 15m/5m последний свинг "
                   "— лоу 39.86 (provisional), что может быть как окончанием коррекции для нового "
                   "лонга к 49+, так и началом импульса вниз. Конфликт между TF и "
                   "непосредственная близость к ключевым инвалидаторам делает сетап "
                   "низкокачественным — лучше ждать подтверждения: либо пробой 44.70 с ретестом "
                   "для лонга, либо слом 38.63 для шорта.",
        "action": "wait",
        "entry": None,
        "stop_loss": None,
        "target": None,
        "risk_reward": None,
        "invalidation_explained": "Ждём разрешения: пробой и закрепление выше 45.31 (хай 4h) "
                                   "активирует лонг-сценарий к 49.7; пробой 38.63 вниз подтвердит "
                                   "медвежий сценарий 4h с целью 35.77. До этого торговля у середины "
                                   "диапазона имеет плохой R/R.",
        "per_tf_notes": {
            "1d": "Коррекция вверх после импульса вниз, цена ушла выше типичных целей 38.2-50% — возможно расширенная коррекция к 1.618×A (42.85 уже пройдена), но рядом сопротивление 47-48.",
            "4h": "Rule-engine сигнализирует риск нового импульса вниз с инвалидацией 45.53 — мы в 0.9% от неё, что означает либо немедленный шорт-триггер, либо отмену сценария.",
            "15m": "Структура 26→45→39→45→40 неоднозначна; лоу 39.86 provisional, отмена лонга на 38.63, цели 42.2/49.7 — но требуется подтверждение разворотом.",
            "5m": "Зеркалит 15m: provisional лоу 39.86, нужна импульсная свеча вверх от текущих или слом 38.61 для определённости.",
        },
    },
]


def build_records() -> list[dict]:
    """Convert the BACKFILL list into prediction_log records."""
    out: list[dict] = []
    for entry in BACKFILL:
        pt = pd.Timestamp(entry["logged_at"])
        minute_key = pt.strftime("%Y-%m-%dT%H:%M")
        action = entry["action"]
        entry_price = entry.get("entry")

        if entry["type"] == "per_tf":
            pid = PL._hash_id("pertf", entry["ticker"], entry["timeframe"],
                                action, entry_price, minute_key)
            rec = {
                "id": pid,
                "type": "per_tf",
                "logged_at": entry["logged_at"],
                "ticker": entry["ticker"],
                "timeframe": entry["timeframe"],
                "current_price": entry["current_price"],
                "action": action,
                "entry": entry_price,
                "stop_loss": entry.get("stop_loss"),
                "target": entry.get("target"),
                "agrees_with_top": entry.get("agrees_with_top", False),
                "confidence": entry.get("confidence", "medium"),
                "preferred_pattern": entry.get("preferred_pattern", ""),
                "reasoning": entry.get("reasoning", ""),
                "agrees_with_forecast": entry.get("agrees_with_forecast"),
                "forecast_critique": entry.get("forecast_critique", ""),
                "forecast_pattern": entry.get("forecast_pattern"),
                "forecast_direction": entry.get("forecast_direction"),
                "outcome": None,
                "exit_price": None,
                "exit_time": None,
                "bars_to_resolve": None,
                "r_realized": None,
                "verified_at": None,
                "backfilled": True,
                "backfill_note": "Recovered from chat screenshots — timestamp is approximate.",
            }
        else:  # holistic
            pid = PL._hash_id("holistic", entry["ticker"],
                                action, entry_price, minute_key)
            rec = {
                "id": pid,
                "type": "holistic",
                "logged_at": entry["logged_at"],
                "ticker": entry["ticker"],
                "timeframe": "all",
                "current_price": entry["current_price"],
                "action": action,
                "entry": entry_price,
                "stop_loss": entry.get("stop_loss"),
                "target": entry.get("target"),
                "bias": entry.get("bias", "neutral"),
                "setup_quality": entry.get("setup_quality", "low"),
                "summary": entry.get("summary", ""),
                "risk_reward": entry.get("risk_reward"),
                "invalidation_explained": entry.get("invalidation_explained", ""),
                "per_tf_notes": entry.get("per_tf_notes") or {},
                "outcome": None,
                "exit_price": None,
                "exit_time": None,
                "bars_to_resolve": None,
                "r_realized": None,
                "verified_at": None,
                "backfilled": True,
                "backfill_note": "Recovered from chat screenshots — timestamp is approximate.",
            }
        out.append(rec)
    return out


def main() -> int:
    PL._ensure_dir()
    existing = PL.load_all()
    existing_ids = {p["id"] for p in existing}
    new_records = build_records()

    added = 0
    skipped = 0
    for rec in new_records:
        if rec["id"] in existing_ids:
            skipped += 1
            continue
        PL._append(rec)
        added += 1

    print(f"Backfill complete: {added} added, {skipped} already existed.")
    print(f"Total predictions in journal: {len(PL.load_all())}")
    print(f"File: {PL.PREDICTIONS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
