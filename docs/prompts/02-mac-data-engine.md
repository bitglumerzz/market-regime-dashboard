# Промт 02 — Движок данных на Mac (Apple Silicon)

---

Ты — инженер данных для количественной торговли. Работаем в моём репозитории
`market-regime-dashboard` (ветка `claude/modest-tesla-bcavni`) на моём Mac с Apple Silicon. Это существующий
Python-пайплайн: `data_provider.py`, `data_loader.py`, `cache_5m.py` (данные через ccxt/yfinance),
`hmm_model.py`, `elliott.py`, `claude_rescore.py`, `prediction_log.py`, `backtest.py` и др.
Прочитай `README.md`, `requirements.txt`, `data_provider.py`, `data_loader.py`, `cache_5m.py`, `feature_engineering.py`
и раздел 3.1, 2 и 7 файла `docs/research/regime-ai-stack-2026-09.md` — там выбран стек и обоснование.

**Цель:** надёжный локальный слой данных, который 24/7 собирает рыночные и альтернативные данные в
Parquet + DuckDB и отдаёт модели признаки без заглядывания в будущее.

## Что сделать
1. **Окружение.** `uv` или venv на Python 3.12, закрепи версии из отчёта: `ccxt>=4.5,<5`, `duckdb>=1.5,<2`,
   `pyarrow`, `apscheduler>=3.11,<4`, `arch`, `statsmodels>=0.15`. Не ломай текущие зависимости Streamlit-приложения.
2. **Раскладка хранилища.** `data/lake/{dataset}/exchange=…/symbol=…/tf=…/year=…/month=….parquet` (zstd).
   Модуль `lake.py`: запись с дедупликацией по (ts, symbol), чтение через DuckDB `read_parquet(..., hive_partitioning=1)`,
   готовые вью. Все ряды хранят **время публикации** (когда значение стало известно), а не только время бара.
3. **Архиватор производных — ЗАПУСТИТЬ ПЕРВЫМ ДЕЛОМ** (`collectors/derivatives.py`): funding rate, open interest,
   long/short ratio, taker buy/sell для BTC, ETH, SOL и топ-10 перпетуалов на Binance USDⓈ-M, Bybit V5, OKX V5 (+ Hyperliquid)
   через ccxt, каденс 1h/4h. Причина: эти эндпоинты хранят только ~30 дней истории — каждый день промедления теряется навсегда.
   Соблюдай rate limits из отчёта, экспоненциальный backoff, логирование.
4. **Историческая загрузка** (`backfill_binance_vision.py`): свечи 1m/5m и `metrics` с data.binance.vision,
   `public.bybit.com`, `okx.com/historical-data` (bulk zip, а не пагинация REST), с проверкой checksum.
5. **Альтернативные данные (ежедневные джобы):** Deribit DVOL; Coin Metrics Community (32 бесплатные on-chain метрики,
   лицензия CC BY-NC — только внутренний вход модели); FRED (ставки, DXY), CFTC COT, Fear & Greed; ликвидации через
   Coinalyze (40 req/min, нужен бесплатный ключ — попроси меня вписать в `.env`).
6. **Признаки** (`feature_engineering.py` расширить, не переписывать): funding_z, oi_change, lsr, всплески ликвидаций,
   dvol_z, on-chain z-scores с лагом 1 бар, макро, волатильность GARCH (`arch`). Джойн — только `ASOF JOIN` по времени
   публикации. Тест на отсутствие утечки: признак на баре t не зависит от данных после t.
7. **Планировщик и 24/7.** APScheduler 3.x в одном процессе `engine.py` + `launchd`-агент
   (`~/Library/LaunchAgents/ai.regime.engine.plist`, KeepAlive, логи в `~/Library/Logs/regime/`), `pmset` чтобы Mac
   не засыпал на зарядке, Healthchecks.io пинг от каждой джобы. Инструкция по установке/остановке — в `ENGINE.md`.
8. **Гео-ограничения.** Проверь из моей сети доступность Binance/Bybit/OKX API. Если что-то недоступно —
   предложи вынести только коллектор на VPS из промта 01 (пишет Parquet → rsync на Mac), без ключей бирж на сервере.
9. **Бэкап.** restic для `data/lake` и `predictions/` (в тот же B2, что и бот, отдельный репозиторий).

## Ограничения
- Только публичные эндпоинты, никаких торговых ключей. Не коммить данные и `.env`.
- Не использовать источники с AGPL/non-commercial лицензией как часть продаваемого продукта без пометки в `DATA_SOURCES.md`
  (сделай такую таблицу: источник · лицензия · можно ли в платном продукте).
- Пункты из раздела 8 отчёта «не удалось подтвердить» (Tardis, Santiment) — перепроверь цены/условия перед использованием.

## Готово, когда
- Архиватор производных работает под launchd минимум сутки без пропусков (покажи отчёт о полноте по часам).
- Бэкфилл за 2+ года для BTC/ETH на 5m и 1h загружен; DuckDB-запрос признаков за год выполняется < 5 с.
- Тест на утечку данных проходит; `ENGINE.md` и `DATA_SOURCES.md` написаны; всё закоммичено и запушено.
