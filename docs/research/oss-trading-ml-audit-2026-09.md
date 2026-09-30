# REGIME AI — аудит открытых реализаций ML-торговли (2023–2026)

Дата: 2026-09-30. Метод: клонирование репозиториев (`/tmp/scout/<name>`), чтение кода меток/сплитов/комиссий, лицензий, дат коммитов. Числа звёзд приблизительные (api.github.com заблокирован прокси, брались с веб-страниц GitHub / MCP-поиска). Никаких обещаний прибыли: ниже — только то, что проверено по коду.

---

## 1. Кратко: что появилось с 2023 года и насколько это реально продвинулось

**Что действительно изменилось**

1. **Валидация стала первоклассной библиотекой.** Появились `purgedcv` (2026, MIT, JOSS-статья) и переизданная `RiskLabAI.py` (BSD-3) — интервально-корректный purge/embargo, CPCV с реконструкцией путей, PBO/PSR/DSR/MinTRL, учёт числа испытаний Optuna. То, что мы написали руками в `signal_lab`, теперь есть в виде тестируемых пакетов.
2. **Крупные платформы честно режут окна.** FreqAI (в составе Freqtrade) обрезает фрейм по концу train-окна *до* вычисления таргета; Qlib использует метку `Ref($close,-2)/Ref($close,-1)-1` (пропуск одного бара под исполнение) и `label_leak_n` в RollingGen. Это уровень «взрослой» инженерии, которого в 2023 у хобби-репозиториев не было.
3. **Появились «честные отрицательные результаты».** EdgeProof (15m крипта, triple-barrier + purged CV + DSR + positive control): «направленной edge после taker-комиссии нет, предсказуема только волатильность». trading-agents-lab: LLM-агенты с анонимизированными промптами −0.9% против B&H +15.7%. Это совпадает с нашим выводом 51–55%.
4. **Foundation-модели для рядов (Kronos, Chronos-2, TimesFM-2.5) — не сигнал.** Независимый тест Kronos на 5-мин BTC: Brier 0.189 против 0.188 у броуновского бейзлайна. Три работы 2025–2026 по Chronos/TimesFM: directional accuracy ≈50–51%, OOS R² < 0. Плюс контаминация: претрейн Kronos до 2024-06, а его собственный тест начинается 2024-04.
5. **LLM-агенты (TradingAgents 109k★, ai-hedge-fund 64k★, RD-Agent) — архитектурно интересны, статистически пусты.** Бэктесты без комиссий, hit-rate без Hold, единичный сэмпл LLM, окно теста внутри training data модели. RD-Agent вообще отбирает факторы по метрикам *тестового* сегмента.

**Что не изменилось**

- Паттерн «95% accuracy = утечка» жив: `khovala/moex-ml-trading` (2026, «76.7%») — таргет текущего бара; `DeepAlpha` («84.6%») — сплит по позиции в конкатенации монет; `Hackathon-MOEX-NN-Advisor` (2023, «95.7%») — та же схема, что у Finam-робота.
- Реалистичная граница на 4h–3d по-прежнему 51–55% hit-rate, и edge живёт в медленном тренде + вол-таргетинге, а не в направлении следующего бара. Ни один из 40+ просмотренных репозиториев не показал воспроизводимого OOS-результата, опровергающего это.

**Честная оценка прогресса:** инструменты валидации — большой шаг вперёд; модели предсказания направления — нет. Продвинутые репозитории 2026 года отличаются от 2023 главным образом тем, что *умеют доказать отсутствие edge*.

---

## 2. Разбор WISEPLAT/Hackathon-Finam-NN-Trade-Robot (2023)

**Что заявлено:** CNN на картинках графиков M1 → предсказание направления M10, «accuracy 95.6%».

**Что нашёл аудит:**
- 99.6% меток вычислимы из последних точек входного изображения: close M10 == последняя точка M1 на картинке. Сеть учится читать последний пиксель, а не предсказывать будущее.
- Честный ре-тест на их же данных: hit-rate 53–57%, ожидание **отрицательное** после 5 bps комиссии.
- Тот же автор, тот же механизм в `Hackathon-MOEX-NN-Advisor` (Dec 2023): `3_train_neural_network_for_one_symbol.py:99` метка `d_close = pr_close.diff()`, `:109` `np.where(d_close>0,1,0)`; при этом в фичах `:98` остаётся AlgoPack `pr_change` того же бара; окно `:137-140` `X = rows[i:i+T], y = label[i+T-1]` — последняя строка входа = сам размечаемый бар. Плюс `:159-236` — 20 случайных архитектур, модель сохраняется при росте *test_acc* (отбор на тесте). Без лицензии, без бэктеста, без PnL.

**Что в нём полезно как шаблон (не как модель):**
- Связка «MOEX-данные → брокерский API → live» у этого автора сделана аккуратно: `backtrader_moexalgo` (MIT, 76★) — data feed / live store для AlgoPack с Super Candles как дополнительными линиями; `backtrader_finam` — то же для Finam; `functions_algopack.py` в Advisor — компактный пример пагинации tradestats/orderstats/obstats (`save_metric_to_file :136-172`).
- Оба репозитория заморожены с января 2024, привязаны к moexalgo 2.0 (сейчас 2.5.12, DataFrame-only, ресемплинг, WebSocket) — ожидайте поломок.
- Вывод: **дата-glue у WISEPLAT — ок, ML — систематическая утечка.** Использовать как учебный отрицательный пример в наших leak-check документах.

---

## 3. Таблица проектов

Вердикты: **внедрить** / **идеи** / **следить** / **избегать**.

### 3.1 Production ML/торговые фреймворки

| Проект | Что делает | Лицензия | Свежесть / ★ | Качество валидации, найденные проблемы | Вердикт |
|---|---|---|---|---|---|
| **Freqtrade + FreqAI** | Крипто-бот (CCXT) + ML-слой: скользящее обучение LightGBM/XGB/CatBoost/PyTorch/SB3, дрифт-фильтры (SVM, DBSCAN, DI, PCA), Telegram из коробки | GPL-3.0 | 2026-09-29, ~54.7k | Сплит строго последовательный (`data_kitchen.py:316-380`), фрейм обрезается по `tr_train.stopdt` до `set_freqai_targets` (`freqai_interface.py:348-352`) — утечки меток на границе нет. Но внутренний train/test — хвост окна без purge (`data_kitchen.py:145-156`); есть флаги `shuffle_after_split` (убьют метрику); индикаторы считаются по всему диапазону (`:340-343`) — полносерийная нормализация утекает; нет slippage; нет PBO/DSR | идеи |
| **Microsoft Qlib** | Факторный движок (Alpha158/360), 25+ моделей, rolling retrain, DDG-DA (адаптация к дрифту), портфельный бэктест | MIT | 2026-09-16, ~48k | Метка `Ref($close,-2)/Ref($close,-1)-1` — пропуск бара под исполнение (`handler.py:90,152`); `fit_end_time` = конец train (`processor.py:197-205`); `label_leak_n` в RollingGen (`gen.py:305`); комиссии 0.15%/0.25% (`exchange.py:48-50`). Минусы: один фиксированный test-сплит, крипта только daily Coingecko без OHLC | идеи |
| **Microsoft RD-Agent(Q)** | LLM-цикл гипотеза→код фактора→Qlib-бэктест→фидбек | MIT | 2026-09-23, ~14.8k | **Отбор на тесте**: `feedback.py:17-21` читает IC/return из recorder, чей backtest-диапазон = test (`conf_baseline.yaml:50-52`, test с 2017). Каждая итерация — ещё один запрос к hold-out. Дедуп IC 0.99 не лечит множественное тестирование | идеи (архитектура) |
| **NautilusTrader** | Event-driven движок (Rust), FillModel/LatencyModel/FeeModel, адаптеры Binance/Bybit/OKX/Hyperliquid | LGPL-3.0 | 2026-09-30, ~29k | Не ML. Событийная модель убирает класс look-ahead из векторных бэктестов. Тяжёлая сборка, частые breaking-релизы | следить |
| **vectorbt OSS / PRO** | Numba-векторный бэктестер, широковещательные гриды параметров | Apache-2.0 + Commons Clause; PRO платный | 2026-09 (только docs), ~9.2k | Fees/slippage есть (`portfolio/base.py:257,348`); сплиты без purge (`accessors.py:1774-1886`); движок не защищает от look-ahead. **Commons Clause запрещает продавать сервис на его основе** | избегать |
| **Jesse** | Крипто-бот + research: ML gather/train, bootstrap-значимость правил, Monte Carlo, MCP-сервер | MIT | 2026-09-27, ~8.6k | Сплит хронологический, но одиночный и без purge (`ml.py:99-108`); slippage нет; зато `rule_significance_testing` — корректный стационарный блочный бутстрап (`bootstrap.py:11-30`) | идеи |
| **Hummingbot** | Маркет-мейкинг/грид/DCA, Strategy V2 с `TripleBarrierConfig` + PositionExecutor | Apache-2.0 | 2026-09-22, ~20.3k | Не ML. Бэктест: плоские 2×2 bps (`position_executor_simulator.py:45-49`), без slippage | следить (спецификация исполнения) |
| **OctoBot** | Retail-бот, TradingView webhooks, LLM-коннекторы, оптимизатор | GPL-3.0 | 2026-09-21, ~6.7k | Оптимизатор — чистый in-sample grid search без hold-out | избегать |
| **QuantConnect Lean** | C#/.NET движок, Python-алгоритмы, каталог fee/slippage моделей по брокерам | Apache-2.0 | 2026-09-29, ~21.8k | ML-примеры переобучаются в событийном цикле на `History()` — look-ahead невозможен по построению. Нет purged CV; формат данных — lock-in | следить |
| **backtesting.py / backtrader** | Лёгкие однобумажные бэктестеры | AGPL-3.0 / GPL-3.0 | активен / заморожен 2023 | In-sample optimize(), нет walk-forward. AGPL для сетевого сервиса неприемлема | избегать |

### 3.2 Deep learning / foundation-модели / LLM-агенты

| Проект | Что делает | Лицензия | Свежесть / ★ | Качество валидации, найденные проблемы | Вердикт |
|---|---|---|---|---|---|
| **Kronos** | Decoder-only трансформер на ~12B OHLCV-баров, fine-tune на Qlib CSI300 | MIT | 2026-04-13, 39.7k | Нормализация окна только по lookback (`finetune/dataset.py:104-114`) — хорошо. Но `config.py:33-35`: val 2022-09..2024-06 перекрывает train и test (test с 2024-04); претрейн до 2024-06 → тест внутри претрейна; README:306 признаёт отсутствие slippage. Независимый тест BTC 5m: Brier 0.189 vs 0.188 у случайного блуждания | идеи (распределение как фича) |
| **TradingAgents** | Мультиагентная LLM-«фирма» на LangGraph | Apache-2.0 | 2026-09-29, 109.3k | Point-in-time для фидов реальный (`tools.py:34,58`, `reddit.py:38-47`). «Бэктест» — hit-rate без Hold (`backtest.py:88-90`), доходность без комиссий (`settlement.py:52-74`), один сэмпл LLM, сами пишут «indicative rather than repeatable» (`:123-125`); cutoff-утечка LLM не контролируется | идеи (as_of-паттерн) |
| **trading-agents-lab** | Строгая репликация TradingAgents: анонимизация промптов, 10 bps, LightGBM walk-forward с embargo, 62 теста | MIT | 2026-07-06, 0 | Единственная точка сдвига сигналов (`engine.py:3-5,50`), embargo = max(embargo, horizon) (`ml.py:74-107`). Результат: агенты −0.9% vs B&H +15.7%; LightGBM daily L/S −19% после комиссий | идеи (методология) |
| **TLOB** | Трансформер для LOB (FI-2010, LOBSTER, BTC LOB) | MIT | 2026-02-24, 178 | Порог класса считается по всему размечаемому массиву, включая test (`utils_data.py:127-161` через `btc.py:153,163`); бэктест `commission=0.00005`, `market_fee=0` (`run_backtesting.py:34,246`); горизонт — секунды | избегать |
| **ai-hedge-fund** | LLM-персоны (Баффет, Грэм…), 2026 — ослеплённый бэктестер | MIT | 2026-09-26, 63.8k | Ослепление тикеров/дат честное (`features/snapshot.py:89-102`, тест `:120`). Но `brokers/sim.py:5` — комиссий нет по дизайну; START_DATE 2024-06-01 внутри training data LLM; нет бенчмарка | идеи (blinded snapshot) |
| **FinRL** | DRL (PPO/A2C/SAC) для акций/крипты | MIT | 2026-09-28, 16.5k | Один фиксированный сплит, test == trade (`config.py:10-17`); исполнение по close того же бара; reward без риска; seed-нестабильность | избегать |
| **FinRL Contest 2025 / Task 2** | LSTM+GRU факторы + DQN на 1-сек BTC LOB | не указана | 2025-10-20, 66 | Квантили меток и скейлинг по всей серии (`seq_data.py:1232-1233,1263-1267`); slippage в eval 7e-7 (`task2_eval.py:154`) | избегать |
| **TradeMaster** | RL-платформа (EIIE, SARL, EarnHFT…) | Apache-2.0 | 2025-06-04, 3.1k | Фиксированные сплиты, единичные DRL-кривые, заброшен 15 мес. | следить |
| **AgenticTrading (Open-Finance-Lab)** | Лидерборд LLM-агентов, paper trading | OpenMDW-1.0 | 2026-09-30, 766 | Cutoff данных для адаптера (`adapter.py:348-356`); комиссии в лидерборде не задокументированы; cutoff-память LLM не контролируется | следить (как бенчмарк) |
| **FinGPT / FinRobot** | LoRA-LLM для sentiment; отчёты | MIT / Apache-2.0 | 2026-09, 21.3k / 8.1k | Forecaster обучен на новостях 2023-05..2024-05 — внутри претрейна Llama-3; ни accuracy, ни бэктеста. FinRobot — только SMA-crossover в Backtrader | избегать |
| **Chronos-2 / TimesFM-2.5** | Zero-shot TS foundation | Apache-2.0 | 2026 | По трём работам 2025–2026: DA ≈51% / <50%, OOS R² <0; контаминация претрейна завышает accuracy на 47–184% | избегать (кроме вол-прогноза) |

### 3.3 MOEX и русскоязычные проекты

| Проект | Что делает | Лицензия | Свежесть / ★ | Качество валидации, найденные проблемы | Вердикт |
|---|---|---|---|---|---|
| **khovala/moex-ml-trading** | «End-to-end ML платформа для MOEX» (T-Invest, LightGBM/XGB + Chronos/TimesFM, Airflow/MLflow), заявка «76.7% DA, PF 2.0» | MIT | 2026-09-14, 0 | **Утечка подтверждена**: таргет `return_1` = close_t/close_{t-1}-1 (`engineering.py:41-42`), `model.fit(..., target_key="return_1")` (`model_training.py:91`), фичи rsi/macd/zscore считаются на индексе t (`engineering.py:45-77`). Модели без текущего close дают 49.85%. Собственный 90-дневный бэктест автора: все ML-стратегии в минусе (`pipeline90_final_description.md:404-438`) | избегать |
| **WISEPLAT/Hackathon-MOEX-NN-Advisor** | LSTM на AlgoPack 5m, «95.7%» | нет LICENSE | 2023-12-15, 39 | Утечка (см. §2): same-bar `pr_change` vs same-bar `d_close`; отбор на test | избегать |
| **antal679-jpg/MoexScalp** (fork bad3p/DeepScalp) | VQ-VAE + LSTM на стакане T-Invest | MIT | 2026-09-17, 0 | Таргет честно вперёд (`TkPreprocessTimeSeriesData.py:166-190`), сплит по сессиям. Но: нет бэктеста, комиссий, стопов, цифр; Windows+CUDA; train/serve mismatch 20 vs 19 фич | следить |
| **moexalgo/moexalgo** | Официальный клиент AlgoPack: свечи, tradestats/orderstats/obstats, FutOI, HI2, WebSocket | Apache-2.0 | 2026-09-02 (v2.5.12), 151 | Только данные, нечему течь. Полная история — платная подписка; терms запрещают редистрибуцию сырых данных (сигналы — можно) | **внедрить** |
| **cia76/FinamPy, TinvestPy, AlorPy** (+FinLabPy) | Тонкие обёртки над Finam Trade API v2, T-Invest gRPC, Alor | MIT (FinLabPy — без лицензии) | FinamPy 2026-05-12, TinvestPy 2026-09-04, FinLabPy 2026-09-26 | Коннекторы, ML нет. FinamPy зафиксирован на Trade API 2.13 при актуальной 2.19; один мейнтейнер | **внедрить** (TinvestPy/AlorPy), FinLabPy — нет |
| **FinamWeb/finam-trade-api** | Официальный SDK Finam (Python `finam-sdk`, Node, proto, `.ai/skills`) | MIT | 2026-09-23 (2.19.1), 24 | Примеры стратегий без комиссий (grep пуст); ничего не заявляют | **внедрить** |
| **WISEPLAT/backtrader_moexalgo, backtrader_finam** | Backtrader feed/store для AlgoPack и Finam | MIT | 2024-01-02, 76 | Устарел относительно moexalgo 2.5 | идеи |
| **AlexWan/OsEngine** | C#/.NET платформа, 60+ коннекторов, оптимизатор с WFO | **Проприетарный EULA** (VAN Technologies) | 2026-09-29, 1031 | ML нет; §4.1(4)-(5) запрещает распространение производных, §5.3 — производные становятся их собственностью | избегать |
| **RussianInvestments/invest-python** | Официальный T-Invest gRPC клиент | Apache-2.0 | 2025-10-09, **archived**, 137 | Клиент. Новая разработка — под T-Bank namespace | следить |
| **TheBolotnik/MOEX-trading-bot** | README-спека «regime → RS → setup → Telegram», 3–20 дней | нет | 2026-09-28, 0 | Кода нет вообще | следить |
| **notryheart/moex-algopack-mcp**, moexalgo/algopack-skills | MCP-сервер и skill-файлы AlgoPack для Claude | Apache-2.0 | 2026-06, 1 / 3 | Данные, ML нет | идеи |

### 3.4 Крипто ML-сигналы и validation-first

| Проект | Что делает | Лицензия | Свежесть / ★ | Качество валидации, найденные проблемы | Вердикт |
|---|---|---|---|---|---|
| **purgedcv** (eslazarev) | sklearn-совместимый purge/embargo, WalkForward, PurgedKFold, CPCV + пути, PBO/PSR/DSR/MinTRL, Optuna-рекордер | MIT | 2026-09-27 (v0.1.8), 35 | Purge интервальный по [pred_time, eval_time) (`_purge.py:19-80`); DSR по Bailey/LdP (`_metrics.py:197`). Крипто-пример честно показывает, что оба сплита теряют деньги | **внедрить** |
| **EdgeProof** (intikhab49) | 15m BTC/ETH/SOL: ATR triple-barrier, purged k-fold, LightGBM/LSTM, DSR по числу порогов, positive control | **нет LICENSE** | 2026-09-25, 1 | Triple-barrier по high/low с обработкой двойного касания (`ml_pipeline.py:368-417`); purge `[i,t1]` vs `[test_start, test_end+embargo]` (`:432-459`); `pnl = dir*ret − 2*fee` (`:492`). Мало данных (5000 свечей). Вывод авторов: edge нет, DSR 0.02–0.28 | идеи (не копировать код) |
| **RiskLabAI.py** | Полная реализация AFML/MLAM: бары, CUSUM, triple-barrier, meta-labeling, uniqueness weights, PurgedKFold, CPCV, PBO/DSR, bet sizing, HRP | BSD-3-Clause | 2026-08-26, 5 (PyPI с 2023, 86 тестов) | PurgedKFold корректен (`purged_kfold.py:28-135`); triple_barrier по close, а не high/low (`labeling.py:193`) — оптимистично по стопам | **внедрить** выборочно |
| **mlfinpy** | Форк последней открытой mlfinlab | MIT | 2025-01-23 (заброшен), 84 | PurgedKFold корректен (`cross_validation.py:80-137`); барьеры по close; нет DSR/PBO | идеи |
| **skfolio** | Портфельная оптимизация на sklearn, CombinatorialPurgedCV | BSD-3-Clause | 2026-09-29, 2453 | Purge позиционный, без знания t1 (`_combinatorial.py:366-377`) — с triple-barrier нужно ставить `purged_size ≥ max(t1−t0)` вручную | следить (портфельный слой) |
| **DeepAlpha** | CCXT-бот, LightGBM 1h, «84.6% walk-forward» | MIT (модели платные) | 2026-05-12, 47 | **Утечка**: `np.concatenate` по монетам, затем сплит 70/15/15 по позиции (`train.py:50-167`) — «test» = последние монеты списка за весь период, train содержит BTC/ETH за те же даты. Метки 3-бар overlapping без purge | избегать |
| **Azalyst Alpha Research Engine** | Cross-sectional 5m, 444 пар, XGBoost, «+564.7%, Sharpe 3.29, DSR 0.9999» | badge MIT, **LICENSE нет** | 2026-09-28, 3 | README сам описывает 4 итерации правил на том же «OOS» окне Sep-2024..Mar-2026; отбор фич по IC текущей OOS-недели (`engine:2044-2062`); шорт неликвидов без фильтра объёма и slippage; leak-тест переписан в `corr(y, y+noise)` (`azalyst_leak_test.py:44-58`) | избегать |
| **NexaQuant** | Gold+BTC H1/H4, SMC + HMM gate + triple-barrier meta-labeling, CPCV/DSR | **нет LICENSE** | 2026-09-30, 0 | Исполнение по next open с shift(1) (`engine.py:18-31`) — честно; но 154 метки / 43 OOS, один режим золота; AUC мета-модели 0.47–0.51 (нет skill) | избегать |
| **Numerai Crypto example** | LightGBM на обфусцированных фичах, target 20d | MIT | 2026-09-28, 1187 | Позиционный 70/30 без embargo на 20-дневный таргет; фичи не аудируемы | следить (внешний scoreboard) |
| **Zarattini «Catching Crypto Trends» / Quant_Strategies docs** | Спецификация Donchian-ансамбля (9 lookback, ratchet-стоп, 25% vol target, 10 bps) | нет LICENSE (docs); SSRN 5209907 | 2026-09-29 | Кода нет; документ перечисляет незаданные в статье параметры | идеи |
| **PNarsis/bitcoin-direction-volatility-ml** | Daily BTC green/red, XGB/CatBoost + кросс-активы | нет LICENSE | 2026-09-27, 0 | Утечек нет (все фичи shift(1)); без комиссий; печатает majority-baseline | избегать (учебный) |

---

## 4. Что конкретно взять в REGIME AI и куда встроить

Лицензионный фильтр для платного сервиса: **можно копировать код** — MIT / Apache-2.0 / BSD-3; **можно линковать без модификации** — LGPL (Nautilus); **только идеи, переписать самим** — GPL/AGPL (Freqtrade, OctoBot, backtesting.py), Commons Clause (vectorbt), репозитории без LICENSE (EdgeProof, Azalyst, NexaQuant, FinLabPy, Quant_Strategies).

### 4.1 `signal_lab/` — валидация и метки

| Что | Откуда (лицензия) | Куда | Действие |
|---|---|---|---|
| Интервальный purge/embargo, `WalkForwardSplit`, `PurgedKFold`, `CombinatorialPurgedCV.reconstruct_paths`, `deflated_sharpe_ratio_full`, `effective_n_trials`, `TrialSharpeRecorder`, `audit_splitter` | purgedcv (MIT) | `signal_lab/cv.py`, `signal_lab/stats.py` | Заменить ручной purged walk-forward; подавать triple-barrier `t1` как `evaluation_times`; считать каждый Optuna/threshold-trial в DSR; `audit_splitter` в тестах — assert нулевого пересечения. Кросс-чек наших PBO/DSR/MinTRL против их реализации |
| Uniqueness + time-decay sample weights, `bet_sizing` (AFML гл.10), `cusum_filter`, `backtest_overfitting_simulation`, `multiple_testing` | RiskLabAI.py (BSD-3) | `signal_lab/v1.py` (sample_weight в LightGBM), `signal_lab/sizing.py`, `signal_lab/stats.py` | Вендорить нужные модули с пином версии |
| Соглашение meta-label `get_events(side_prediction) → get_bins` | mlfinpy (MIT) | `signal_lab/labels.py` | Сверить нашу схему с AFML-эталоном; не зависеть от пакета |
| Метка «пропуск одного бара»: `Ref(close,-2)/Ref(close,-1)-1` | Qlib (MIT) | `signal_lab/labels.py` | Для 4h: доходность от открытия/закрытия *следующего* бара, не от текущего close |
| `label_leak_n` / horizon-purge в RollingGen (`gen.py:305-345`) | Qlib (MIT) | `signal_lab/cv.py` | Как второй независимый чек нашего purge |
| `learn_processors` vs `infer_processors` с `fit_start/fit_end` | Qlib (MIT) | `signal_lab/features.py` | Единый паттерн для stateful-трансформов (скейлеры, PCA) |
| Positive control: подмешать `0.55*label + noise`, требовать DSR ≥ 0.95 | EdgeProof (без лицензии → переписать) | `signal_lab/tests/test_pipeline_sanity.py` | CI-gate: если пайплайн не ловит подсаженную утечку — пайплайн сломан |
| Обработка «двойного касания» (high и low пробивают барьеры в одной 4h-свече): drop или считать как loss | EdgeProof (идея) | `signal_lab/barrier_eval.py` | Проверить, что делаем сейчас |
| Стационарный блочный бутстрап значимости правила, Monte Carlo перестановки сделок | Jesse (MIT) | `signal_lab/placebo.py`, `signal_lab/drawdown.py` | Дополняет PBO/DSR (они про selection bias, это — про одиночное правило при серийной зависимости); распределение просадок → строка «ожидаемая просадка» в Telegram |
| Единственная точка сдвига сигналов + perturbation-тесты no-lookahead | trading-agents-lab (MIT) | `signal_lab/backtest.py`, `tests/test_engine.py` | Правило: `position = signal.shift(1)` ровно в одном месте; тест — возмущение будущих баров не меняет прошлых сигналов |
| Чеклист параметров Donchian-ансамбля Zarattini (closes, ratchet mid-stop, vol-scaled веса усредняются, ребаланс только по vol-изменениям, 10 bps) | Quant_Strategies docs (только идеи) + SSRN 5209907 | `signal_lab/strategies/donchian_ensemble.py` | Сделать нашу реализацию точной репликой, зафиксировать сделанные выборы, прогнать через purged/placebo |
| Kill-criteria дрифта: OOS IC > 0, Jaccard-стабильность фич > 0.5, ML должен бить наивный бейзлайн | Azalyst (идея) | `signal_lab/monitor.py` | Каждый retrain: LightGBM vs Donchian; алерт при падении 4-недельного hit-rate/IC к нулю |
| Kronos multi-path прогноз как *распределительная* фича (квантильный разброс на 4h–3d) | Kronos (MIT) | `signal_lab/features/kronos_dist.py` | Только как вход в meta-labeling, оценивать строго на данных после 2024-06 |
| Chronos-2 квантили realized vol | chronos-forecasting (Apache-2.0) | `signal_lab/vol_target.py` | Единственное место, где TSFM показывают пользу; тоже только post-cutoff |

### 4.2 `bot/` — Telegram-сервис

| Что | Откуда | Куда | Действие |
|---|---|---|---|
| Спецификация сигнала = `TripleBarrierConfig` (stop_loss, take_profit, time_limit, trailing, order types) | Hummingbot (Apache-2.0) | `bot/schemas/signal.py`, HMAC signal API | Пользователь сможет исполнить сигнал дословно; при авто-исполнении — Hummingbot-контроллер читает наш API |
| Settle-after-holding vs benchmark | TradingAgents (Apache-2.0) | `bot/scoreboard.py` | К hit/expectancy добавить alpha относительно BTC buy&hold за горизонт |
| Адаптивные пороги: Gaussian по последним N предсказаниям, сигнал только за mean ± k·std (`fit_live_predictions`) | FreqAI (GPL → переписать) | `bot/signal_gate.py` | Вместо фиксированного порога вероятности |
| Dissimilarity Index: расстояние live-фич до train-множества как abstain-gate | FreqAI/datasieve (GPL → переписать идею) | `bot/signal_gate.py` | Не публиковать сигнал, если рынок «не похож» на обучающую выборку |
| Bet sizing по вероятности (AFML) | RiskLabAI (BSD-3) | `bot/formatter.py` | Размер позиции в сообщении из калиброванной вероятности; размер фиксировать при входе (NexaQuant: пересчёт по барам жрёт комиссии) |

### 4.3 Движок данных / retrain

| Что | Откуда | Куда | Действие |
|---|---|---|---|
| Словарь конфига `train_period_days / live_retrain_hours / expiration_hours` | FreqAI (только термины) | `data_engine/retrain_scheduler.py` | |
| Retrain только на барах с `close_time < now` (паттерн «train внутри event loop на History()») | Lean (Apache-2.0) | `data_engine/retrain_scheduler.py` | Гарантия отсутствия look-ahead при переобучении |
| `as_of()/as_of_window()` клампы для новостей/on-chain | TradingAgents (Apache-2.0) | `data_engine/feeds/` | Point-in-time для всех несвечных источников |
| Числа комиссий/slippage по площадкам (`Binance*FeeModel.cs`, `MarketImpactSlippageModel.cs`) | Lean (Apache-2.0) | `data_engine/costs.yaml` | Таблица допущений по venue |
| Embargo/overlap assertion per ticker (`guards.py:41-63`), champion.json с причинами промоушена | khovala (MIT) — единственное полезное | `data_engine/leakage_guards.py`, `models/registry/` | Мелкие, но аккуратные паттерны |

### 4.4 Claude-as-judge

| Что | Откуда | Действие |
|---|---|---|
| Blinded snapshot: периоды `t-0…t-n`, без дат и тикеров, цены rebased к 100, объём в % от среднего | ai-hedge-fund (MIT), trading-agents-lab (MIT) | Все входы судьи анонимизировать; «delta между именованным и анонимным промптом = утечка памяти LLM» — как регулярный probe |
| Bull/Bear debate-структура | TradingAgents (Apache-2.0) | Шаблон промпта судьи; оценивать только post-cutoff и forward |
| Цикл гипотеза→код→бэктест→фидбек, CoSTEER-промпты для написания pandas-фич с юнит-проверками | RD-Agent (MIT) | Если автоматизируем поиск факторов: судье — метрики **валидационного** окна, тест — заперт, каждая итерация списывается в PBO/DSR |
| Paper-trading как forward test после cutoff | AgenticTrading (OpenMDW) | Делать у себя в боте; лидерборд — для внешнего сравнения |

---

## 5. Чего избегать и типовые ловушки, найденные в коде

1. **Таргет того же бара.** `khovala`: `return_1 = close_t/close_{t-1}-1` при фичах на индексе t. `WISEPLAT` (оба): `d_close = diff()` + `pr_change` того же бара, окно заканчивается на размечаемом баре. Чек: `assert label_time > max(feature_time)` для каждой строки.
2. **Сплит по позиции после конкатенации активов.** `DeepAlpha train.py:50-167`: «test» = последние монеты списка за весь период. Чек: сплит только по времени и одновременно для всех активов.
3. **Порог/квантили меток и скейлинг по всей серии, включая test.** TLOB `utils_data.py:127-161`, FinRL Contest `seq_data.py:1232-1233,1263-1267`. Чек: любой `np.quantile/mean/std` для меток и нормализации — только на train.
4. **Отбор на тесте.** RD-Agent (LLM-фидбек по тестовому сегменту), Advisor (сохранение модели по `test_acc`), Azalyst (4 итерации правил на «OOS» окне), Kronos (val/test overlap `config.py:33-35`). Чек: заперт test-окно, все итерации — в счётчик DSR.
5. **Overlapping-метки без purge/embargo.** Jesse `ml.py:99-108`, FreqAI внутренний сплит `data_kitchen.py:145-156`, Numerai example (20d target, 0 embargo), DeepAlpha (3 бара). Чек: purge ≥ max(t1−t0), embargo после test.
6. **Комиссии и slippage.** Полное отсутствие: ai-hedge-fund (`sim.py:5`), TradingAgents, FinamWeb-примеры, PNarsis. Нереалистично малые: TLOB 0.5 bps + `market_fee=0`, FinRL Contest `slippage=7e-7`, Hummingbot плоские 2 bps. Чек: taker-fee площадки ×2 + slippage ≥ 5 bps; отчёт с/без комиссий.
7. **Претрейн/knowledge cutoff.** Kronos тест 2024-04 при претрейне до 2024-06; FinGPT-Forecaster внутри Llama-3; ai-hedge-fund START_DATE 2024-06-01; TradingAgents paper — Jan–Mar 2024 внутри GPT-4o. Чек: для LLM/TSFM — только post-cutoff + анонимизация.
8. **Исполнение по close наблюдаемого бара.** FinRL `env_stocktrading.py` step; vectorbt по умолчанию. Чек: fill на next open, `shift(1)` в одном месте.
9. **Нереализуемый универсум.** Azalyst: шорт 400+ пар без фильтра ликвидности, топ-фичи — меры неликвидности, 100% оборот еженедельно. Чек: фильтр объёма/спреда, стоимость шорта.
10. **«Leak-test», который ничего не тестирует.** Azalyst `azalyst_leak_test.py:44-58` коррелирует y с y+noise. Чек: positive control должен *подсаживать утечку в фичи* и требовать её обнаружения.
11. **Статистически пустые выборки.** NexaQuant 43 OOS-сделки, EdgeProof 52 дня 15m, trading-agents-lab 64 дня/1 тикер. Чек: MinTRL/MinBTL перед выводами.
12. **Лицензии.** GPL (Freqtrade, OctoBot, backtrader), AGPL (backtesting.py), Commons Clause (vectorbt), проприетарный EULA под названием «Open Source» (OsEngine), «MIT badge без LICENSE» (Azalyst), полное отсутствие LICENSE (EdgeProof, NexaQuant, FinLabPy, WISEPLAT Advisor). Правило: копировать только из MIT/Apache/BSD, при отсутствии файла LICENSE — только идеи.

---

## 6. Если позже добавим MOEX — рекомендуемый стек

**Данные**
- `moexalgo/moexalgo` (Apache-2.0, v2.5.12, 2026-09) — свечи 1m…1d, Super Candles (tradestats/orderstats/obstats: imbalance_vol, put/cancel, спреды), FutOI по типам клиентов, HI2. Бесплатный ISS-уровень — обычные свечи; полная история и realtime — платная подписка AlgoPack (~2–3k ₽/мес). Для горизонта 4h–3d нужны дневные/5m агрегаты как regime-фичи, не интрадей-поток. Учесть: сертификат Минцифры (2.5.11 ставит автоматически), гео-блокировки с иностранных хостов, запрет редистрибуции сырых данных (производные сигналы — можно).
- Загрузчик: 20-строчный `moexalgo → parquet` рядом с текущим ccxt-лоадером, не Backtrader.
- Для Claude-судьи: `moexalgo/algopack-skills` или `notryheart/moex-algopack-mcp` (Apache-2.0) — контекст FutOI/imbalance по тикеру в момент решения.

**Брокерский API (данные/котировки, при необходимости — исполнение)**
- **Finam:** `FinamWeb/finam-trade-api` (MIT, официальный, 2.19.1, proto + Python `finam-sdk` + `.ai/skills` для Claude Code) — предпочтительнее `cia76/FinamPy` (MIT, но зафиксирован на 2.13).
- **T-Invest:** `cia76/TinvestPy` (MIT, 2026-09); официальный `invest-python` помечен archived — проверить новый namespace T-Bank перед зависимостью. Sandbox T-Invest — самый простой бесплатный paper-trading в РФ.
- **Alor:** `cia76/AlorPy` (MIT).
- Единый стиль вызовов у `cia76` снижает стоимость поддержки трёх брокеров; риск — bus factor одного мейнтейнера.

**Чего не брать:** OsEngine (EULA), FinLabPy (нет лицензии), backtrader_moexalgo/finam (заморожены, GPL Backtrader), любые ML-части WISEPLAT/khovala.

**Метод:** тот же `signal_lab` (purged WFO, triple-barrier, PBO/DSR, positive control), комиссии брокера + биржи + проскальзывание; отдельно учесть лимит-ап/даун и планки (по образцу `limit_threshold` в Qlib).

---

## 7. Источники

Клонировано и прочитано (`/tmp/scout/*`): Freqtrade, Qlib, RD-Agent, NautilusTrader, vectorbt, Jesse, Hummingbot, OctoBot, Lean, Kronos, TradingAgents, trading-agents-lab, TLOB, ai-hedge-fund, FinRL, FinRL_Contest_2025, TradeMaster, AgenticTrading, FinGPT, FinRobot, purgedcv, EdgeProof, RiskLabAI.py, mlfinpy, skfolio, DeepAlpha, Azalyst, NexaQuant, Numerai example-scripts, PNarsis, khovala/moex-ml-trading, Hackathon-MOEX-NN-Advisor, MoexScalp, moexalgo, FinamPy/TinvestPy/AlorPy/FinLabPy, finam-trade-api, backtrader_moexalgo, OsEngine, invest-python, MOEX-trading-bot, moex-algopack-mcp.

- https://github.com/WISEPLAT/Hackathon-Finam-NN-Trade-Robot · https://github.com/WISEPLAT/Hackathon-MOEX-NN-Advisor · https://github.com/WISEPLAT/backtrader_moexalgo
- https://github.com/freqtrade/freqtrade · https://github.com/microsoft/qlib · https://github.com/microsoft/RD-Agent (arXiv 2505.15155) · https://github.com/nautechsystems/nautilus_trader · https://github.com/polakowo/vectorbt · https://github.com/jesse-ai/jesse · https://github.com/hummingbot/hummingbot · https://github.com/Drakkar-Software/OctoBot · https://github.com/QuantConnect/Lean · https://github.com/kernc/backtesting.py
- https://github.com/shiyu-coder/Kronos · https://github.com/TauricResearch/TradingAgents · https://github.com/Kantamaniprakash/trading-agents-lab · https://github.com/LeonardoBerti00/TLOB · https://github.com/virattt/ai-hedge-fund · https://github.com/AI4Finance-Foundation/FinRL · https://github.com/Open-Finance-Lab/FinRL_Contest_2025 · https://github.com/TradeMaster-NTU/TradeMaster · https://github.com/Open-Finance-Lab/AgenticTrading · https://github.com/AI4Finance-Foundation/FinGPT · https://github.com/AI4Finance-Foundation/FinRobot · https://github.com/amazon-science/chronos-forecasting · https://github.com/google-research/timesfm (arXiv 2606.27100, 2511.18578, 2609.20554)
- https://github.com/eslazarev/purged-cross-validation · https://github.com/intikhab49/edgeproof-crypto-trading-backtest · https://github.com/RiskLabAI/RiskLabAI.py · https://github.com/baobach/mlfinpy · https://github.com/skfolio/skfolio · https://github.com/stefanoviana/deepalpha · https://github.com/gitdhirajsv/Azalyst-Alpha-Research-Engine · https://github.com/gujja330/NexaQuant · https://github.com/numerai/example-scripts/tree/master/crypto · https://github.com/alfred1123/Quant_Strategies/blob/main/docs/research/catching-crypto-trends.md (SSRN 5209907) · https://github.com/PNarsis/bitcoin-direction-volatility-ml
- https://github.com/khovala/moex-ml-trading · https://github.com/antal679-jpg/MoexScalp · https://github.com/moexalgo/moexalgo · https://github.com/cia76/FinamPy · https://github.com/FinamWeb/finam-trade-api · https://github.com/AlexWan/OsEngine · https://github.com/RussianInvestments/invest-python · https://github.com/TheBolotnik/MOEX-trading-bot · https://github.com/notryheart/moex-algopack-mcp

**Ограничения аудита:** api.github.com, arxiv.org, huggingface.co, gresearch.com, kaggle.com заблокированы прокси — числа из статей взяты из поисковых сниппетов, звёзды — с веб-страниц; `zcakhaa/DeepLOB`, `cia76/BackTraderFinam`, `RoboMOEX` не клонировались (недоступны); решения G-Research Crypto не проверены; ни одна модель не запускалась — только чтение кода; backtesting.py/backtrader оценены по документации.