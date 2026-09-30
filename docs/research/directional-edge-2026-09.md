# Предсказуемость направления криптоактивов: сводный отчёт по проверенным предикторам (v2, 30.09.2026)

**Статус верификации (важно читать первым).** В этом раунде рефери смогли открыть только github.com. Все академические и вендорские хосты (arXiv и зеркала, SSRN, NBER, Wiley, ScienceDirect, Springer, OUP, Cambridge, BIS, INFORMS, Emerald, Kaggle, Quantpedia, Zenodo, Deribit, Binance, CFTC, Farside, alternative.me, Medium, TheBlock, CoinDesk) были заблокированы egress-прокси, а бюджет WebSearch исчерпан (200/200). Поэтому:

- Подтверждены первоисточником только цифры с GitHub (помечены **[подтверждено]**).
- Всё остальное — пересказ абстрактов/сниппетов, сверенный с памятью о статьях; помечено **[не проверено]**. Ни одно такое число нельзя зашивать в модель как prior, пока PDF не прочитан с машины с открытым интернетом.
- Три препринта arXiv от июня–августа 2026 (2606.00060, 2607.09426, 2608.21888) находятся после knowledge cutoff моделей-рефери; их существование и цифры не подтверждены вообще.
- Отчёт не обещает доходности. Все эффекты — верхняя граница; по McLean–Pontiff ожидайте примерно половину опубликованного эффекта после публикации и меньше после издержек.

---

## 1. Честный ответ: можно ли предсказывать направление и насколько

**Коротко: да, слабо, на горизонте от нескольких часов до нескольких дней, и почти вся экономическая ценность приходится не на угадывание знака, а на управление размером позиции и просадками.**

Реалистичные ожидания по горизонтам (для BTC/ETH, после издержек ≈ 5–11 б.п. round-trip taker на Binance USDT-M):

| Горизонт | Реалистичная точность знака | Что реально работает | Вердикт для Telegram-сервиса с задержкой в минуты |
|---|---|---|---|
| Тики – 1 мин | 60–80 % на тестах (DeepLOB, OFI, queue imbalance) — эффект реальный и рецензированный на акциях | Микроструктура: OFI, дисбаланс стакана | **Непригодно.** Период полураспада — секунды; при задержке в минуты получаете adverse selection, т.е. ожидаемо отрицательный результат |
| 15 мин – 1 ч | ~52–53 % (заявлено; 2–3 AUC-пункта) [не проверено] | 15-мин sign reversal, quarter-hour taker imbalance, hourly ML | **Ниже издержек.** Заявленный edge 5–17 б.п. против 5–11 б.п. round-trip; препринты 2026, репликаций нет |
| **4 ч – 3 дня** | **51–55 %** на BTC/ETH от трендовых признаков (рабочая гипотеза, измерять самим) | Медленный тренд/breakout + vol targeting; дневной taker/order-flow imbalance (Sharpe LS 1.9–3.6 заявлено, не проверено); режимные фильтры | **Целевой горизонт сервиса.** Edge маленький, но оборот низкий, издержки терпимы, abstain-правило делает его пригодным |
| 1–4 недели | Качественно: TSMOM и MA-фильтры на BTC работают in-sample (Liu–Tsyvinski RFS 2021; Detzel et al. FM 2021); post-2022 OOS с издержками — нет рецензированных реплик | Тренд, vol targeting | Хорошо как режимный слой; знак фрагилен во флэте (2022-тип) |
| Месяцы – циклы | Не измеримо: ~3–4 независимых цикла | MVRV-Z, NUPL, 12m TSMOM | Только контекст; пороги переобучены по построению |

Три вещи, которые точно переживают скептическую проверку:

1. **Volatility targeting / vol-managed sizing** — единственный компонент со strong, многодесятилетней, многоклассовой рецензированной поддержкой (Kim–Tse–Wald JFM 2016; Moreira–Muir JF 2017; Barroso–Santa-Clara JFE 2015). Улучшает Sharpe и просадки, **не** повышает точность знака.
2. **Методология валидации** — purged/embargoed CPCV + DSR + PBO. **[подтверждено]**: на непредсказуемой цели наивный shuffled KFold даёт R² 0.83 (k-NN) / 0.91 (RandomForest), PurgedKFold — −1.48 / −1.87; перекрытие меток train/test 100 % → 0 % (github.com/eslazarev/purged-cross-validation). Иными словами, большинство «60–90 % accuracy» в крипто-ML — артефакт утечки.
3. **Медленный многооконный тренд на ликвидных монетах** (Zarattini et al. 2025 Donchian-ансамбль): две независимые GitHub-реплики **[подтверждено]** воспроизводят BTC CAGR 30 %, vol 17 %, Sharpe 1.56, MDD 19 % net of 10 б.п. — и одновременно подтверждают внутренние нестыковки статьи (MAR 0.88 vs 1.15; Sortino 2.03 vs 1.23) и незаявленные параметры. Оценка moderate-minus: единый in-sample бэктест 2015–2025 без holdout.

Всё, что заявляет >55 % на дневном горизонте или Sharpe >3 на альтах, в этом раунде либо не подтверждено, либо имеет очевидный дефект (survivorship, overlapping windows, отсутствие издержек, post-hoc селекция конфигурации).

---

## 2. Таблица предикторов

Сила: **strong / moderate / weak / no-evidence**. «Эффект» — как заявлено в источнике; статус верификации в скобках. Издержки — round-trip Binance USDT-M taker ≈ 10–11 б.п., maker ≈ 4–5 б.п. (docs проекта: taker 0.05 %, maker 0.02 %).

### 2.1. Ценовые: тренд, momentum, reversal, сезонность

| Фактор | Формула | Горизонт | Сила | Эффект (статус) | Данные | Статус после публикации / издержек |
|---|---|---|---|---|---|---|
| Donchian breakout ансамбль + vol targeting (long-only) | L∈{5,10,20,30,60,90,150,250,360}d: long если close > max(close,L); trailing stop = max(prev stop,(chmax+chmin)/2); позиция = среднее по L; size = 0.25/σ_90d, cap 1x, порог ребаланса 20 % | дни–месяцы | **moderate(−)** | BTC net 10 б.п.: CAGR 30 %, vol 17 %, Sharpe 1.56, MDD 19 %; ETH 27 %/16 %/1.51/15 %; top-20 ротация Sharpe 1.58, alpha +14 %/г vs BTC **[подтверждено через 2 GitHub-реплики]**; MAR 0.88 vs 1.15, Sortino 2.03 vs 1.23 — нестыковки в самой статье | Daily OHLC BTC/ETH/top-20 via ccxt; listing dates для ротации | Опубликовано 04.2025, post-publication неизвестен; in-sample, без holdout; [UNSTATED] универсум и stop-механика — ждите цифры ниже при пересборке |
| TSMOM 1–4 недели (BTC/ETH/индекс) | sign(r_{t−k,t}), k=1..4 нед.; w = sign·σ_target/σ̂_t; либо EWMA fast−slow / vol | 1–4 нед. | **weak(→moderate)** | Liu–Tsyvinski RFS 2021: недельная доходность рынка значимо предсказывает 1–4 нед. вперёд (in-sample 2011–2018, рецензировано); post-2020 Sharpe 0.82→1.22, 31.96 % годовых, 0.74 vs 0.79 — все из нерецензированных источников [не проверено] | Daily OHLCV via ccxt; VW индекс CoinGecko | Низкий оборот, 5–10 б.п. терпимо; во флэте (2022) flat-to-negative; robust-компонент — снижение просадок от vol scaling, не знак |
| Price-to-MA ratio / VMA (BTC) | 1{P_t > MA_L}, L=1–20 нед.; регрессия r_{t+1} = a + b·log(P_t/MA_L); VMA: MA(1–5d) > MA(50–200d) | 1 день – недели | **weak** | Detzel et al. FM 2021: предсказуемость in- и out-of-sample, рост Sharpe, меньше просадки (рецензировано); «11–58×» и «+0.2–0.6 Sharpe» [не проверено]. Hudson & Urquhart — Annals of OR 2021, ~15 000 правил (в прошлом драфте ошибочно FRL 2019/2 127 правил) | Daily close via ccxt | Короткие MA чувствительны к 10+ б.п.; Sullivan–Timmermann–White 1999: «лучшее из многих» правило не повторяется OOS; использовать log(P/MA_L) как непрерывные признаки, не выбирать одно |
| Cross-sectional momentum монет (1–4 нед. формирование, 1 нед. удержание) | Еженедельный ранг по r_{t−k,t}; long top / short bottom квинтиль (Liu–Tsyvinski–Wu JF 2022, cap >$1M, VW); CTREND — rolling cross-sectional регрессия на MA-индикаторы | 1 нед. | **weak** | JF 2022: значимый фактор in-sample 2014–2020 (1 827 монет); «~3 %/нед.», «11.2 %/нед.», Sharpe 0.48, CTREND 3.87 %/нед., Fieberg −255 % крах — [не проверено]. GitHub prams2104, 20 б.п.: Sharpe 0.42 (train) → −0.48 (val) → −2.18 (OOS 2025–26); дисперсия 3.20 % → 1.98 % (−38 %) **[подтверждено]**, НО универсум из 25 монет включает SOL, AVAX, NEAR, APT, ICP, SHIB, которых не было в 01.2020 — survivorship | Point-in-time универсум с делистингами (CoinGecko/CMC снапшоты, даты листинга Binance) | Оборот 100–200 %/нед. + funding на шортах; post-2021 коллапс дисперсии; только как режимный признак (dispersion-gated) |
| Short-term cross-sectional reversal (1 день / 1 нед.) | Ранг по r за 1d/1w; long bottom, short top | 1 д – 1 нед. | **weak** | Zaremba et al. IRFA 2021: реверсал в мелких неликвидных, у крупных — continuation; GitHub 1-day, 20 б.п.: OOS gross alpha +73 % (t=4.64) → net Sharpe −1.73 (≈ −29 % net), оборот 138 %/день; train-период gross alpha −5.5 % (знак нестабилен) **[подтверждено]** | Daily close/volume/cap point-in-time | Умирает выше ~5 б.п. round-trip; на BTC/ETH факт — дневное continuation |
| Long-run reversal (>1 мес.) | Сорт по J≥4–8 нед.; long losers, hold K≥4 нед. | 1–3 мес. | **weak** | Dobrynskaya SSRN 2021: momentum до ~4 нед., затем реверсал, ~2 000 монет 2014–2020 [не проверено] | Weekly prices/caps point-in-time | Loser-leg неликвиден, делистинг; post-2021 вероятно decay |
| 15-мин sign reversal | sign(r_{t+1}) = −sign(r_t) на 15-мин барах, AUC | 15 мин | **no-evidence** | Заявлено: 183 пары Binance, 90 % значимы, 2–3 AUC-пункта (arXiv 2608.21888, авг. 2026) [существование не подтверждено] | Binance 15m klines | 2 б.п. maker ≈ заявленный edge на бар; только как признак входа |
| Quarter-hour / turn-of-candle periodicity, OI на границах | mean r в минутах {0,15,30,45}; OI = (buy−sell)/total в первые секунды после :00/:15/:30/:45, регрессия r_{t+4..12h} | 1 мин – 12 ч | **weak** | Hansen–Kim–Kimbrough JFEc 2024: периодичность в vol/volume, улучшает GARCH OOS (рецензировано, holds); 0.58 б.п./мин t>9 и OI→4–12 ч (arXiv 2607.09426, t 1.6–2.6) [не проверено] | Binance aggTrades, 1m klines | Per-minute ниже taker; OI→часы — единственная торгуемая часть, на одном непроверенном препринте |
| Hour-of-day / overnight seasonality | Hold BTC только в окне (напр. 21:00–23:00 UTC) | 2–8 ч | **weak** | Quantpedia: 33 % годовых, vol 20.9 %, MDD −22.5 % [не проверено], найдено перебором 24 окон | Hourly klines, NYSE calendar | Multiple testing; дамми сессий — только как признаки vol/liquidity |
| Day-of-week / weekend | r_t = Σ b_d D_d | 1 день | **no-evidence** | Знак меняется по подпериодам; ниже объём/vol на выходных | Daily/hourly klines | Не сигнал; дамми для sizing |
| Intraday TSMOM (first-30 → last-30 min) | r_last30 = a + b·r_first30 | intraday | **weak** | Wen–Bouri–Xu–Zhao NAJEF 2022: и momentum, и reversal, условно на jumps/FOMC/COVID (рецензировано; коэффициенты не проверены) | 1–5m klines, FOMC calendar | Артефакт US-сессии; 1 round-trip/день ≈ 5 б.п. терпимо, эффект мал и условен |
| **Volatility scaling / vol-managed overlay** | w_t = signal_t·σ_target/σ̂_t, σ̂ = 20–90d realized/EWMA, cap 1x; risk-managed WML = WML·c/σ_WML | любой | **strong** (как risk overlay) | Kim–Tse–Wald 2016: большая часть TSMOM-альфы MOP — от vol scaling (точный split 1.27 % vs 0.41 % не проверен); Moreira–Muir 2017; Barroso–Santa-Clara 2015; крипто 0.74 vs 0.79 — блог [не проверено] | Daily returns | Не даёт направления; churn от ресайзинга — использовать полосы 10–20 % |
| Hourly ML на TA-признаках с cost-фильтром | XGBoost/LSTM/iTransformer на 1h BTC; сделка если |ŷ| > τ·c; 27-fold walk-forward | 1 ч | **no-evidence / weak** | arXiv 2606.00060: наивные sign-стратегии убыточны при 10 б.п.; «>65 % годовых, Sharpe >1» — post-hoc селекция конфигурации [не проверено] | Hourly klines | Любой часовой сигнал — только с порогом по величине |
| [Добавлено] 12m TSMOM / 52-wk-high proximity | sign(r_{t−12m,t−1m}); P_t / max(P,252d) | 1–3 мес. | **no-evidence** (крипто) | В акциях/фьючерсах — десятилетия доказательств (MOP 2012; George–Hwang 2004); в BTC ~4 цикла халвинга = 4 наблюдения | Daily close | Оборот ничтожен; только медленный режимный признак |
| [Добавлено] BTC → alt lead-lag | r_alt,t+1 = a + b·r_BTC,t + c·(r_alt,t − β·r_BTC,t) | 1 ч – 1 день | **weak** | Connectedness-литература (Kristoufek, Ji, Corbet 2018–2023): BTC ведёт, эффект мал и в основном contemporaneous на часах | Hourly/daily klines | Арбитражируется быстро; признак residual momentum vs BTC beta |
| [Добавлено] MAX / lottery effect | MAX за 1–4 нед.; long low-MAX, short high-MAX | 1 нед. | **weak** | Grobys & Junttila JBEF 2021: отрицательная премия, мелкие монеты (in-sample) | Daily returns point-in-time | Short-leg неликвиден; не торгуется на perps масштабно |

### 2.2. Деривативы: funding, basis, OI, ликвидации, опционы

| Фактор | Формула | Горизонт | Сила | Эффект (статус) | Данные | Статус после публикации / издержек |
|---|---|---|---|---|---|---|
| Funding rate: уровень/экстремумы (time-series) | F_t 8h; z = (F−μ_30d)/σ_30d, contrarian при |z|>2; режим: 30d mean F<0 и длительность | 8 ч – 360 д | **weak** | Fulgur: coef −0.087, p=0.008, R²=0.003 (8h); z-score бэктест ≈0 net; K33: win rate 83–96 % при <10 независимых эпизодов — всё [не проверено] | Binance fapi fundingRate (с 2019), ccxt fetchFundingRateHistory | Taker-комиссия на порядок больше 8h-edge; funding сжался с 2024 |
| Funding cross-sectional carry-fade | Ранг perps по trailing funding; short top decile, long bottom; dollar-neutral, daily | 1 день | **no-evidence** (направление) | **[подтверждено, pre-registered]**: gross premium +0.1525 %/день; Sharpe 0.37; alpha +0.0424 %/д, CI [−0.073, +0.159]; DSR 0.762 (<0.95); **price-компонент short leg −0.0198 %/д, CI [−0.265, +0.221] — механизм фальсифицирован**; оборот 1.85×/д, издержки съедают ~74 %; holdout 2025-01..2026-06 не открывался (в прошлом драфте ошибка) | data.binance.vision klines + fundingRate + aggTrades | Направленной информации нет даже gross; премия = carry cash-flow |
| Futures basis / carry | b = (F/S − 1)·365/(T−t); perp premium; сорт по basis; high carry → crash risk | день (CS); недели (crash) | **weak** | BIS WP 1087 / Schmeling et al.: большой time-varying carry, high carry → повышенный crash-risk, retail leverage vs limited arb capital (абстракт совпадает с памятью; «60 % p.a.», корреляции 0.50/0.54 — [не проверено]) | CF Benchmarks, Coin Metrics, Deribit/Binance quarterly via ccxt | Post-ETF (01.2024) basis сжат basis-фондами; crash-gate низкооборотен, дневные сорты — судьба funding fade |
| Open interest change / quadrant | dOI = ln(OI_t/OI_{t−k}); leverage = OI/reserves; logit P(crash) | 1 день | **weak** | Emerald 2026: OR 1.48, OOS AUC 0.677 на 2023–2024 (мало независимых крахов, порог — степень свободы) [не проверено]; нет свидетельств предсказания знака | Binance openInterestHist (**только 30 дней — архивировать**), data.binance.vision metrics | Только tail-risk gate |
| CFTC COT (CME BTC) | NetPos_i = (L−S)/OI по TFF-классам; dNS retail; hedging pressure | 1 нед.+ | **weak** | Baur & Smales JFM 2022; Dunbar & Owusu-Amoako JBEF/FRL 2023: leveraged funds net short с timing ability; OOS R² в FRL [не проверено]; ~200 недель до 01.2024 | cftc.gov TFF (бесплатно) | Look-ahead: данные вторника публикуются в пятницу; после ETF Leveraged Funds shorts = cash-and-carry hedges — семантика сломалась |
| Long/short account ratio, top-trader ratio | LSR = #long/#short; top-20 % margin accounts | intraday–daily | **no-evidence** | Только вендорские дашборды | Binance ratio endpoints (30 дней), Coinglass | Уровень структурно >1, история сшита из третьих рук |
| Ликвидации / каскады | Liq_t long/short; imbalance; CSD early-warning | мин – 3 дня | **weak** (направление — no-evidence) | 7-cascade study: нет инвариантного EWS (CSD в 5/7) [не проверено]; Binance forceOrder throttled 1/сек → вендорские суммы — нижняя граница | Binance WS !forceOrder@arr (self-collect), Coinalyze | Сигнал возникает в стрессе, где спред/slippage максимальны; только vol/tail-state feature |
| Options IV / VRP (DVOL) | DVOL_t; VRP = IV² − RV² | 1 нед. – 1 мес. | **weak** | Alexander & Imeraj: VRP отрицательна в среднем, растёт перед большими движениями любого знака [не проверено]; ORL 2024: IV slopes → RV, не returns | Deribit API get_volatility_index_data (с 04.2021) | Нет направленного edge; хороший vol-режимный признак и для ширины барьеров |
| Options skew: 25Δ risk reversal, PCR | RR25 = IV(25Δc) − IV(25Δp); правило sign(RR) = sign(r до экспирации) | 30–90 д | **no-evidence** | **[подтверждено, GitHub]**: T-30 33/56 = 58.9 %, SE 6.7 %, p≈0.18, R²≈0.006; T-90 24/31 = 77.4 % — 3× overlapping windows (eff. n≈10), без издержек, confounded с дрейфом BTC; JFM 2023: option order imbalance → vol, не returns [не проверено] | Deribit API, Laevitas | Дешёвый sentiment/режимный признак; не сигнал |
| Futures OI на quarter-hour границах | OI в первые сек после :00/:15/:30/:45, регрессия r_{t+10s..12h} | 10 с – 12 ч | **weak** | 4–12 ч значимо у 4/6 контрактов, t 1.6–2.6; 10-с OOS R² 3.37 %, AUC 0.601; после издержек не отчитано (arXiv 2607.09426) [не проверено] | aggTrades с taker side | Маргинальные t; decay после публикации вероятен |
| CME weekend gap fill | Gap = Fri settle − Sun open; fade | дни–месяцы | **no-evidence** | «77 %/92 % fill» без бенчмарка безусловной вероятности ревизита | CME settlements, spot via ccxt | Неограниченный период удержания, без стопа |
| [Добавлено] Taker buy/sell ratio, signed OFI на perps | TBR = takerBuy/takerSell (5m–1h); OFI = Σ signed aggressive volume / total, z по 24h | сек – 1 день | **weak** | Chordia–Roll–Subrahmanyam; Cont–Kukanov–Stoikov; Silantyev 2019 (Digital Finance): contemporaneous R² на минутах, предиктивность затухает за минуты | data.binance.vision klines (taker_buy_base_asset_volume) и aggTrades (isBuyerMaker) **[поля подтверждены]** | На 1h+ внутри комиссий; признак flow-режима, не сигнал |
| [Добавлено] Dealer GEX / max pain / OI walls | GEX = Σ OI·γ·S²·0.01; max pain = strike, минимизирующий payout | дни до экспирации | **no-evidence** | Только вендоры; на Deribit сторона дилера ненаблюдаема — знак GEX есть допущение; Ni–Pearson–Poteshman 2005 в акциях — clustering, не направление | Deribit get_book_summary_by_currency | Максимум — vol-режим вокруг экспираций |
| [Добавлено] Spot ETF net flows | Flow_t = Σ creations − redemptions (Farside, T+1) | 1–5 дней | **weak / no-evidence** | Lim SSRN 6592830 (313 дней): $100M ≈ +53 б.п. same-day, R² 21 %, bidirectional Granger [не проверено]; same-day неторгуем | Farside, SoSoValue | ~15 мес. launch-режима; строгий publication lag |
| [Добавлено] Hyperliquid on-chain positioning | Top-N whale net exposure, distance to liquidation, HLP delta | часы–дни | **no-evidence** | Полная прозрачность ≠ edge; академических свидетельств нет | Hyperliquid info API, S3 | Дёшево собирать; не путать видимость с предсказуемостью |

### 2.3. Микроструктура (минуты–часы)

| Фактор | Формула | Горизонт | Сила (для сервиса) | Эффект (статус) | Данные | Статус |
|---|---|---|---|---|---|---|
| OFI (Cont–Kukanov–Stoikov), multi-level | e_n = 1{P^b_n ≥ P^b_{n−1}}q^b_n − 1{P^b_n ≤ P^b_{n−1}}q^b_{n−1} − 1{P^a_n ≤ P^a_{n−1}}q^a_n + 1{P^a_n ≥ P^a_{n−1}}q^a_{n−1}; ΔP = β·ΣOFI | 100 мс – 10 с | **no-evidence** (на минутах); strong на секундах | Акции: contemporaneous R² ≈65 % (JFEc 2014, рецензировано); крипто R² 0.094 (1с)/0.019 (10с) [не проверено] | Binance WS depth@100ms, Tardis | При задержке в минуты — adverse selection |
| Queue / book imbalance | I = (Q^b − Q^a)/(Q^b + Q^a), взвешенно по 20 уровням | след. тик – 30 с | **no-evidence** (минуты) | Gould–Bonart 2016 (рецензировано, акции); Kim–Hansen: «little predictive association» на 1/5-мин границах — свидетельство ПРОТИВ | WS depth | Полураспад секунды; taker 5 б.п. > ожидаемого движения |
| Taker imbalance 15м–4ч, quarter-hour boundary | TI_w = (V_takerbuy − V_takersell)/(V_takerbuy + V_takersell) | 15 мин – 12 ч | **weak** | 5–6 б.п. устойчиво + до 9.8/16.9 б.п. на 8/12 ч (Kim–Hansen); 90 % из 183 пар 15-мин реверсал [не проверено]; репликаций на GitHub — 0 | data.binance.vision klines/aggTrades **[подтверждено]** | Единственный кандидат семьи, переживающий задержку; edge на грани издержек; обязателен собственный OOS |
| Daily/weekly «world order flow» (Anastasopoulos et al., JFM 2026) | OF_{i,t} = Σ(buyer − seller-initiated volume) за день по парам; r_{t+1} = a + b·OF_t; ML LS-портфели | 1 д / 1 нед. | **weak → moderate (provisional)** | R² 1.2 % (день) / 3.4 % (нед.); weekly LS Sharpe 1.93; ML daily LS Sharpe 3.5–3.63, alpha 0.76–0.79 %/д, break-even 0.48 %/д, OOS 02.2020–06.2022 [не проверено; все 4 ссылки заблокированы] | aggTrades по многим парам; tick rule где нет флага | Sharpe 3.5 на дневных LS-альтах — признак концентрации в неликвидных монетах; один OOS-цикл, репликаций нет |
| VPIN / toxicity | VPIN = Σ|V^B − V^S|/(n·V), n≈50, BVC | часы (vol/jumps) | **weak** (режим) | Без знака по построению; Andersen–Bondarenko 2014: нет инкрементальной силы при контроле volume/vol; крипто-репо без валидации **[подтверждено]** | aggTrades / BVC по минутам | Только фильтр риска |
| DeepLOB / CNN-LSTM / TransLOB | T=100 снапшотов × 10 уровней → {up, flat, down} | доли сек – 2 с | **no-evidence** (сервис) | **[подтверждено, GitHub]**: F1 83.40/72.82/80.35 % (k=10/20/50, FI-2010); LSE 70.17 → 68.62 % на unseen; авторы: симуляция «merely a metric of testing predictability» без комиссий/спреда/латентности; LOBCAST: «significant performance drop when exposed to new data»; Jha et al. Coinbase BTC 2 с — 76 % (в прошлом драфте ошибочно 71 %) | Tardis L2, WS depth | Порог α ≈ спред; неприменимо для Telegram |
| Cross-exchange lead-lag | r_{j,t→t+500ms} = Σβ_ij·OFI_{i,t−Δ} + …; Hasbrouck IS | 500 мс | **no-evidence** (минуты) | 10–37 % OOS R² (Albers et al. AMF 2022) [не проверено]; данные включают FTX/Huobi | Tardis multi-exchange | Единственный вывод — строить признаки на Binance perps |
| L2 liquidity state (spread/depth regime) | s_t ∈ {calm, mixed, stressed} по кластеризации | мин – часы | **no-evidence** | Один препринт 2026 (2607.09230), AUC не раскрыт | WS depth20@100ms | Ненаправленный; режимный признак |
| Public trader identity (Hyperliquid) | Ранг кошельков по post-trade return; signed flow топ-кошельков | 1 с | **no-evidence** | Данные ДЕКАБРЯ 2025 (Zhai, SSRN 7234658), не «июль 2026»; цифры 0.52 / R² 12.31 % / t=9.2 в цитируемом репо ОТСУТСТВУЮТ **[подтверждено]** | Hyperliquid L4, Zenodo | Гипотеза |
| [Добавлено] Ликвидационный поток → реверсал | L_w = Σ liq long − Σ liq short (5–60 мин); реверсал после кластера одного знака | 15 мин – 4 ч | **no-evidence** | Не исследовано; единственный медленный по конструкции order flow, доступный retail бесплатно | Binance WS !forceOrder@arr | Издержки в стрессе многократно выше 5 б.п. |
| [Добавлено] Intraday-сезонность потока (US open, funding 00/08/16 UTC) × TI | Дамми × TI_w → r_{t+1..4h} | 1–4 ч | **weak** | Wen et al. NAJEF 2022 (рецензировано, in-sample) | klines, fundingRate history | Дешёвый GBM-признак |

### 2.4. On-chain, потоки, sentiment, макро

| Фактор | Формула | Горизонт | Сила | Эффект (статус) | Данные | Статус |
|---|---|---|---|---|---|---|
| Exchange reserve change / net flow | dReserve_t; NetFlow = In − Out; z-score 30–90d, lag 1 | 1 д; недели | **weak** | Hoang & Baur JBF 2022: рост резервов ↔ отрицательная доходность/vol (in-sample); arXiv 2411.06327: BTC inflows не предсказывают BTC на большинстве intraday-горизонтов [не проверено] | CryptoQuant (~$109/мес), Glassnode | Метки бирж перекластеризуются задним числом — не point-in-time; хранить дневные снапшоты |
| USDT net inflow to exchanges | Σ USDT non-exchange → exchange за 1–6 ч | 1–6 ч | **weak** | arXiv 2411.06327: USDT inflows положительно для BTC/ETH; ETH inflows отрицательно для ETH [не проверено, один препринт] | CryptoQuant stablecoin netflow (hourly, paid) | Intraday-оборот: нужно >10–20 б.п. |
| Stablecoin issuance / Tether mints | dSupply_t; treasury → market | часы – недели | **no-evidence** | Griffin & Shams JF 2020 vs Lyons & Viswanath-Natraj JIMF 2023, Wei 2018 — не реплицируется | DefiLlama, Coin Metrics Community | Только медленный liquidity-контекст |
| MVRV-Z / NUPL / CVDD | MVRV = MC/RC; Z = (MC − RC)/σ(MC); NUPL = (MC − RC)/MC; использовать rolling percentile | недели – годы | **weak** | Grobys RIBAF 2026: Sharpe 1.28 vs 0.45 B&H, пороги in-sample; пики MVRV 5.88→4.72→3.96→2.74; ни один top-индикатор не сработал на 10.2025 [не проверено] | Coin Metrics Community (CapMVRVCur, CapRealUSD, free) | ~3 цикла; амплитуда затухает; фиксированные пороги переобучены |
| SOPR / aSOPR | Σ USD spent at spend / Σ at creation | дни – недели | **no-evidence** | Нет рецензированного OOS | Glassnode/CryptoQuant (paid) | — |
| Miner reserves / hash rate / Puell | MinerOutflow; Puell = issuance USD / 365d MA; hash ribbon | дни – месяцы | **weak** | Granger: price → hash rate, не наоборот; ~450 BTC/день после халвинга — малая доля объёма | Coin Metrics Community, mempool.space | — |
| Google Trends attention | log(SVI_t) − mean(4 нед.); NegAtt = SVI('bitcoin hack')/SVI('bitcoin') | 1–2 нед. | **weak** | Liu–Tsyvinski RFS 2021: +1.8–2.3 % на 1 s.d. (in-sample 2011–2018); Physica A 2022: не проходит OOS [не проверено] | pytrends | Decay после 2018; индекс ресемплируется — не point-in-time |
| Twitter/X sentiment | (bull − bear)/total per day; zero-shot classifier | daily | **weak** | Gurgul et al. IJF 2025: рост Sharpe по rolling folds [не проверено]; остальное — короткие выборки | X API ($5/1 000 reads), Santiment | Архив невоспроизводим; доля ботов растёт |
| Reddit sentiment | VADER/transformer mean | daily | **no-evidence** | Null | Pushshift/Arctic Shift | Pushshift закончился 2023 |
| News NLP / LLM headline scoring | Score ∈ {−1,0,1} per headline; daily mean | след. день | **weak** | Lopez-Lira & Tang 2023: GPT-4 предсказывает next-day US equities, сильнее small caps/негатив, decay с adoption (рецензировано, акции); крипто — только препринты | RSS CoinDesk/CoinTelegraph/The Block + Claude scoring; GDELT | Бэктест до knowledge cutoff модели = утечка |
| Fear & Greed Index | 25 % vol + 25 % momentum/volume + 15 % social + 15 % surveys + 10 % dominance + 10 % Trends **[веса совпадают с документацией]** | daily/weekly | **no-evidence** | 2026 VAR: returns → FGI, не наоборот (механически ожидаемо) [не проверено] | alternative.me API (free) | Lagging transform цены |
| DXY daily change | 100·Δln DXY; VAR(p) → BTC; сделка при |forecast| > τ | 1 д | **weak** | Arain & Snudden J. Forecasting 2026: USD index и Shanghai дают значимую прибыль на 10.2021–02.2024 (28 мес., один режим) [не проверено]; Benigno & Rosa NY Fed 2023: BTC ортогонален MP surprises | FRED DTWEXBGS | Связь режимно-зависима (≈0 до 2020) |
| Δ корреляции с акциями (DCC) | ρ_t (DCC-GARCH BTC vs S&P/Nasdaq); Δρ_t | 1 д / 1 нед. | **weak** | Physica A 2022: OOS R² 2.7 % BTC / 1.7 % ETH / 2.1 % XRP, выборка до ~2021 [не проверено] | Yahoo/Stooq + ccxt; `arch` | R² 2 % ≈ hit rate чуть >50 %; маргинально после 10 б.п. |
| Global M2 | Σ M2 в USD, YoY, лаг 70–110 д | 3–6 мес. | **no-evidence** | Alden/Callahan: корр. уровней 0.94, 83 % 12-мес. совпадение направления; в разностях исчезает | FRED, ALFRED vintages | Spurious levels regression; лаг публикации и ревизии |
| Real yields / FOMC surprises | ΔDFII10; MP surprise в 30-мин окне | event – daily | **weak** (direction: no) | Post-FOMC mean |r| ≈ ×2 (vol, не знак) [не проверено] | FRED, календари Fed/BLS | Использовать для расширения барьеров / отключения сигналов |
| [Добавлено] Coinbase premium | (Coinbase BTC-USD − Binance BTC-USDT)/Binance | часы – дни | **no-evidence** | Только вендор (CryptoQuant) | ccxt OHLCV Coinbase+Binance | Быстро арбитражируется |
| [Добавлено] STH realized price, accumulation score, SSR, dominance, VIX/gold | — | — | не оценено | Только вендорские | Glassnode, checkonchain | Кандидаты для последующего раунда |

### 2.5. ML-модели и валидация

| Компонент | Формула / дизайн | Сила | Эффект (статус) | Статус |
|---|---|---|---|---|
| **Purged/embargoed CPCV + DSR + PBO** | Purge overlap меток; embargo; CPCV C(N,K); PBO via CSCV; DSR = PSR при E[max Sharpe | N trials] | **strong** | **[подтверждено]**: shuffled KFold R² 0.83/0.91 на шуме vs −1.48/−1.87 purged | Gate: DSR > 0.95, PBO < 0.2, положительное после-издержечное expectancy на всех путях |
| GBDT (LightGBM/CatBoost) на engineered признаках | y = triple-barrier; X = lagged returns, RV, volume z, range, cross-asset, time-of-day | **weak** (как литературный факт) | G-Research 2022 winner = LightGBM, Jaquart 1–60 мин >50 %, Akyildirim 55–65 % (non-purged) — все [не проверено] | Разумный default; 52–55 % на 4h–1d — рабочая гипотеза, измерять самим |
| Meta-labeling (JFDS series) | Primary side → triple-barrier → M2(X, side) → p; trade if p > τ; size = f(p) | **weak** (качественно рецензировано) | **[подтверждено, GitHub]**: 4 статьи JFDS 2022–2023; улучшает Sharpe/MaxDD через фильтр и sizing; калибровка «significantly improved» fixed sizers. Цифра «17 % → 63 %» — не цитировать | Не создаёт edge без primary signal |
| LSTM/RNN | 240 lagged returns → P(r > median) | **weak** | Fischer–Krauss 2018: 0.46 %/д, Sharpe 5.8 до издержек 1992–2015, decay после 2010 [не проверено]; «59.1 %» — снять | Урок = post-publication decay |
| Transformers | Spacetimeformer context 128/250, horizon 16 **[подтверждено config]** | **no-evidence** | Score и сравнение с LightGBM в README отсутствуют | «Не побили LightGBM» — inference, не факт |
| Regime models (HMM/NHHM/MS) как признаки | filtered posterior P(S_t | F_t), только forward | **weak** | Koki et al.: лучшая predictive density (NHHM 4 states) [не проверено]; направленного edge нет | Smoothed posteriors = look-ahead |
| Ensembles / stacking | mean или веса по OOS log-loss на purged folds | **weak** | JFDS: «особенно полезно при нескольких режимах» [подтверждён абстракт]; «+0.005–0.02 rank-corr» — снято | Инженерная практика, не предиктор |
| TS foundation models (Chronos-2, TimesFM, Kronos) | zero-shot quantiles, sign(q_0.5) | **no-evidence** | **[подтверждено]**: TSFM_Finance = Rahimikia–Ni–Wang arXiv 2511.18578; Kronos MIT, «not a production-ready quantitative trading system»; Chronos-2 Apache-2.0 (20.10.2025). ~50–51 %, Chronos OOS R² −1.37 %, Kronos Brier 0.189 vs 0.188 — [не проверено] | Нет edge; look-ahead через pretraining cutoff |
| Numerai Crypto | target = 30d forward return, ranked, gaussianized, 5 bins; CORR = Pearson(sign·|gauss(rank)|^1.5, sign·|target−mean|^1.5) **[подтверждено]** | **weak** | Типичных величин Numerai не публикует | Не BTC-сигнал; 30-дневная cross-sectional задача |
| [Добавлено] Калибровка + conformal abstention (MAPIE/crepes) | isotonic/Platt на purged OOF; сигнал если |p − 0.5| > τ и conformal set singleton; E = p·TP − (1−p)·SL − cost > 0 | **weak** (механизм) | JFDS: калибровка помогает fixed sizers [абстракт подтверждён] | Главный рычаг после издержек для редких single-asset сигналов |

---

## 3. Рекомендуемая модель v1 (горизонт 4 ч – 3 дня, сигналы через Telegram)

**Принцип:** знак предсказываем слабо, поэтому модель строится как «фильтр + размер», а не как «оракул». Сигнал выходит только когда калиброванная вероятность и ожидаемая стоимость после издержек это оправдывают; в остальное время бот молчит.

### 3.1. Активы и данные
- BTC-USDT, ETH-USDT perps Binance (основные); SOL, BNB, XRP и др. top-20 по ликвидности — во второй итерации, с point-in-time универсумом (даты листинга, делистинги).
- Бары: 1h (агрегируем до 4h признаков), daily. Источник: data.binance.vision (klines с `taker_buy_base_asset_volume`, aggTrades с `isBuyerMaker`, fundingRate) + ccxt live. **Сразу поднять архиватор** для openInterestHist / LSR / takerlongshortRatio (Binance отдаёт только 30 дней; cadence 4h, как уже предписано в docs/research/regime-ai-stack-2026-09.md). Сбор — с VPS/Mac: из контейнера API бирж заблокированы.
- Внешние ряды (бесплатные): FRED DTWEXBGS, DFII10; Deribit DVOL; alternative.me FGI; Coin Metrics Community MVRV; Farside ETF flows (T+1 lag строго).

### 3.2. Признаки (12 штук, все причинные, только forward-filter)

| # | Группа | Признак | Определение |
|---|---|---|---|
| F1 | Тренд | Donchian-ансамбль | mean_L 1{close > max(close, L)}, L ∈ {10, 20, 60, 120} дней (упрощённый Zarattini) |
| F2 | Тренд | log(P/MA_L) | L ∈ {20, 50, 100} дней, три непрерывных признака |
| F3 | Тренд | TSMOM 1w/4w | sign и величина r_{t−7d,t}, r_{t−28d,t}, делённые на σ_30d |
| F4 | Vol | realized vol ratio | σ_5d / σ_60d (EWMA) + абсолютный уровень σ_20d — нужен и как признак, и для барьеров |
| F5 | Flow | taker imbalance | TI_4h, TI_24h = (takerBuy − takerSell)/(takerBuy + takerSell), z-score по 30 дням |
| F6 | Flow | dOI | ln(OI_t/OI_{t−24h}), ln(OI_t/OI_{t−7d}) — как tail-risk gate |
| F7 | Positioning | funding regime | 30d mean funding (знак, величина) + длительность текущего знака; **realized**, не predicted |
| F8 | Vol/opt | DVOL и ΔDVOL | уровень DVOL, 5d change; VRP = DVOL² − RV²_30d |
| F9 | Cross-asset | BTC-beta residual (для альтов) | r_alt − β_60d·r_BTC за 24h/7d; для BTC — Δρ_60d с Nasdaq |
| F10 | Macro | DXY Δ, FOMC/CPI dummy | 100·Δln DXY (1d, 5d); календарный флаг событий ±24h |
| F11 | Regime | HMM filtered posterior | P(S_t = k | F_t) из существующего HMM пайплайна, forward filter only |
| F12 | Time | session dummies | UTC hour bucket, weekday, минуты до funding settlement (00/08/16 UTC) |

Опционально (после H-тестов): дневной «world order flow» по панели монет; LLM headline score (только post-cutoff окно); MVRV percentile как медленный контекст.

**Не включать в v1:** LOB/OFI на секундах, 15-мин реверсал, quarter-hour OI, LSR ratios, F&G как сигнал, M2, Tether mints, GEX/max pain, Hyperliquid whales.

### 3.3. Целевая переменная: triple-barrier мета-метка
- Primary side s_t ∈ {−1, +1} = знак ансамбля F1+F3 (тренд). Если ансамбль ≈ 0 — нет primary сигнала, нет метки.
- Барьеры: TP = +2·σ_h, SL = −1.5·σ_h, где σ_h — EWMA-волатильность, масштабированная на горизонт; вертикальный барьер h = 48 ч (тест 24/48/72 ч). Ширина барьера ≥ 3× round-trip cost (иначе метки = шум).
- Метка y_t = 1, если первым тронут барьер в сторону s_t; 0 иначе. Веса наблюдений = uniqueness (пересечение окон).
- Итог: M2 предсказывает P(primary-сигнал окажется прав), а не знак сам по себе.

### 3.4. Модель
- LightGBM (или CatBoost) с монотонными ограничениями там, где они экономически осмысленны (напр. F4 → ширина барьера, не знак); max_depth 3–4, min_child_samples большой, сильная регуляризация. Ансамбль по 5 seed × CPCV-путям.
- Калибровка: isotonic на purged OOF; проверка reliability curve, Brier, log-loss против константной модели (base rate).
- Валидация: purged CPCV (embargo ≥ h), N трайлов считать честно для DSR (каждый optuna-trial, набор признаков, ширина барьера, порог).
- Бенчмарки, которые нужно побить **после издержек**: (a) vol-managed long-only без сигнала; (b) buy-and-hold; (c) primary без M2.

### 3.5. Abstain-правило («торговать только при сильном сигнале»)
Сигнал публикуется, если одновременно:
1. p_cal > τ, где τ выбран на OOF так, чтобы E = p·TP − (1−p)·SL − cost_roundtrip > 0 с запасом (стартовое τ ≈ 0.58–0.62; уточнять по путям CPCV);
2. conformal set (MAPIE/crepes, α = 0.2) — singleton;
3. нет FOMC/CPI в ±24 ч; dOI_7d и DVOL не в верхних 5 % (tail gate);
4. HMM posterior «high-vol crash» < 0.5.
Иначе — «нет сигнала». Ожидайте, что бот молчит большую часть времени; это нормально.

### 3.6. Управление риском
- Размер = min(1, σ_target / σ̂_t) × f(p_cal), σ_target ≈ 20–25 % годовых, cap 1x, без плеча в v1; ребаланс только при изменении веса >15 %.
- Стоп = SL-барьер триплета; time-stop = вертикальный барьер.
- Лимит одновременных позиций и per-coin exposure (short-leg blow-ups: Fieberg/Grobys — не проверено, но известный риск).
- Мониторинг дрейфа: PSI по признакам, ADWIN по hit rate; переобучение rolling каждые 4–8 недель с фиксированным протоколом.
- В сообщении Telegram — только p_cal, барьеры, размер и «без гарантий»; никаких прогнозов доходности.

---

## 4. План экспериментов

Общие правила для всех H: point-in-time данные; purged CPCV с embargo = горизонт метки; учёт издержек 10 б.п. round-trip (taker) и 5 б.п. (maker) отдельно; **критерий успеха = DSR > 0.95 при честном N трайлов, PBO < 0.2, положительное после-издержечное expectancy на ≥80 % CPCV-путей, и превосходство над vol-managed long-only**. Placebo-тест (перемешанные во времени признаки / случайные метки) обязателен для каждой H. Бонферрони-подобная поправка встроена через DSR (N = число всех спецификаций в семье).

| # | Гипотеза | Точное определение признака | Тест | Критерий успеха |
|---|---|---|---|---|
| **H1** | Vol targeting улучшает risk-adjusted результат любого сигнала | w = σ_target/σ̂_20d, cap 1x, полосы 15 % | Sharpe, MDD, Calmar long-only BTC/ETH 2017–2026 vs B&H; bootstrap CI | Sharpe выше B&H и MDD ниже на всех 5-летних окнах; это baseline, не «edge» |
| **H2** | Donchian-ансамбль на BTC/ETH даёт положительное net expectancy | F1 с L ∈ {10,20,60,120}, trailing stop по (chmax+chmin)/2, 10 б.п. | Реплика Zarattini на своих данных 2015–03.2025, затем holdout 04.2025–09.2026 | Net Sharpe ≥ 0.8 на holdout, DSR > 0.95 (N ≥ 9 lookbacks × 3 варианта stop); ожидайте цифры ниже статьи |
| **H3** | Meta-label M2 повышает точность primary тренд-сигнала на 48 ч | Triple-barrier (2σ/1.5σ/48h), LightGBM на F1–F12 | CPCV, Brier/log-loss vs base rate, precision@τ | Log-loss ниже константы на ≥80 % путей; precision при τ=0.6 ≥ 0.58 после издержек; PBO < 0.2 |
| **H4** | Taker imbalance 4h/24h несёт инкрементальную информацию для 48h метки | TI_w z-score, w ∈ {4h, 24h} | SHAP/permutation importance в M2; ablation ±F5 | Ablation ухудшает log-loss значимо (bootstrap p < 0.05 с поправкой на N=2 окна) |
| **H5** | Дневной signed order flow по панели top-20 предсказывает next-day returns (реплика Anastasopoulos) | OF_{i,t} из aggTrades isBuyerMaker; rank-IC; LS-децили | Rank-IC t-stat, LS Sharpe с издержками 20 б.п. + funding; отдельно на BTC/ETH только | IC t > 3 после Newey–West, net Sharpe > 0.5 на ликвидных 10 монетах; если edge только в неликвидных — отклонить |
| **H6** | Funding-режим (30d mean < 0 ≥ 14 дней) предсказывает положительную 30d доходность выше безусловной | F7 | Событийное исследование, только неперекрывающиеся эпизоды; сравнение с always-long | Ожидаем провал по мощности (<10 эпизодов): регистрируем как «нет ответа», не как «работает» |
| **H7** | dOI и DVOL как tail gate снижают MDD без потери expectancy | F6, F8 в верхнем 5 % → нет сигнала | Сравнение стратегии H3 с гейтом и без | MDD ниже на ≥20 % при expectancy не хуже на 90 % путей |
| **H8** | Δρ(BTC, Nasdaq) и ΔDXY дают OOS R² > 0 на дневном/недельном горизонте (реплика Physica A 2022 / Arain–Snudden) | DCC-GARCH ρ_t, 100·Δln DXY | Campbell–Thompson OOS R² vs historical mean, 2018–2026 rolling | OOS R² > 0 с CI, не пересекающим 0; иначе — только как признак M2 |
| **H9** | Δ калибровки + conformal abstention повышают net expectancy на единицу сигнала | isotonic + MAPIE α=0.2 | Сравнение expectancy/сделку и числа сделок vs без abstain | Expectancy выше при DSR > 0.95; число сигналов ≥ 2/мес. (иначе продукт мёртв) |
| **H10** | HMM filtered posteriors улучшают M2 | F11 forward-only | Ablation | Как H4 |
| **H11** | Quarter-hour taker imbalance → 4–12 ч (реплика Kim–Hansen на данных после 10.2024) | TI за первые 10 с после :00/:15/:30/:45 | Регрессия с block-bootstrap; отдельно post-publication окно | t > 2.5 после издержек maker 5 б.п.; только тогда — признак, не сигнал |
| **H12** | LLM headline score (Claude) предсказывает next-day BTC на post-cutoff окне | Score ∈ {−1,0,1}, daily mean, RSS | Только окно после knowledge cutoff модели | IC t > 2.5; иначе отклонить |
| **H13** | Cross-sectional momentum 1–4 нед. жив на point-in-time универсуме после 2022 | LTW-стиль, VW квинтили, с делистингами | Net LS Sharpe 2022–2026, 20 б.п. + funding | Ожидаем провал; фиксируем decay как результат |
| **H14** | ETF net flow (T+1) предсказывает r_{t+1} | z-score 5d/20d flow, строго после публикации | Регрессия 2024–2026 | t > 2.5 после Newey–West; учитывая 2 года данных — только как признак |

Порядок: H1–H3 обязательны до любого продакшена; H4–H7 — расширение v1; H8–H14 — второй спринт. Каждый отклонённый тест фиксируется в docs/research как отрицательный результат (не удалять).

---

## 5. Что НЕ работает или переоценено

1. **Всё суб-минутное для Telegram** — OFI, queue imbalance, DeepLOB, cross-exchange lead-lag. Эффекты реальны и рецензированы на акциях (Cont et al. 2014; Gould–Bonart 2016; Zhang et al. 2019; Kolm et al. 2023), но горизонт 0.1–30 с; авторы DeepLOB сами исключают комиссии/спред/латентность **[подтверждено]**, LOBCAST фиксирует падение на новых данных **[подтверждено]**. При задержке в минуты — adverse selection. Источники: https://arxiv.org/abs/1808.03668, https://github.com/jjakimoto/research-issues/issues/1883, https://github.com/matteoprata/LOBCAST, https://onlinelibrary.wiley.com/doi/10.1111/mafi.12413.
2. **Cross-sectional funding carry-fade как направленный сигнал** — pre-registered study фальсифицировала price-механизм (short-leg price −0.0198 %/д, CI через ноль), 74 % gross съедены издержками, DSR 0.76 **[подтверждено]**: https://github.com/JosephBerachah/crypto-fundingrate-alpha. В прошлом драфте неверно сказано про «sealed holdout» — holdout не открывался.
3. **1-day cross-sectional reversal на ликвидных монетах** — +73 % gross → ≈ −29 % net при 20 б.п. и 138 % дневного оборота **[подтверждено]**; знак нестабилен по подпериодам: https://github.com/prams2104/crypto-momentum-backtest.
4. **«Чистый point-in-time тест 2020–2026» cross-sectional momentum** — не чистый: 25-монетный список содержит SOL, AVAX, NEAR, APT, ICP, SHIB, которых не существовало в 01.2020 **[подтверждено]**. Decay-числа — намёк, не репликация.
5. **Options 25Δ risk reversal** — 58.9 % (n=56, p≈0.18, R² 0.006); T-90 77.4 % (n=31) — overlapping windows и дрейф BTC **[подтверждено]**: https://github.com/JERONIMODIFRANCO/deribit-rr/tree/main.
6. **Fear & Greed, Global M2, Tether mints, Reddit, day-of-week** — no-evidence: F&G — lagging-трансформ цены; M2 — корреляция уровней 0.94, исчезающая в разностях (https://www.lynalden.com/bitcoin-a-global-liquidity-barometer/); Tether — не реплицируется (https://cepr.org/voxeu/columns/stable-coins-dont-inflate-crypto-markets, https://www.sciencedirect.com/science/article/abs/pii/S0165176518302556); day-of-week — знак меняется по подпериодам (https://mlquants.substack.com/p/are-day-of-the-week-effects-in-cryptocurrencies).
7. **TS foundation models (Chronos-2, TimesFM, Kronos)** — ~50 % направления, авторы Kronos: «not a production-ready quantitative trading system» **[подтверждено]**: https://github.com/shiyu-coder/Kronos, https://github.com/DeepIntoStreams/TSFM_Finance.
8. **Цикловые индикаторы (MVRV-Z, NUPL, Puell, Pi Cycle)** — ~3 независимых наблюдения, затухание амплитуды, фиксированные пороги переобучены по построению: https://www.sciencedirect.com/science/article/pii/S0275531926002138, https://arxiv.org/abs/2607.26188 [не проверено].
9. **Три интрадей-препринта 2026** (2608.21888 — 15-мин реверсал; 2607.09426 — quarter-hour OI; 2606.00060 — hourly ML) — после knowledge cutoff, не открыты, репликаций нет (GitHub — 0 репозиториев). Не кодировать до прочтения PDF.
10. **Ошибки предыдущих раундов, исправленные здесь:** Hudson & Urquhart — Annals of OR 2021, ~15 000 правил (не FRL 2019/2 127); Hyperliquid public-trader-identity — данные декабря 2025 (Zhai, SSRN 7234658), цифр 0.52 / 12.31 % / t=9.2 в репо нет (https://github.com/daojingzhai/public-trader-identity); Jha et al. deep-orderbook — 76 %, не 71 % (https://github.com/Globe-Research/deep-orderbook); meta-labeling «17 % → 63 %» — не найдено ни в одном подтверждённом источнике.
11. **Structural breaks:** спотовые ETF (01.2024) сжали basis и funding и изменили смысл COT Leveraged Funds shorts (теперь cash-and-carry); работы на FTX/Huobi устарели; после 10.10.2025 глубина в стрессе исчезает нелинейно.
12. **Общая ловушка:** hit rate ≠ expectancy; smoothed HMM-posteriors = look-ahead; метки бирж и Google Trends не point-in-time; любой LLM-бэктест до cutoff — утечка.

---

## 6. Источники

### Подтверждены первоисточником в этом раунде (GitHub)
- https://github.com/prams2104/crypto-momentum-backtest
- https://github.com/zebadee2kk/DeFi-TraderStack-Agent/issues/137
- https://github.com/alfred1123/Quant_Strategies/pull/61
- https://github.com/JosephBerachah/crypto-fundingrate-alpha
- https://raw.githubusercontent.com/JosephBerachah/crypto-fundingrate-alpha/main/docs/preregistration.md
- https://github.com/JERONIMODIFRANCO/deribit-rr/tree/main
- https://github.com/binance/binance-public-data/
- https://github.com/jjakimoto/research-issues/issues/1883
- https://github.com/matteoprata/LOBCAST
- https://github.com/FinancialComputingUCL/LOBFrame
- https://github.com/Globe-Research/deep-orderbook
- https://github.com/toma-x/exploring-order-book-predictability
- https://github.com/MobiHussain/vpin-jump-detector
- https://github.com/daojingzhai/public-trader-identity
- https://github.com/hudson-and-thames/meta-labeling
- https://github.com/eslazarev/purged-cross-validation
- https://github.com/qAp/gresearch_crypto_forecasting_kaggle
- https://github.com/DeepIntoStreams/TSFM_Finance
- https://github.com/shiyu-coder/Kronos
- https://github.com/amazon-science/chronos-forecasting
- https://github.com/numerai/docs/blob/master/numerai-crypto/data.md
- https://github.com/numerai/docs/blob/master/numerai-crypto/staking.md
- https://github.com/numerai/docs/blob/master/numerai-tournament/scoring/correlation-corr.md
- https://github.com/bond-labs-dev/hyperliquid-data

### Ценовые факторы (не открыты; сверено с памятью)
- https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.13119
- https://www.nber.org/papers/w25882
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4287573
- https://link.springer.com/article/10.1007/s11408-025-00474-9
- https://onlinelibrary.wiley.com/doi/abs/10.1002/ijfe.70036
- https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/trend-factor-for-the-cross-section-of-cryptocurrency-returns/4C1509ACBA33D5DCAF0AC24379148178
- https://academic.oup.com/rfs/article-abstract/34/6/2689/5912024
- https://www.nber.org/system/files/working_papers/w24877/w24877.pdf
- https://w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf
- https://www.sciencedirect.com/science/article/abs/pii/S1386418116301379
- https://zenodo.org/records/19671502
- https://www.journals.vu.lt/BATP/en/article/download/44540/42590/138419
- https://summitward.com/learn/crypto-trend-following
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5209907
- https://onlinelibrary.wiley.com/doi/abs/10.1111/fima.12310
- https://www.sciencedirect.com/science/article/abs/pii/S1544612319303770
- http://dirkgerritsen.nl/uploads/gerritsen_et_al_2020_bitcoin_trading_rules.pdf
- https://www.kevinsheppard.com/files/teaching/mfe/advanced-econometrics/Sullivan_Timmermann_White.pdf
- https://www.sciencedirect.com/science/article/pii/S1057521921002349
- https://ideas.repec.org/a/eee/phsmap/v523y2019icp691-701.html
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3913263
- https://arxiv.org/abs/2608.21888
- https://arxiv.org/abs/2607.09426
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10015199/
- https://arxiv.org/abs/2109.12142
- https://academic.oup.com/jfec/advance-article/doi/10.1093/jjfinec/nbac034/6759403
- https://quantpedia.com/strategies/intraday-seasonality-in-bitcoin
- https://paperswithbacktest.com/blog/bitcoin-never-sleeps-exploiting-seasonality
- https://mlquants.substack.com/p/are-day-of-the-week-effects-in-cryptocurrencies
- https://www.mdpi.com/1911-8074/17/8/351
- https://www.sciencedirect.com/science/article/abs/pii/S1062940822000833
- https://centaur.reading.ac.uk/100181/3/21Sep2021Bitcoin%20Intraday%20Time-Series%20Momentum.R2.pdf
- https://arxiv.org/abs/2606.00060

### Деривативы и позиционирование
- https://medium.com/@fulgur.ventures/bitcoin-funding-rates-and-price-predictability-27ce95535af1
- https://tradingstrategies.work/blog/funding-rate-signal-btc-backtest
- https://www.theblock.co/post/400182/bitcoin-longest-negative-funding-streak-this-decade-k33-flags-short-squeeze-risk
- https://www.theblock.co/post/397531/bitcoin-breakout-odds-rise-negative-funding-streak-mirrors-bottoming-regimes-k33
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5576424
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6701738
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6818558
- https://arxiv.org/pdf/2510.14435
- https://www.bis.org/publ/work1087.pdf
- https://pubsonline.informs.org/doi/10.1287/mnsc.2024.05069
- https://onlinelibrary.wiley.com/doi/full/10.1002/fut.22425
- https://www.cfbenchmarks.com/blog/revisiting-the-bitcoin-basis-how-momentum-sentiment-impact-the-structural-drivers-of-basis-activity
- https://www.emerald.com/sef/article/doi/10.1108/SEF-04-2026-0367/1388748/Systemic-risk-from-financial-leverage-in-digital
- https://phemex.com/academy/open-interest-bitcoin-trading-2026
- https://www.coindesk.com/markets/2026/08/25/a-bitcoin-short-squeeze-for-the-ages-as-futures-open-interest-collapses
- https://onlinelibrary.wiley.com/doi/full/10.1002/fut.22332
- https://www.sciencedirect.com/science/article/abs/pii/S2214635023000266
- https://www.sciencedirect.com/science/article/abs/pii/S1544612323003811
- https://digitalcommons.uncfsu.edu/college_business_economics/314/
- https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Top-Trader-Long-Short-Ratio
- https://www.coinglass.com/LongShortRatio
- https://www.kaggle.com/datasets/jesusgraterol/bitcoin-longshort-ratio-binance-futures
- https://arxiv.org/abs/2607.27070
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6579278
- https://arxiv.org/abs/2102.04591
- https://blog.amberdata.io/how-3.21b-vanished-in-60-seconds-october-2025-crypto-crash-explained-through-7-charts
- https://cryptoslate.com/new-bitcoin-study-shows-the-strongest-recurring-liquidation-warning-signs-cannot-warn-of-an-individual-crash/
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3383734
- https://arxiv.org/abs/2410.15195
- https://www.sciencedirect.com/science/article/abs/pii/S0167637724000713
- https://insights.deribit.com/industry/bitcoin-options-finding-edge-in-four-years-of-volatility-regimes/
- https://www.sciencedirect.com/science/article/pii/S1386418122000544
- https://arxiv.org/pdf/2109.02776
- https://onlinelibrary.wiley.com/doi/full/10.1002/fut.70004
- https://arxiv.org/html/2607.09426
- https://finance.yahoo.com/markets/crypto/articles/bitcoin-first-cme-gap-free-174649486.html
- https://phemex.com/academy/cme-futures-gap
- https://link.springer.com/article/10.1007/s42521-019-00007-w
- https://insights.deribit.com/
- https://www.jstor.org/stable/3694735
- https://farside.co.uk/btc/
- https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/rate-limits-and-user-limits
- https://www.binance.com/en/futures/funding-history/perpetual/real-time-funding-rate
- https://www.cmegroup.com/markets/cryptocurrencies/bitcoin/bitcoin.html

### Микроструктура
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1712822
- https://arxiv.org/pdf/1011.6402
- https://onlinelibrary.wiley.com/doi/10.1111/mafi.12413
- https://arxiv.org/pdf/2308.14235
- https://www.tandfonline.com/doi/full/10.1080/1350486X.2022.2080083
- https://arxiv.org/abs/1512.03492
- https://arxiv.org/abs/2602.00776
- https://arxiv.org/abs/2506.05764
- https://www.sciencedirect.com/science/article/pii/S1386418126000029
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5020002
- https://www.efmaefm.org/0EFMAMEETINGS/EFMA%20ANNUAL%20MEETINGS/2025-Greece/papers/OrderFlowpaper.pdf
- https://www.sciencedirect.com/science/article/pii/S0275531925004192
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4814346
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1881731
- https://arxiv.org/abs/1808.03668
- https://arxiv.org/abs/2108.09750
- https://arxiv.org/abs/2607.09230
- https://arxiv.org/abs/2608.04373
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6795938
- https://arxiv.org/abs/2105.10430

### On-chain, sentiment, макро
- https://www.sciencedirect.com/science/article/abs/pii/S0378426622002023
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3902504
- https://arxiv.org/abs/2411.06327
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4630115
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3508006
- https://www.sciencedirect.com/science/article/abs/pii/S0165176518302556
- https://cepr.org/voxeu/columns/stable-coins-dont-inflate-crypto-markets
- https://www.sciencedirect.com/science/article/pii/S0275531926002138
- https://arxiv.org/abs/2607.26188
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4719874
- https://arxiv.org/abs/2606.00071
- https://docs.glassnode.com/introduction/metric-catalog
- https://www.sciencedirect.com/science/article/pii/S0275531925005197
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6592830
- https://ledger.pitt.edu/ojs/ledger/article/view/393
- https://www.sciencedirect.com/science/article/abs/pii/S0378437122002928
- https://www.sciencedirect.com/science/article/pii/S0169207025000147
- https://arxiv.org/pdf/2311.14759
- https://www.etd.ceu.edu/2023/zhumagaziyev_sh.pdf
- https://arxiv.org/pdf/2204.05781
- https://arxiv.org/abs/2304.07619
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4412788
- https://arxiv.org/pdf/2407.09546
- https://www.sciencedirect.com/science/article/pii/S305070062600006X
- https://alternative.me/crypto/fear-and-greed-index/
- https://onlinelibrary.wiley.com/doi/full/10.1002/for.70077
- https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr1052.pdf
- https://econpapers.repec.org/RePEc:eee:phsmap:v:598:y:2022:i:c:s0378437122002928
- https://www.lynalden.com/bitcoin-a-global-liquidity-barometer/
- https://blog.traderspost.io/article/m2-money-supply-bitcoin-correlation-explained
- https://www.sciencedirect.com/science/article/abs/pii/S1544612326006021
- https://cryptoquant.com/asset/btc/chart/market-indicator/coinbase-premium-index
- https://www.deribit.com/statistics/BTC/volatility-index
- https://docs.deribit.com/

### ML и валидация
- https://www.gresearch.com/news/wrapping-up-the-g-research-crypto-forecasting-competition/
- https://www.kaggle.com/c/g-research-crypto-forecasting
- https://www.sciencedirect.com/science/article/pii/S2405918821000027
- https://link.springer.com/article/10.1007/s10479-020-03575-y
- https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/
- https://www.sciencedirect.com/science/article/abs/pii/S0377221717310652
- https://arxiv.org/abs/2004.10178
- https://mlcontests.com/tabular-data/
- https://arxiv.org/abs/2011.03741
- https://arxiv.org/abs/2401.03393
- https://arxiv.org/abs/2511.18578
- https://cornfordandcross.com/market-insights/week-three-foundation-model-vs-brownian-motion-kronos-on-five-minute-btc/
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253

---

**Итоговая позиция.** Направление криптоактивов на горизонте 4 ч – 3 дня предсказуемо слабо (реалистично 51–55 % на BTC/ETH от трендовых признаков), и эта предсказуемость сама по себе не гарантирует прибыли. Практическая ценность сервиса будет складываться из vol-managed sizing, abstain-правила, tail-gate по OI/DVOL/событиям и честной валидации (purged CPCV + DSR + PBO), а не из «угадывания рынка». Ни одно число из этого отчёта, помеченное [не проверено], не должно попасть в код до прочтения первоисточника с машины с открытым интернетом. Доходность не обещается.