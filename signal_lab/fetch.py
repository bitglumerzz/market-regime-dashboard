"""Загрузка реальных данных для лаборатории — запускать на своём компьютере.

    python -m signal_lab.fetch --symbol BTCUSDT --tf 4h --since 2019-09-01            # фьючерсы USDⓈ-M
    python -m signal_lab.fetch --symbol BTCUSDT --tf 1d --since 2017-08-17 --market spot
    python -m signal_lab.fetch --symbol BTCUSDT --tf 4h --extras                       # + oi, dvol, dxy, hmm_p_*
    → data/BTCUSDT_4h.parquet: ts, open, high, low, close, volume, taker_buy_volume, funding [, oi, dvol, dxy…, hmm_p_k]

Только публичные эндпоинты, без ключей. Время `ts` — открытие свечи (UTC, мс); признак бара известен на его закрытии
(ts + длина бара). Поэтому каждый внешний ряд присоединяется «последним значением, доступным к закрытию бара»
с учётом задержки публикации:
  oi   — data.binance.vision daily metrics, sum_open_interest (5-мин снимки; в live тот же ряд даёт openInterestHist);
  dvol — Deribit get_volatility_index_data, часовые свечи, значение на конец часа;
  dxy  — FRED DTWEXBGS (H.10): выходит раз в неделю по понедельникам за прошлую неделю → считаем доступным
         в среду 00:00 UTC следующей недели (с запасом на праздники);
  hmm_p_k — фильтрованные (forward) вероятности режимов HMM на дневных барах, переобучение на расширяющемся окне.
"""
from __future__ import annotations

import argparse
import io
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

TF_MS = {"15m": 900_000, "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "8h": 28_800_000,
         "12h": 43_200_000, "1d": 86_400_000}
DATA = Path("data")


def to_ms(t) -> pd.Series:
    """Время → миллисекунды эпохи UTC независимо от внутреннего разрешения datetime64 (с/мс/нс в разных pandas)."""
    t = pd.to_datetime(t, utc=True)
    return (t - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)


def fetch(symbol: str, tf: str, since: str, market: str = "um") -> pd.DataFrame:
    """Свечи Binance (фьючерсы USDⓈ-M или спот) с объёмом агрессивных покупок; для фьючерсов — ставка финансирования."""
    import ccxt
    ex = ccxt.binanceusdm({"enableRateLimit": True}) if market == "um" else ccxt.binance({"enableRateLimit": True})
    get_klines = ex.fapiPublicGetKlines if market == "um" else ex.publicGetKlines
    start = int(pd.Timestamp(since, tz="UTC").timestamp() * 1000)
    now = int(time.time() * 1000)
    rows = []
    t = start
    while t < now:
        # «сырые» свечи Binance: в них есть объём агрессивных покупок (индекс 9), которого нет в fetch_ohlcv
        batch = get_klines({"symbol": symbol, "interval": tf, "startTime": t, "limit": 1000})
        if not batch:
            break
        rows += batch
        t = int(batch[-1][0]) + TF_MS[tf]
        print(f"  свечи до {pd.Timestamp(t, unit='ms', tz='UTC')}", end="\r")
    df = pd.DataFrame(rows).iloc[:, [0, 1, 2, 3, 4, 5, 9]]
    df.columns = ["ts", "open", "high", "low", "close", "volume", "taker_buy_volume"]
    df = df.astype(float).drop_duplicates("ts")
    df["ts"] = df["ts"].astype("int64")
    df = df[df["ts"] + TF_MS[tf] <= now]                    # незакрытую последнюю свечу не берём

    if market != "um":
        return df.reset_index(drop=True)
    fr, t = [], start
    while t < now:
        batch = ex.fapiPublicGetFundingRate({"symbol": symbol, "startTime": t, "limit": 1000})
        if not batch:
            break
        fr += batch
        t = int(batch[-1]["fundingTime"]) + 1
    if fr:
        f = pd.DataFrame(fr)[["fundingTime", "fundingRate"]].astype(float)
        f.columns = ["ts", "funding"]
        f["ts"] = f["ts"].astype("int64")
        # ставка известна в момент начисления: присоединяем «последнюю известную» к каждому бару (без заглядывания вперёд)
        df = pd.merge_asof(df.sort_values("ts"), f.sort_values("ts"), on="ts", direction="backward")
    return df.reset_index(drop=True)


# ------------------------------------------------------------------ присоединение внешних рядов без заглядывания
def attach_asof(bars: pd.DataFrame, tf: str, series: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """К каждому бару — последнее значение ряда с available_ms ≤ закрытия бара (ts + длина бара).
    series: колонки available_ms (когда значение стало известно) + cols."""
    b = bars.drop(columns=[c for c in cols if c in bars], errors="ignore").copy()
    b["_close_ms"] = b["ts"].astype("int64") + TF_MS[tf]
    s = series.dropna(subset=cols, how="all").sort_values("available_ms")[["available_ms", *cols]]
    out = pd.merge_asof(b.sort_values("_close_ms"), s, left_on="_close_ms", right_on="available_ms", direction="backward")
    return out.drop(columns=["_close_ms", "available_ms"]).sort_values("ts").reset_index(drop=True)


# ------------------------------------------------------------------ open interest: data.binance.vision metrics
def _get(url: str, retries: int = 4):
    import requests
    for i in range(retries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 ** i)


def parse_metrics_csv(raw: bytes) -> pd.DataFrame:
    """Один дневной файл metrics → ts снимка (мс) и открытый интерес в монетах."""
    m = pd.read_csv(io.BytesIO(raw))
    out = pd.DataFrame({"available_ms": to_ms(m["create_time"]), "oi": m["sum_open_interest"].astype(float)})
    return out.drop_duplicates("available_ms")


def fetch_oi(symbol: str, start: str = "2020-01-01", cache: Path = DATA) -> pd.DataFrame:
    """5-минутный открытый интерес из дневных архивов. Кэш: data/<SYMBOL>_oi_5m.parquet (дописывается)."""
    from concurrent.futures import ThreadPoolExecutor
    path = cache / f"{symbol}_oi_5m.parquet"
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame({"available_ms": pd.Series(dtype="int64"),
                                                                        "oi": pd.Series(dtype="float64")})
    last = pd.Timestamp(int(have["available_ms"].max()), unit="ms", tz="UTC").normalize() if len(have) else None
    first = pd.Timestamp(start, tz="UTC") if last is None else last + pd.Timedelta(days=1)
    days = pd.date_range(first, pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1), freq="D")
    base = f"https://data.binance.vision/data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-"

    def one(d):
        r = _get(base + d.strftime("%Y-%m-%d") + ".zip")
        if r is None:
            return None
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            return parse_metrics_csv(z.read(z.namelist()[0]))

    with ThreadPoolExecutor(8) as pool:
        parts = [p for p in pool.map(one, days) if p is not None]
    print(f"  OI {symbol}: {len(parts)} новых дней из {len(days)}")
    out = pd.concat([have, *parts]).drop_duplicates("available_ms").sort_values("available_ms")
    out["available_ms"] = out["available_ms"].astype("int64")
    assert out["available_ms"].min() > 1e12, "время OI должно быть в миллисекундах"
    cache.mkdir(exist_ok=True)
    out.reset_index(drop=True).to_parquet(path, index=False)
    return out


# ------------------------------------------------------------------ DVOL (Deribit)
def fetch_dvol(currency: str, start: str = "2021-03-20") -> pd.DataFrame:
    """Часовые свечи индекса DVOL; значение (close) известно в конце часа."""
    import requests
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end = int(time.time() * 1000)
    rows = []
    while end > t0:
        r = requests.get("https://www.deribit.com/api/v2/public/get_volatility_index_data",
                         params={"currency": currency, "start_timestamp": t0, "end_timestamp": end, "resolution": 3600},
                         timeout=30).json()["result"]
        rows += r["data"]
        if not r.get("continuation") or not r["data"]:
            break
        end = int(r["continuation"])
    d = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "dvol"]).drop_duplicates("ts")
    d["available_ms"] = d["ts"].astype("int64") + 3_600_000
    return d[["available_ms", "dvol"]].sort_values("available_ms").reset_index(drop=True)


# ------------------------------------------------------------------ DXY (FRED DTWEXBGS, H.10)
def dxy_available_ms(dates: pd.Series) -> pd.Series:
    """Значение за день d (пн–пт недели W) публикуется в понедельник недели W+1 после 16:15 ET.
    Консервативно: доступно в среду 00:00 UTC недели W+1."""
    d = pd.to_datetime(dates, utc=True)
    monday = d.dt.normalize() - pd.to_timedelta(d.dt.dayofweek, unit="D")
    return to_ms(monday + pd.Timedelta(days=9))


def fetch_dxy() -> pd.DataFrame:
    import requests
    txt = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS", timeout=60).text
    d = pd.read_csv(io.StringIO(txt), na_values=".")
    d.columns = ["date", "dxy"]
    d = d.dropna()
    d["dxy_chg_5d"] = 100 * np.log(d["dxy"] / d["dxy"].shift(5))      # 5 рабочих дней, по опубликованным значениям
    d["available_ms"] = dxy_available_ms(d["date"])
    # один момент публикации на неделю: к нему доступны все значения недели — берём последнее
    return d.groupby("available_ms", as_index=False)[["dxy", "dxy_chg_5d"]].last()


# ------------------------------------------------------------------ F11: HMM forward-filter вероятности
def hmm_filtered_probs(daily: pd.DataFrame, n_states: int = 3, min_train_days: int = 365, refit_days: int = 30,
                       seed: int = 42) -> pd.DataFrame:
    """Вероятности режимов P(S_t | данные до t) на дневных барах. На каждой точке переобучения r: скейлер и
    GaussianHMM обучаются только на [0, r); затем forward-фильтр (без сглаживания) даёт вероятности для [r, r+refit).
    Состояния упорядочены по средней реализованной волатильности (hmm_p_0 — самый спокойный режим).
    Возвращает available_ms (закрытие дня) и hmm_p_0…hmm_p_{k−1}."""
    import sys
    import warnings
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hmmlearn.hmm import GaussianHMM
    from hmm_model import forward_filter
    from sklearn.preprocessing import StandardScaler

    c = daily["close"].astype(float)
    r = np.log(c).diff()
    feats = pd.DataFrame({"rv": r.rolling(20).std(), "vr": daily["volume"] / daily["volume"].rolling(20).mean(),
                          "hl": (daily["high"] - daily["low"]) / c}, index=daily.index)
    ok = feats.notna().all(axis=1).values
    X = feats.values
    T = len(X)
    P = np.full((T, n_states), np.nan)
    first = int(np.argmax(ok))
    for r0 in range(first + min_train_days, T, refit_days):
        tr = np.arange(first, r0)
        sc = StandardScaler().fit(X[tr])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200, random_state=seed, tol=1e-3)
            m.fit(sc.transform(X[tr]))
        order = np.argsort(m.means_[:, 0])                  # по волатильности (колонка rv)
        r1 = min(T, r0 + refit_days)
        post = forward_filter(m, sc.transform(X[first:r1]))[:, order]
        P[r0:r1] = post[r0 - first:r1 - first]
    out = pd.DataFrame(P, columns=[f"hmm_p_{k}" for k in range(n_states)])
    out["available_ms"] = daily["ts"].astype("int64").values + TF_MS["1d"]
    return out


def add_extras(df: pd.DataFrame, symbol: str, tf: str) -> pd.DataFrame:
    """oi, dvol (+ dvol недоступен для альтов), dxy, dxy_chg_5d, hmm_p_* — каждый с задержкой публикации."""
    df = attach_asof(df, tf, fetch_oi(symbol), ["oi"])
    cur = symbol.replace("USDT", "")
    if cur in ("BTC", "ETH"):
        df = attach_asof(df, tf, fetch_dvol(cur), ["dvol"])
    df = attach_asof(df, tf, fetch_dxy(), ["dxy", "dxy_chg_5d"])
    daily_path = DATA / f"{symbol}_1d.parquet"
    daily = pd.read_parquet(daily_path) if daily_path.exists() else fetch(symbol, "1d", "2019-09-01")
    hmm = hmm_filtered_probs(daily)
    df = attach_asof(df, tf, hmm, [c for c in hmm.columns if c.startswith("hmm_p_")])
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--tf", default="4h", choices=list(TF_MS))
    ap.add_argument("--since", default="2019-09-10")
    ap.add_argument("--market", default="um", choices=["um", "spot"])
    ap.add_argument("--extras", action="store_true", help="добавить oi, dvol, dxy, hmm_p_* к существующему файлу")
    a = ap.parse_args()
    out = DATA / f"{a.symbol}{'_spot' if a.market == 'spot' else ''}_{a.tf}.parquet"
    if a.extras:
        df = add_extras(pd.read_parquet(out), a.symbol, a.tf)
    else:
        df = fetch(a.symbol, a.tf, a.since, a.market)
    out.parent.mkdir(exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\n{len(df)} баров, колонки {list(df.columns)} → {out}")


if __name__ == "__main__":
    main()
