"""Загрузка реальных данных с Binance USDⓈ-M (фьючерсы) для лаборатории — запускать на своём компьютере.

    python -m signal_lab.fetch --symbol BTCUSDT --tf 4h --since 2019-09-01
    → data/BTCUSDT_4h.parquet: ts, open, high, low, close, volume, taker_buy_volume, funding

Только публичные эндпоинты, без ключей. Если Binance недоступен из вашей сети — используйте bulk-архивы
data.binance.vision (см. промт 02) или VPS вне РФ.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

TF_MS = {"15m": 900_000, "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "8h": 28_800_000,
         "12h": 43_200_000, "1d": 86_400_000}


def fetch(symbol: str, tf: str, since: str) -> pd.DataFrame:
    import ccxt
    ex = ccxt.binanceusdm({"enableRateLimit": True})
    start = int(pd.Timestamp(since, tz="UTC").timestamp() * 1000)
    now = int(time.time() * 1000)
    rows = []
    t = start
    while t < now:
        # «сырые» свечи Binance: в них есть объём агрессивных покупок (индекс 9), которого нет в fetch_ohlcv
        batch = ex.fapiPublicGetKlines({"symbol": symbol, "interval": tf, "startTime": t, "limit": 1500})
        if not batch:
            break
        rows += batch
        t = int(batch[-1][0]) + TF_MS[tf]
        print(f"  свечи до {pd.Timestamp(t, unit='ms', tz='UTC')}", end="\r")
    df = pd.DataFrame(rows).iloc[:, [0, 1, 2, 3, 4, 5, 9]]
    df.columns = ["ts", "open", "high", "low", "close", "volume", "taker_buy_volume"]
    df = df.astype(float).drop_duplicates("ts")
    df["ts"] = df["ts"].astype("int64")

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
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--tf", default="4h", choices=list(TF_MS))
    ap.add_argument("--since", default="2019-09-10")
    a = ap.parse_args()
    df = fetch(a.symbol, a.tf, a.since)
    out = Path("data") / f"{a.symbol}_{a.tf}.parquet"
    out.parent.mkdir(exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\n{len(df)} баров → {out}")


if __name__ == "__main__":
    main()
