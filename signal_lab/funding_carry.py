"""H-FundingCarry — дельта-нейтральный арбитраж ставки финансирования (лонг спот + шорт перпетуал).
Правила, комиссии и критерии зафиксированы в docs/research/experiments/H-FundingCarry.md до кода.

    python -m signal_lab.funding_carry --fetch            # сырые выплаты funding + спот 4h для топ-20
    python -m signal_lab.funding_carry                    # период разработки, регистрация в реестр
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import deflated_sharpe
from .registry import HOLDOUT_START, n_trials, open_holdout, register, trade_moments
from .stats import stationary_bootstrap_ci

DATA = Path("data")
BAR_MS = 14_400_000
FEE_SIDE = 15e-4          # спот taker 10 б.п. + перп taker 5 б.п. — на вход и отдельно на выход
CAPITAL_PER_NOTIONAL = 1.5
FILTER_EVENTS = 21
SINCE = "2019-09-01"
TOP20 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "HYPEUSDT", "SUIUSDT", "LINKUSDT",
         "BCHUSDT", "AVAXUSDT", "HBARUSDT", "LTCUSDT", "1000PEPEUSDT", "AAVEUSDT", "ICPUSDT", "APTUSDT", "NEARUSDT",
         "ETCUSDT", "ONDOUSDT"]
SPOT_NAME = {"1000PEPEUSDT": "PEPEUSDT"}
HOLDOUT_GROUP = "carry-2025"
HOLDOUT_END = pd.Timestamp("2026-10-02", tz="UTC")       # конец периода зафиксирован в пре-регистрации


def fetch_funding_raw(symbol: str, since: str = SINCE) -> pd.DataFrame:
    """Каждое событие выплаты — одна строка (ts выплаты, мс; ставка за интервал)."""
    import ccxt
    ex = ccxt.binanceusdm({"enableRateLimit": True})
    t, now, rows = int(pd.Timestamp(since, tz="UTC").timestamp() * 1000), int(time.time() * 1000), []
    while t < now:
        batch = ex.fapiPublicGetFundingRate({"symbol": symbol, "startTime": t, "limit": 1000})
        if not batch:
            break
        rows += batch
        t = int(batch[-1]["fundingTime"]) + 1
    f = pd.DataFrame(rows)[["fundingTime", "fundingRate"]].astype(float)
    f.columns = ["ts", "rate"]
    f["ts"] = f["ts"].astype("int64")
    return f.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def _bars(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    return pd.Series(df["close"].astype(float).values, index=df["ts"].astype("int64").values)


def align(perp: pd.Series, spot: pd.Series, funding: pd.DataFrame) -> pd.DataFrame:
    """4h-бары, общие для обеих ног. Выплата в момент T начисляется бару, который ЗАКАНЧИВАЕТСЯ в T
    (её получает тот, кто держал позицию до T включительно); выплаты внутри бара тоже относятся к нему."""
    idx = perp.index.intersection(spot.index).sort_values()
    df = pd.DataFrame({"perp": perp.reindex(idx), "spot": spot.reindex(idx)}, index=idx)
    bar_of_event = ((funding["ts"].values - 1) // BAR_MS) * BAR_MS
    fb = pd.Series(funding["rate"].values, index=bar_of_event).groupby(level=0).sum()
    df["fund"] = fb.reindex(idx).fillna(0.0)
    # сигнал фильтра: средняя по последним 21 событию, известна в момент события → действует со следующего бара
    sig = funding["rate"].rolling(FILTER_EVENTS).mean()
    s = pd.Series(sig.values, index=bar_of_event + BAR_MS).groupby(level=0).last()
    df["sig"] = s.reindex(idx).ffill()
    return df


def simulate(df: pd.DataFrame, rule: str) -> pd.Series:
    """Доходность на капитал за каждый бар. Позиция на баре t решена до начала бара t."""
    r_s = df["spot"].pct_change().fillna(0.0)
    r_p = df["perp"].pct_change().fillna(0.0)
    if rule == "static":
        pos = pd.Series(1.0, index=df.index)
    elif rule == "filter":
        pos = (df["sig"] > 0).astype(float)
    else:
        raise ValueError(rule)
    turn = pos.diff().abs().fillna(pos.abs())
    turn.iloc[-1] += pos.iloc[-1]                     # закрытие позиции в конце периода
    pnl = pos * (r_s - r_p + df["fund"]) - turn * FEE_SIDE
    return pnl / CAPITAL_PER_NOTIONAL


def evaluate(ret: pd.Series, price: pd.Series, n_for_dsr: int) -> dict:
    t = pd.to_datetime(ret.index, unit="ms", utc=True)
    r = pd.Series(ret.values, index=t)
    monthly = (1 + r).resample("ME").prod() - 1
    eq = (1 + r).cumprod()
    years = (t[-1] - t[0]).days / 365.25
    lo, hi = stationary_bootstrap_ci(monthly.values, block=3.0, n_boot=2000)
    p = pd.Series(price.values, index=t)
    rise30 = (p.rolling("30D").max() / p.rolling("30D").min() - 1).max()
    return {"years": years, "ann": eq.iloc[-1] ** (1 / years) - 1, "m_mean": monthly.mean(), "ci_lo": lo, "ci_hi": hi,
            "maxdd": (eq / eq.cummax() - 1).min(), "neg_months": float((monthly < 0).mean()),
            "dsr": deflated_sharpe(monthly.values, n_for_dsr), "rise30": rise30, "monthly": monthly}


def passes(e: dict) -> bool:
    return e["ann"] > 0.04 and e["ci_lo"] > 0 and e["dsr"] > 0.95 and e["maxdd"] > -0.10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--holdout", action="store_true", help="однократная проверка на группе carry-2025")
    a = ap.parse_args()
    if a.fetch:
        from .fetch import fetch
        for sym in TOP20:
            spot = SPOT_NAME.get(sym, sym)
            try:
                fetch_funding_raw(sym).to_parquet(DATA / f"{sym}_funding.parquet", index=False)
                fetch(spot, "4h", SINCE, market="spot").to_parquet(DATA / f"{spot}_spot_4h.parquet", index=False)
                print(f"{sym}: ok", flush=True)
            except Exception as e:  # нет спота/перпа на Binance — актив исключается до расчёта
                print(f"{sym}: ИСКЛЮЧЁН ({type(e).__name__}: {str(e)[:80]})", flush=True)
        return

    cutoff = int(HOLDOUT_START.timestamp() * 1000)
    end = int(HOLDOUT_END.timestamp() * 1000)
    if a.holdout and not a.no_register:
        open_holdout("H-FundingCarry: однократная проверка правил static/filter без изменений", group=HOLDOUT_GROUP)
    N = n_trials() + 2 * len(TOP20)
    rows = []
    for sym in TOP20:
        spot = SPOT_NAME.get(sym, sym)
        paths = [DATA / f"{sym}_4h.parquet", DATA / f"{spot}_spot_4h.parquet", DATA / f"{sym}_funding.parquet"]
        if not all(p.exists() for p in paths):
            print(f"{sym:13s} нет данных — исключён")
            continue
        df = align(_bars(paths[0]), _bars(paths[1]), pd.read_parquet(paths[2]))
        dev = df[df.index < cutoff]
        if a.holdout:
            df = df[(df.index >= cutoff) & (df.index < end)]
        else:
            df = dev
        if len(dev) < 6 * 365:
            print(f"{sym:13s} меньше года истории до {HOLDOUT_START.date()} ({len(dev)} баров) — исключён")
            continue
        for rule in ("static", "filter"):
            e = evaluate(simulate(df, rule), df["perp"], N)
            ok = passes(e)
            print(f"{sym:13s} {rule:6s} {e['years']:4.1f}л  год {e['ann']*100:+6.1f}%  мес {e['m_mean']*100:+5.2f}% "
                  f"ДИ[{e['ci_lo']*100:+.2f};{e['ci_hi']*100:+.2f}]  DSR {e['dsr']:.3f}  DD {e['maxdd']*100:+5.1f}%  "
                  f"мес<0 {e['neg_months']*100:3.0f}%  рост30д {e['rise30']*100:4.0f}%  {'ПРОШЁЛ' if ok else '—'}")
            m = e["monthly"].values
            rows.append({"hypothesis": "H-FundingCarry", "asset": sym, "tf": "4h", "model": "cash-and-carry",
                         "rule": f"{rule} fee={FEE_SIDE*1e4:.0f}bps/side cap={CAPITAL_PER_NOTIONAL}",
                         "period_start": HOLDOUT_START.date().isoformat() if a.holdout else "",
                         "period_end": (HOLDOUT_END if a.holdout else HOLDOUT_START).date().isoformat(),
                         "holdout": a.holdout, "n_trades": len(m),
                         "hit": float((m > 0).mean()), "mean_bps": float(m.mean() * 1e4),
                         "ci_lo_bps": e["ci_lo"] * 1e4, "ci_hi_bps": e["ci_hi"] * 1e4, **trade_moments(m),
                         "n_trials_at_reg": N, "dsr_at_reg": e["dsr"], "verdict": "прошёл" if ok else "не прошёл",
                         "notes": f"единица — месяц; год {e['ann']*100:+.1f}%, maxDD {e['maxdd']*100:.1f}%"})
    if rows and not a.no_register:
        register(rows)
        print(f"зарегистрировано {len(rows)} строк")


if __name__ == "__main__":
    main()
