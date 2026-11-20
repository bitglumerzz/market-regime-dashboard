"""H1 — управление размером позиции по волатильности (docs/research/directional-edge-2026-09.md, раздел 4).

    python -m signal_lab.h1 --data data/BTCUSDT_spot_1d.parquet

Спецификация (зафиксирована до прогона): long-only, вес w_t = min(1, σ_цель / σ̂_20д), σ_цель = 25 % годовых,
σ̂_20д — стандартное отклонение дневных лог-доходностей за 20 дней × √365, известное на закрытии дня t;
вес применяется к доходности t → t+1. Ребаланс только если целевой вес отличается от текущего больше чем на 15 %
(«полосы»). Издержки 5 б.п. за единицу оборота (taker Binance). Сравнение с buy & hold на всех скользящих 5-летних
окнах (сдвиг — месяц) до начала holdout. Успех: Sharpe выше и максимальная просадка ниже на всех окнах.
Это базовый слой риск-менеджмента, а не преимущество в направлении.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .registry import HOLDOUT_START, before_holdout, n_trials, register
from .run import load
from .stats import stationary_bootstrap_ci

ANN = 365


def vol_target_weights(close: pd.Series, target: float = 0.25, window: int = 20, band: float = 0.15) -> pd.Series:
    """Вес на закрытии дня t (применяется к доходности t→t+1). Ребаланс только за пределами полосы band."""
    r = np.log(close).diff()
    sig = r.rolling(window).std() * math.sqrt(ANN)
    tgt = np.minimum(1.0, target / sig).values
    w = np.full(len(tgt), np.nan)
    cur = np.nan
    for t, x in enumerate(tgt):
        if np.isnan(x):
            continue
        if np.isnan(cur) or abs(x - cur) > band * max(cur, 1e-9):
            cur = x
        w[t] = cur
    return pd.Series(w, index=close.index)


def strategy_returns(close: pd.Series, w: pd.Series, fee_bps: float = 5.0) -> pd.Series:
    """Дневная простая доходность: w_{t−1}·r_t − издержки за оборот |w_t − w_{t−1}|."""
    r = close.pct_change()
    wl = w.shift(1)
    turn = w.diff().abs().fillna(w.abs())
    return (wl * r - turn.shift(1).fillna(0) * fee_bps / 1e4).dropna()


def perf(x: pd.Series) -> dict:
    eq = (1 + x).cumprod()
    return {"sharpe": float(x.mean() / x.std() * math.sqrt(ANN)) if x.std() > 0 else 0.0,
            "cagr": float(eq.iloc[-1] ** (ANN / len(x)) - 1), "vol": float(x.std() * math.sqrt(ANN)),
            "mdd": float((1 - eq / eq.cummax()).max())}


def rolling_windows(bh: pd.Series, vm: pd.Series, years: int = 5) -> pd.DataFrame:
    rows = []
    starts = pd.date_range(bh.index[0].normalize(), bh.index[-1] - pd.DateOffset(years=years), freq="MS", tz="UTC")
    for s in starts:
        e = s + pd.DateOffset(years=years)
        a, b = bh[(bh.index >= s) & (bh.index < e)], vm[(vm.index >= s) & (vm.index < e)]
        pa, pb = perf(a), perf(b)
        rows.append({"start": s.date(), "end": e.date(), "sharpe_bh": pa["sharpe"], "sharpe_vm": pb["sharpe"],
                     "mdd_bh": pa["mdd"], "mdd_vm": pb["mdd"]})
    return pd.DataFrame(rows)


def evaluate_h1(df: pd.DataFrame, target: float = 0.25, fee_bps: float = 5.0) -> dict:
    df = before_holdout(df)
    c = df["close"]
    w = vol_target_weights(c, target)
    vm = strategy_returns(c, w, fee_bps)
    bh = c.pct_change().loc[vm.index]
    win = rolling_windows(bh, vm)
    pair = np.c_[bh.values, vm.values]                      # CI разницы Sharpe — бутстрап пар дней
    rng = np.random.default_rng(0)
    boots = []
    n = len(pair)
    for _ in range(1000):
        idx = np.empty(n, int)
        i = rng.integers(n)
        for t in range(n):
            idx[t] = i
            i = rng.integers(n) if rng.random() < 1 / 20 else (i + 1) % n
        s = pair[idx]
        boots.append(s[:, 1].mean() / s[:, 1].std() - s[:, 0].mean() / s[:, 0].std())
    lo, hi = np.quantile(boots, [0.025, 0.975]) * math.sqrt(ANN)
    ok = bool(len(win) and (win.sharpe_vm > win.sharpe_bh).all() and (win.mdd_vm < win.mdd_bh).all())
    return {"bh": perf(bh), "vm": perf(vm), "windows": win, "ok": ok, "dsharpe_ci": (float(lo), float(hi)),
            "turnover": float(w.diff().abs().sum() / (len(w) / ANN)), "mean_w": float(w.mean()),
            "period": (vm.index[0], vm.index[-1]), "vm_rets": vm.values, "target": target, "fee_bps": fee_bps}


def report_h1(res: dict, name: str) -> str:
    b, v, W = res["bh"], res["vm"], res["windows"]
    L = [f"# H1 — размер позиции по волатильности · {name}", "",
         f"Период {res['period'][0].date()} — {res['period'][1].date()} (до holdout {HOLDOUT_START.date()}). "
         f"σ_цель {res['target']:.0%}, σ̂ за 20 дней, полосы ребаланса 15 %, издержки {res['fee_bps']:g} б.п. за оборот. "
         f"Средний вес {res['mean_w']:.2f}, оборот {res['turnover']:.1f}× в год.", "",
         "| стратегия | Sharpe | CAGR | волатильность | макс. просадка |", "|---|---|---|---|---|",
         f"| buy & hold | {b['sharpe']:.2f} | {b['cagr']:+.1%} | {b['vol']:.1%} | {b['mdd']:.1%} |",
         f"| long, размер по волатильности | {v['sharpe']:.2f} | {v['cagr']:+.1%} | {v['vol']:.1%} | {v['mdd']:.1%} |", "",
         f"Разница Sharpe (vol-managed − B&H), 95 % ДИ стационарного бутстрапа: [{res['dsharpe_ci'][0]:+.2f}; "
         f"{res['dsharpe_ci'][1]:+.2f}].", "",
         f"## Скользящие 5-летние окна ({len(W)} шт., сдвиг месяц)", "",
         f"- Sharpe выше у vol-managed: **{int((W.sharpe_vm > W.sharpe_bh).sum())} из {len(W)}** "
         f"(мин. разница {(W.sharpe_vm - W.sharpe_bh).min():+.2f}, макс. {(W.sharpe_vm - W.sharpe_bh).max():+.2f});",
         f"- просадка ниже у vol-managed: **{int((W.mdd_vm < W.mdd_bh).sum())} из {len(W)}** "
         f"(просадка vol-managed {W.mdd_vm.min():.0%}–{W.mdd_vm.max():.0%} против B&H {W.mdd_bh.min():.0%}–{W.mdd_bh.max():.0%}).", "",
         "## Вердикт", "",
         f"**H1 {'подтверждена' if res['ok'] else 'не подтверждена'}** по критерию «Sharpe выше и просадка ниже на всех "
         "5-летних окнах». Это слой управления риском: он не предсказывает направление и не является «преимуществом»."]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", type=float, default=0.25)
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args()
    name = Path(a.data).stem
    res = evaluate_h1(load(a.data), a.target)
    rep = report_h1(res, name)
    if not a.no_register:
        from .registry import trade_moments
        v = res["vm"]
        register([{"hypothesis": "H1", "asset": name.split("_")[0], "tf": "1d", "model": "vol-target",
                   "features": "sigma_20d", "barriers": "", "horizon_h": 24, "rule": f"w=min(1,{a.target}/σ20) band15%",
                   "period_start": res["period"][0], "period_end": res["period"][1], "holdout": False,
                   "mean_bps": float(np.mean(res["vm_rets"]) * 1e4), **trade_moments(res["vm_rets"]),
                   "n_trials_at_reg": n_trials() + 1, "verdict": "подтверждена" if res["ok"] else "не подтверждена",
                   "notes": f"Sharpe {v['sharpe']:.2f} vs B&H {res['bh']['sharpe']:.2f}; MDD {v['mdd']:.1%} vs {res['bh']['mdd']:.1%}; "
                            f"{name}"}])
    out = Path("reports") / f"H1_{name}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(rep, encoding="utf-8")
    print(rep, f"\n\nСохранено: {out}", sep="")


if __name__ == "__main__":
    main()
