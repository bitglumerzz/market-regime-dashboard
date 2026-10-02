"""H-FlyForage — алгоритм поиска пищи дрозофилы (частота и свежесть дискретных встреч с запахом) как правило
принятия торговых решений. НЕ использует коннектом/LIF/FlyHash — это отдельный, поведенческий механизм дрозофилы
(см. docs/research/experiments/H-FlyForage.md). Параметры (τ, floor, k) зафиксированы калибровкой на чистом шуме
(GBM без сноса, seed 0) ДО единого взгляда на рыночные данные — здесь менять нельзя, иначе это новое испытание.

    python -m signal_lab.fly_forage --data data/SOLUSDT_4h.parquet --config stoch

«Встреча» — момент пробоя одного из 9 окон Дончиана (зеркально: long — как в h2.py, short — зеркальный канал по
минимумам). Экспоненциальная память копит частоту/свежесть встреч по каждой стороне; решение — по одной из трёх
конфигураций (F-stoch/F-det/F-cont, см. §2.4 спецификации), принимается только на барах-встречах (кроме F-cont).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import deflated_sharpe
from .h2 import WINDOWS, donchian_state
from .registry import HOLDOUT_START, before_holdout, n_trials, register, trade_moments
from .run import BARS_PER_DAY, load
from .stats import min_track_record, psr, stationary_bootstrap_ci

# Зафиксировано калибровкой на шуме (§3 H-FlyForage.md) — НЕ подбирать по рынку.
TAU_BARS = 120.0
FLOOR_BUSY = 6.94
K_SENS = 0.328


def short_state(close: pd.Series, n: int) -> pd.Series:
    """Зеркало h2.donchian_state (версия v1) для short: канал по минимумам, стоп-храповик подтягивается ВНИЗ."""
    c = close.values
    hi = close.rolling(n).max().values
    lo = close.rolling(n).min().values
    pos, stop = 0, np.inf
    out = np.zeros(len(c))
    for t in range(len(c)):
        if np.isnan(hi[t]):
            continue
        mid = (hi[t] + lo[t]) / 2
        exited = False
        if pos == 1:
            if c[t] >= stop:
                pos, exited = 0, True
            else:
                stop = min(stop, mid)
        if pos == 0 and not exited and c[t] <= lo[t]:
            pos, stop = 1, mid
        out[t] = pos
    s = pd.Series(out, index=close.index)
    s[np.isnan(hi)] = np.nan
    return s


def encounter_counts(close: pd.Series, bars_per_day: int, windows_days=WINDOWS) -> tuple[np.ndarray, np.ndarray]:
    """count_long(t), count_short(t): сколько окон именно на баре t впервые вошли в пробой (переход 0→1)."""
    Ls = [max(2, round(d * bars_per_day)) for d in windows_days]
    long_df = pd.concat([donchian_state(close, L, "mid_ratchet", "v1") for L in Ls], axis=1)
    short_df = pd.concat([short_state(close, L) for L in Ls], axis=1)
    count_long = long_df.diff().clip(lower=0).sum(axis=1).fillna(0).values
    count_short = short_df.diff().clip(lower=0).sum(axis=1).fillna(0).values
    valid = long_df.notna().all(axis=1).values & short_df.notna().all(axis=1).values
    return count_long, count_short, valid


def memory_traces(count_long: np.ndarray, count_short: np.ndarray, tau: float = TAU_BARS) -> tuple[np.ndarray, np.ndarray]:
    """e_long(t), e_short(t): экспоненциально забывающая память встреч, decay = exp(-1/tau). Каузально (цикл по t)."""
    decay = math.exp(-1.0 / tau)
    n = len(count_long)
    e_l, e_s = np.zeros(n), np.zeros(n)
    for t in range(1, n):
        e_l[t] = decay * e_l[t - 1] + count_long[t]
        e_s[t] = decay * e_s[t - 1] + count_short[t]
    return e_l, e_s


def decide(e_long: np.ndarray, e_short: np.ndarray, count_long: np.ndarray, count_short: np.ndarray, config: str,
          floor: float = FLOOR_BUSY, k: float = K_SENS, seed: int = 0) -> np.ndarray:
    """Сторона на каждом баре (+1/−1/0), удерживается между решениями. config ∈ {'stoch','det'} (§2.4).

    Между встречами e_long и e_short затухают с одним decay, их разность (bias) тоже затухает к нулю, но не меняет
    знак — знак bias может перевернуться только в момент новой встречи. Поэтому при детерминированном правиле
    sign(bias) «решать каждый бар» и «решать только в момент встречи» дают один и тот же результат (проверено
    тестом); момент решения содержателен только вместе со стохастикой — отдельного config 'cont' нет."""
    bias = e_long - e_short
    busy = e_long + e_short
    in_search = busy < floor
    is_encounter = (count_long + count_short) > 0
    rng = np.random.default_rng(seed)
    p_long = 1.0 / (1.0 + np.exp(-k * bias))
    side = np.zeros(len(bias))
    cur = 0.0
    for t in range(len(bias)):
        if in_search[t]:
            cur = 0.0
        elif config == "stoch":
            if is_encounter[t]:
                cur = 1.0 if rng.random() < p_long[t] else -1.0
        elif config == "det":                            # детерминированно, знаку bias можно верить каждый бар
            cur = 1.0 if bias[t] > 0 else (-1.0 if bias[t] < 0 else cur)
        else:
            raise ValueError(config)
        side[t] = cur
    return side


def strat_returns(close: pd.Series, side: np.ndarray, fee_bps: float = 5.0) -> pd.Series:
    """Та же конвенция, что h1.py/h2.py/apex_cloud.py: решение на закрытии t работает на доходность бара t+1."""
    r = close.pct_change()
    s = pd.Series(side, index=close.index).shift(1).fillna(0.0)
    turn = s.diff().abs().fillna(s.abs())
    return (s * r - turn * fee_bps / 1e4).dropna()


def trade_returns(close: pd.Series, side: np.ndarray, fee_bps: float = 5.0) -> np.ndarray:
    """Доходность по сделкам целиком (между переворотами), как в apex_cloud.py — единица для PSR/DSR/бутстрапа."""
    s = pd.Series(side, index=close.index).shift(1).fillna(0.0).values
    o, c = close.values, close.values  # вход по открытию следующего бара аппроксимируем ценой закрытия решающего бара
    n = len(s)
    changes = np.where(np.diff(np.r_[0.0, s]) != 0)[0]
    changes = np.r_[changes, n]
    rets = []
    for a, b in zip(changes[:-1], changes[1:]):
        side_ab = s[a]
        if side_ab == 0 or a + 1 >= n:
            continue
        entry = close.values[a + 1]
        exit_ = close.values[b] if b < n else c[-1]
        if not np.isfinite(entry) or not np.isfinite(exit_) or entry <= 0 or exit_ <= 0:
            continue
        cost = 2 * fee_bps / 1e4
        rets.append(side_ab * np.log(exit_ / entry) - cost)
    return np.array(rets)


def run(df: pd.DataFrame, tf: str, config: str, fee_bps: float = 5.0, seed: int = 0,
       cutoff: pd.Timestamp | None = HOLDOUT_START) -> dict:
    if cutoff is not None and isinstance(df.index, pd.DatetimeIndex):
        df = before_holdout(df, cutoff)
    bpd = BARS_PER_DAY[tf]
    cl, cs, valid = encounter_counts(df["close"], bpd)
    el, es = memory_traces(cl, cs)
    side = decide(el, es, cl, cs, config, seed=seed)
    side = np.where(valid, side, 0.0)
    bar_rets = strat_returns(df["close"], side, fee_bps)
    trades = trade_returns(df["close"], side, fee_bps)
    bh = df["close"].pct_change().dropna()
    bh = bh[bh.index.isin(bar_rets.index)]
    stats = {}
    if len(trades) >= 10:
        lo, hi = stationary_bootstrap_ci(trades, block=3.0, n_boot=1000)
        stats = {"n_trades": len(trades), "hit": float(np.mean(trades > 0)), "mean_bps": float(trades.mean() * 1e4),
                 "ci_lo_bps": lo * 1e4, "ci_hi_bps": hi * 1e4, "psr": psr(trades), "min_trl": min_track_record(trades)}
    return {"config": config, "side": side, "bar_rets": bar_rets, "trades": trades, "stats": stats,
            "bh_mean_bps": float(bh.mean() * 1e4) if len(bh) else float("nan"), "busy_frac": float(np.mean(el + es >= FLOOR_BUSY))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tf", default="4h", choices=list(BARS_PER_DAY))
    ap.add_argument("--config", default="stoch", choices=["stoch", "det"])
    ap.add_argument("--fee-bps", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args()
    name = Path(a.data).stem
    res = run(load(a.data), a.tf, a.config, a.fee_bps, a.seed)
    s = res["stats"]
    print(f"{name} [{a.config}] seed={a.seed}: trades={s.get('n_trades',0)} hit={s.get('hit',float('nan')):.2f} "
          f"mean={s.get('mean_bps',float('nan')):+.1f}bps CI=[{s.get('ci_lo_bps',float('nan')):+.0f};{s.get('ci_hi_bps',float('nan')):+.0f}] "
          f"vs B&H={res['bh_mean_bps']:+.1f}bps busy={res['busy_frac']:.2f}")
    if not a.no_register and s:
        N = n_trials() + 1
        register([{"hypothesis": "H-FlyForage", "asset": name.split("_")[0], "tf": a.tf, "model": "forage-encounter",
                   "rule": f"{a.config} seed={a.seed} tau={TAU_BARS} floor={FLOOR_BUSY} k={K_SENS}",
                   "holdout": False, "n_trades": s["n_trades"], "hit": s["hit"], "mean_bps": s["mean_bps"],
                   "ci_lo_bps": s["ci_lo_bps"], "ci_hi_bps": s["ci_hi_bps"], **trade_moments(res["trades"]),
                   "n_trials_at_reg": N, "dsr_at_reg": deflated_sharpe(res["trades"], N)}])


if __name__ == "__main__":
    main()
