"""H2 — ансамбль пробоев канала Дончиана, реплика Zarattini, Pagani, Barbon (2025), SSRN 5209907.

    python -m signal_lab.h2 --data data/BTCUSDT_spot_1d.parquet               # точная версия статьи (v1)
    python -m signal_lab.h2 --data data/BTCUSDT_spot_1d.parquet --version v0  # первая версия (воспроизводимость реестра)

Версия v1 — по PDF статьи (§4.1–4.3; сверка: docs/research/directional-edge-2026-09-verification.md):
  • окна n ∈ {5, 10, 20, 30, 60, 90, 150, 250, 360} дней; канал по ЗАКРЫТИЯМ, сегодняшнее закрытие включено;
  • вход модели n: Close_t = max(Close_{t−n+1..t}); выход: Close_t ≤ TS_t (выход проверяется раньше входа);
  • TS_{t+1} = max(TS_t, Mid_t), при входе TS = Mid_t; Mid = (max + min)/2;
  • вес модели: min(0.25 / σ90_t, 2.0) × Pos_n; σ90 — выборочное ст. откл. простых дневных доходностей за 90 дней × √365;
  • ансамбль = среднее весов 9 моделей; вес закрытия t работает на день t+1;
  • порог 20 % (относительно текущего веса ансамбля) — только для изменений из-за волатильности;
    вход/выход любой модели исполняется сразу; издержки 10 б.п. × |Δw|.
  Не указано в статье (наши допущения): простые доходности, √365, выход раньше входа, порог относительный к весу
  ансамбля, издержки на |Δw|. Данные — Binance spot, а не агрегат CoinMarketCap.
Версия v0 (первый прогон, до сверки со статьёй): канал по n предыдущим закрытиям без сегодняшнего, вход при
  Close_t > max, выход при Close_t < TS, кэп 1.0, лог-доходности для σ, порог 20 % на все изменения веса.
Испытания (N ≥ 27): 9 одиночных окон × 3 варианта выхода + 3 ансамбля — все в реестре.
Варианты выхода: mid_ratchet (статья), mid_plain (выход под текущей серединой, без храповика), lower (выход под
минимумом канала).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import deflated_sharpe
from .h1 import perf
from .registry import HOLDOUT_START, before_holdout, n_trials, register, trade_moments
from .run import load

WINDOWS = (5, 10, 20, 30, 60, 90, 150, 250, 360)
EXITS = ("mid_ratchet", "mid_plain", "lower")
ANN = 365


def donchian_state(close: pd.Series, n: int, exit_rule: str = "mid_ratchet", version: str = "v1") -> pd.Series:
    """1, если модель n в позиции на закрытии дня t (решение по данным до t включительно), иначе 0."""
    c = close.values
    src = close if version == "v1" else close.shift(1)          # v1: канал включает сегодняшнее закрытие
    hi = src.rolling(n).max().values
    lo = src.rolling(n).min().values
    below = (lambda x, lvl: x <= lvl) if version == "v1" else (lambda x, lvl: x < lvl)
    pos, stop = 0, -np.inf
    out = np.zeros(len(c))
    for t in range(len(c)):
        if np.isnan(hi[t]):
            continue
        mid = (hi[t] + lo[t]) / 2
        exited = False
        if pos == 1:
            if exit_rule == "mid_ratchet":
                if below(c[t], stop):
                    pos, exited = 0, True
                else:
                    stop = max(stop, mid)                          # стоп на завтра
            elif exit_rule == "mid_plain":
                if below(c[t], mid):
                    pos, exited = 0, True
            elif exit_rule == "lower":
                if below(c[t], lo[t]):
                    pos, exited = 0, True
            else:
                raise ValueError(exit_rule)
        entry = c[t] >= hi[t] if version == "v1" else c[t] > hi[t]
        if pos == 0 and not exited and entry:
            pos, stop = 1, mid
        out[t] = pos
    s = pd.Series(out, index=close.index)
    s[np.isnan(hi)] = np.nan
    return s


def vol_scale(close: pd.Series, target: float = 0.25, window: int = 90, version: str = "v1") -> pd.Series:
    r = close.pct_change() if version == "v1" else np.log(close).diff()
    sig = r.rolling(window).std() * math.sqrt(ANN)
    return np.minimum(2.0 if version == "v1" else 1.0, target / sig)


def apply_band(w: np.ndarray, band: float = 0.20, force: np.ndarray | None = None) -> np.ndarray:
    """Ребаланс только при относительном изменении > band, при входе/выходе (переход через ноль)
    или в дни force (изменилась позиция хотя бы одной модели)."""
    out = np.full(len(w), np.nan)
    cur = np.nan
    for t, x in enumerate(w):
        if np.isnan(x):
            continue
        if (np.isnan(cur) or (x == 0) != (cur == 0) or (cur > 0 and abs(x - cur) > band * cur)
                or (force is not None and force[t])):
            cur = x
        out[t] = cur
    return out


def model_weights(close: pd.Series, windows=WINDOWS, exit_rule="mid_ratchet", target=0.25, band=0.20,
                  version: str = "v1") -> pd.Series:
    vs = vol_scale(close, target, 90, version)
    states = pd.concat([donchian_state(close, n, exit_rule, version) for n in windows], axis=1)
    raw = states.mul(vs, axis=0).mean(axis=1, skipna=False)    # старт, когда готово самое длинное окно
    force = None
    if version == "v1":                                        # сигнальные изменения — сразу, волатильность — с порогом
        force = states.diff().abs().sum(axis=1).fillna(0).values > 0
    return pd.Series(apply_band(raw.values, band, force), index=close.index)


def strat_returns(close: pd.Series, w: pd.Series, fee_bps: float = 10.0) -> pd.Series:
    r = close.pct_change()
    w0 = w.fillna(0.0)
    turn = w0.diff().abs().fillna(w0.abs())                 # оборот на закрытии t, издержки списываем с дня t+1
    x = w0.shift(1) * r - turn.shift(1).fillna(0) * fee_bps / 1e4
    return x[w.shift(1).notna()]


def evaluate_h2(df: pd.DataFrame, holdout: bool = False, fee_bps: float = 10.0, n_prior: int = 0,
                version: str = "v1") -> dict:
    """holdout=False: только данные до HOLDOUT_START. holdout=True: состояние считается по всей истории
    (стопам нужна предыстория), результаты — только на отложенном периоде."""
    d = df if holdout else before_holdout(df)
    c = d["close"]
    specs = [("ансамбль", WINDOWS, e) for e in EXITS] + [(f"n={n}", (n,), e) for n in WINDOWS for e in EXITS]
    res = []
    for name, wins, e in specs:
        wts = model_weights(c, wins, e, version=version)
        x = strat_returns(c, wts, fee_bps)
        if holdout:
            x = x[x.index >= HOLDOUT_START]
        res.append({"name": f"{name} · {e}", "windows": wins, "exit": e, "rets": x, **perf(x),
                    "exposure": float((wts.reindex(x.index) > 0).mean()),
                    "turnover": float(wts.fillna(0).diff().abs().reindex(x.index).sum() / (len(x) / ANN))})
    N = n_prior + len(specs)
    for r in res:
        r["dsr"] = deflated_sharpe(r["rets"].values, N)
        r["n_trials"] = N
    bh = c.pct_change()
    bh = bh[bh.index >= (HOLDOUT_START if holdout else res[0]["rets"].index[0])]
    main = res[0]
    ok = main["sharpe"] >= 0.8 and main["dsr"] > 0.95
    return {"specs": res, "main": main, "bh": perf(bh.dropna()), "ok": ok, "holdout": holdout, "N": N,
            "version": version, "period": (main["rets"].index[0], main["rets"].index[-1]), "fee_bps": fee_bps}


def report_h2(res: dict, name: str) -> str:
    m = res["main"]
    L = [f"# H2 — ансамбль Дончиана (реплика Zarattini 2025, версия {res['version']}) · {name}"
         f"{' · HOLDOUT' if res['holdout'] else ''}", "",
         f"Период оценки {res['period'][0].date()} — {res['period'][1].date()}"
         + (" (только отложенный период)." if res["holdout"] else f" (до holdout {HOLDOUT_START.date()}).")
         + f" Издержки {res['fee_bps']:g} б.п. на оборот. N испытаний для DSR: **{res['N']}**.", "",
         "| спецификация | Sharpe | CAGR | вол. | MDD | дней в позиции | оборот/год | DSR |",
         "|---|---|---|---|---|---|---|---|",
         f"| buy & hold | {res['bh']['sharpe']:.2f} | {res['bh']['cagr']:+.1%} | {res['bh']['vol']:.1%} | "
         f"{res['bh']['mdd']:.1%} | 100% | — | — |"]
    for s in res["specs"]:
        L.append(f"| {s['name']} | {s['sharpe']:.2f} | {s['cagr']:+.1%} | {s['vol']:.1%} | {s['mdd']:.1%} | "
                 f"{s['exposure']:.0%} | {s['turnover']:.1f}× | {s['dsr']:.2f} |")
    L += ["", "## Вердикт", "",
          f"Основная спецификация (ансамбль, стоп-храповик по середине канала): Sharpe **{m['sharpe']:.2f}**, "
          f"DSR **{m['dsr']:.2f}** при N = {res['N']}. Критерий H2 (на holdout): Sharpe ≥ 0.8 и DSR > 0.95. "
          + ("**Выполнен.**" if res["ok"] else "**Не выполнен.**")
          + ("" if res["holdout"] else " _Это период разработки; окончательный вердикт — на holdout._")]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--version", default="v1", choices=["v0", "v1"])
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args()
    name = Path(a.data).stem
    res = evaluate_h2(load(a.data), n_prior=n_trials(), version=a.version)
    spec = "v1: 0.25/σ90 cap2 band20%(σ) 10bps" if a.version == "v1" else "v0: 0.25/σ90 cap1 band20% 10bps"
    if not a.no_register:
        register([{"hypothesis": "H2", "asset": name.split("_")[0], "tf": "1d", "model": "donchian",
                   "features": ",".join(map(str, s["windows"])), "barriers": s["exit"], "horizon_h": "",
                   "rule": f"{s['name']}; {spec}", "period_start": res["period"][0],
                   "period_end": res["period"][1], "holdout": False, "mean_bps": float(s["rets"].mean() * 1e4),
                   **trade_moments(s["rets"].values), "n_trials_at_reg": res["N"], "dsr_at_reg": s["dsr"],
                   "notes": f"Sharpe {s['sharpe']:.2f} MDD {s['mdd']:.1%}; {name}"} for s in res["specs"]])
    rep = report_h2(res, name)
    out = Path("reports") / f"H2_{name}{'_v0' if a.version == 'v0' else ''}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(rep, encoding="utf-8")
    print(rep, f"\n\nСохранено: {out}", sep="")


if __name__ == "__main__":
    main()
