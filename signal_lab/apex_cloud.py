"""Проверка стороннего индикатора Apex Cloud (TradingView, veilazaro010) на экспортированных данных графика.

Правило зафиксировано до прогона: docs/research/experiments/H-ApexCloud.md. Сюда ничего не добавлять и не
менять после того, как увидены числа доходности — любое изменение логики это новое испытание.

    python -m signal_lab.apex_cloud --data data/apex/SOLUSDT_binance_4h.csv --fee-bps 5
    python -m signal_lab.apex_cloud --data data/apex/BNBUSDT_bybit_4h.csv --fee-bps 5.5
    python -m signal_lab.apex_cloud --data data/apex/DOGEUSDT_bybit_4h.csv --fee-bps 5.5

Формат входа — «Экспорт данных графика» TradingView с добавленным индикатором Apex Cloud (и чем угодно ещё —
лишние колонки игнорируются). Обязательные колонки: time, open, high, low, close, Trend UP, Trend DOWN,
Cloud flip ▲, Cloud flip ▼.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import deflated_sharpe
from .h1 import perf
from .registry import n_trials, register, trade_moments
from .stats import min_track_record, psr, stationary_bootstrap_ci

VARIANTS = {"trend": ("Trend UP", "Trend DOWN"), "cloud": ("Cloud flip ▲", "Cloud flip ▼")}


def load_export(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time").sort_index()


def stop_and_reverse(df: pd.DataFrame, up_col: str, dn_col: str) -> pd.Series:
    """Позиция на закрытии бара: 0 до первого сигнала, затем +1/−1, держится до противоположного сигнала.
    Каузально: событие на баре t определяет позицию, которая будет применена к доходности t+1 → t+2
    (вход по открытию следующего бара — это делает strat_returns)."""
    up, dn = df[up_col].fillna(0).values, df[dn_col].fillna(0).values
    pos = np.zeros(len(df))
    cur = 0.0
    for i in range(len(df)):
        if up[i] == 1:
            cur = 1.0
        elif dn[i] == 1:
            cur = -1.0
        pos[i] = cur
    return pd.Series(pos, index=df.index)


def strat_returns(df: pd.DataFrame, pos: pd.Series, fee_bps: float) -> pd.Series:
    """Решение на закрытии бара t реализуется в доходности бара t+1 (close[t]→close[t+1], приближение
    «вход по следующему открытию» — та же конвенция, что в h1.py/h2.py). Издержки — на баре, где новая
    позиция впервые действует."""
    r = df["close"].pct_change()
    p = pos.shift(1).fillna(0.0)
    turn = p.diff().abs().fillna(p.abs())
    return (p * r - turn * fee_bps / 1e4).dropna()


def trade_returns(df: pd.DataFrame, pos: pd.Series, fee_bps: float) -> np.ndarray:
    """Доходность по сделкам целиком (вход на перевороте → следующий переворот), а не по барам — для PSR/DSR/MinTRL,
    где единица наблюдения должна быть независимым решением, а не каждым баром удержания.
    Последняя сделка, если держится до конца ряда, закрывается по последнему close (mark-to-market)."""
    p = pos.shift(1).fillna(0).values                  # эффективная позиция, применяемая к барам (next-open entry)
    o, c = df["open"].values, df["close"].values
    n = len(p)
    changes = np.where(np.diff(np.r_[0.0, p]) != 0)[0]
    changes = np.r_[changes, n]
    rets = []
    for a, b in zip(changes[:-1], changes[1:]):
        side = p[a]
        if side == 0 or a + 1 >= n:
            continue
        entry = o[a + 1]
        exit_ = o[b] if b < n else c[-1]                # открытая до конца ряда сделка — выход по последнему close
        if not np.isfinite(entry) or not np.isfinite(exit_) or entry <= 0 or exit_ <= 0:
            continue
        cost = 2 * fee_bps / 1e4                       # вход + выход
        rets.append(side * np.log(exit_ / entry) - cost)
    return np.array(rets)


def placebo(df: pd.DataFrame, up_col: str, dn_col: str, fee_bps: float, n_perm: int = 500, seed: int = 0) -> dict:
    """Число и чередование знаков событий сохраняются, их расположение на оси времени — случайное
    (после разминки индикатора: первые valid_from баров исключены из возможных позиций событий)."""
    up, dn = df[up_col].fillna(0).values, df[dn_col].fillna(0).values
    events = sorted([(i, 1) for i in np.where(up == 1)[0]] + [(i, -1) for i in np.where(dn == 1)[0]])
    if not events:
        return {"real": np.nan, "perm": np.array([]), "p": np.nan}
    signs = [s for _, s in events]
    valid_from = int(df[["Cloud top", "SSL high"]].notna().all(axis=1).values.argmax()) if "Cloud top" in df else 0
    real_pos = stop_and_reverse(df, up_col, dn_col)
    real_rets = trade_returns(df, real_pos, fee_bps)
    real_sharpe = perf(strat_returns(df, real_pos, fee_bps))["sharpe"]
    rng = np.random.default_rng(seed)
    n = len(df)
    sh = []
    for _ in range(n_perm):
        idx = rng.choice(np.arange(valid_from, n), size=len(signs), replace=False)
        idx.sort()
        u, d = np.zeros(n), np.zeros(n)
        for i, s in zip(idx, signs):
            (u if s == 1 else d)[i] = 1
        pos = stop_and_reverse(df.assign(**{up_col: u, dn_col: d}), up_col, dn_col)
        sh.append(perf(strat_returns(df, pos, fee_bps))["sharpe"])
    sh = np.array(sh)
    p = float((np.sum(sh >= real_sharpe) + 1) / (n_perm + 1))
    return {"real": real_sharpe, "perm": sh, "p": p, "real_trades": real_rets}


def evaluate(df: pd.DataFrame, variant: str, fee_bps: float) -> dict:
    up_col, dn_col = VARIANTS[variant]
    pos = stop_and_reverse(df, up_col, dn_col)
    bar_rets = strat_returns(df, pos, fee_bps)
    trades = trade_returns(df, pos, fee_bps)
    bh = df["close"].pct_change().dropna()
    bh = bh[bh.index.isin(bar_rets.index)]
    p = perf(bar_rets)
    stats = {}
    if len(trades) >= 2:
        lo, hi = stationary_bootstrap_ci(trades, block=2.0, n_boot=2000)
        stats = {"n_trades": len(trades), "hit": float(np.mean(trades > 0)), "mean_bps": float(trades.mean() * 1e4),
                 "ci_lo_bps": lo * 1e4, "ci_hi_bps": hi * 1e4, "psr": psr(trades), "min_trl": min_track_record(trades)}
    return {"variant": variant, "pos": pos, "bar_rets": bar_rets, "trades": trades, "perf": p,
            "bh_perf": perf(bh), "stats": stats, "n_events": int((df[up_col].sum() + df[dn_col].sum()))}


def report(name: str, results: list[dict], pl: dict, fee_bps: float, n_trials_total: int) -> str:
    L = [f"# Apex Cloud (TradingView, veilazaro010) · {name}", "",
         f"Издержки {fee_bps:g} б.п. за сторону. N испытаний для DSR (реестр на момент прогона): **{n_trials_total}**. "
         "Правило зафиксировано до прогона: docs/research/experiments/H-ApexCloud.md.", ""]
    for r in results:
        s, p, bh = r["stats"], r["perf"], r["bh_perf"]
        L += [f"## Вариант «{r['variant']}» ({r['n_events']} событий)", "",
              "| | Sharpe | CAGR | вол. | MDD |", "|---|---|---|---|---|",
              f"| buy & hold | {bh['sharpe']:+.2f} | {bh['cagr']:+.1%} | {bh['vol']:.1%} | {bh['mdd']:.1%} |",
              f"| Apex Cloud stop-reverse | {p['sharpe']:+.2f} | {p['cagr']:+.1%} | {p['vol']:.1%} | {p['mdd']:.1%} |", ""]
        if s:
            dsr = deflated_sharpe(r["trades"], n_trials_total)
            min_trl = "∞" if not np.isfinite(s["min_trl"]) else f"{s['min_trl']:.0f}"
            L += [f"Сделок: **{s['n_trades']}**, прибыльных {s['hit']:.0%}, средняя {s['mean_bps']:+.1f} б.п. "
                  f"(95% ДИ [{s['ci_lo_bps']:+.0f}; {s['ci_hi_bps']:+.0f}]), PSR {s['psr']:.2f}, "
                  f"MinTRL {min_trl}, DSR {dsr:.2f}.",
                  "", ("**Меньше 10 сделок — нет ответа, статистической мощности недостаточно.**" if s["n_trades"] < 10
                       else "**Меньше 100 сделок — вне общего гейта (ориентир, не прошедший результат).**"), ""]
        else:
            L += ["Сделок меньше 2 — результат не считается.", ""]
    if np.isfinite(pl.get("real", np.nan)):
        L += ["## Плацебо (перестановка моментов событий, число и чередование сохранены)", "",
              f"Настоящий Sharpe (основной вариант): **{pl['real']:+.2f}**. На {len(pl['perm'])} перестановках: "
              f"медиана {np.median(pl['perm']):+.2f}, 95-й перцентиль {np.quantile(pl['perm'], 0.95):+.2f}, "
              f"p = {pl['p']:.3f}.", "",
              ("**Момент сигнала отличим от случайного.**" if pl["p"] < 0.05 else
               "**Момент сигнала НЕ отличим от случайного** — тот же эффект дают случайные моменты разворота "
               "с тем же числом событий."), ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--fee-bps", type=float, required=True)
    ap.add_argument("--placebo", type=int, default=500)
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args()
    name = Path(a.data).stem
    asset = name.split("_")[0]
    df = load_export(a.data)
    results = [evaluate(df, v, a.fee_bps) for v in VARIANTS]
    pl = placebo(df, *VARIANTS["trend"], a.fee_bps, a.placebo) if a.placebo else {}
    N = n_trials() + len(results) + (1 if pl else 0)
    rep = report(name, results, pl, a.fee_bps, N)
    if not a.no_register:
        rows = []
        for r in results:
            s = r["stats"]
            rows.append({"hypothesis": "H-ApexCloud", "asset": asset, "tf": "4h", "model": "indicator",
                        "rule": f"stop-reverse {r['variant']} (Apex Cloud)", "horizon_h": "", "holdout": False,
                        "n_trades": s.get("n_trades", 0), "hit": s.get("hit"), "mean_bps": s.get("mean_bps"),
                        "ci_lo_bps": s.get("ci_lo_bps"), "ci_hi_bps": s.get("ci_hi_bps"),
                        **(trade_moments(r["trades"]) if len(r["trades"]) >= 10 else {}),
                        "n_trials_at_reg": N, "dsr_at_reg": deflated_sharpe(r["trades"], N) if len(r["trades"]) >= 10 else "",
                        "verdict": "нет ответа" if s.get("n_trades", 0) < 10 else "ориентир (< 100 сделок)",
                        "notes": f"Sharpe {r['perf']['sharpe']:+.2f} vs B&H {r['bh_perf']['sharpe']:+.2f}; {a.data}"})
        if pl:
            rows.append({"hypothesis": "H-ApexCloud-placebo", "asset": asset, "tf": "4h", "model": "indicator",
                        "rule": "перестановка моментов событий (trend)", "n_trials_at_reg": N,
                        "verdict": "контроль", "notes": f"real Sharpe {pl['real']:+.2f}; p={pl['p']:.3f}"})
        register(rows)
    out = Path("reports") / f"ApexCloud_{name}.md"
    out.write_text(rep, encoding="utf-8")
    print(rep)
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()
