"""R2-A — временной моментум панелью на дневках (раунд R2, docs/research/experiments/R2-A-tsmom-daily.md).

    python -m signal_lab.r2a_tsmom --control            # положительный контроль на синтетике (рынок не трогает)
    python -m signal_lab.r2a_tsmom --half dev           # разработка: первая половина монет
    python -m signal_lab.r2a_tsmom --half dev --placebo 200
    python -m signal_lab.r2a_tsmom --half check         # только если dev прошла гейт; без подстройки

Спецификация (зафиксирована до прогона): сигнал s = sign(log C_t − log C_{t−L}), L ∈ {7, 14, 28} дней;
перекрывающиеся портфели Джегадиша–Титмана с удержанием H ∈ {7, 28}: позиция = среднее s за последние H дней,
умноженное на min(1, 0.25 / σ̂_20д). Решение на закрытии дня t, исполнение по открытию t+1 (доходность open→open).
Портфель — равные веса по доступным монетам, long-short. Издержки 7 б.п. за единицу оборота. Ориентир — та же
панель long-only с тем же размером по волатильности. Монета входит в панель через 30 дней после первой свечи.
"""
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .registry import dsr_from_moments, n_trials, register, trade_moments
from .run import load
from .stats import pbo_cscv, stationary_bootstrap_ci

ANN = 365
END = pd.Timestamp("2026-09-30", tz="UTC")          # forward-holdout с 01.10.2026 не читаем
DEV = ["ADA", "ATOM", "DOT", "LINK", "LTC", "NEAR", "SOL", "TRX", "UNI", "XRP"]
CHECK = ["AAVE", "AVAX", "BCH", "BNB", "BTC", "DOGE", "ETC", "ETH", "FIL", "XLM"]
LOOKBACKS, HOLDS = (7, 14, 28), (7, 28)
FEE_BPS, TARGET, VOL_WIN, WARMUP = 7.0, 0.25, 20, 30
N_ROUND = 26                                          # бюджет испытаний раунда R2


def load_panel(coins: list[str], data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Цены открытия и закрытия на общей дневной сетке UTC до END включительно."""
    o, c = {}, {}
    for k in coins:
        df = load(Path(data_dir) / f"{k}USDT_spot_1d.parquet")
        df = df[df.index <= END]
        o[k], c[k] = df["open"], df["close"]
    idx = pd.date_range(min(s.index[0] for s in c.values()), END, freq="D", tz="UTC")
    return pd.DataFrame(o).reindex(idx), pd.DataFrame(c).reindex(idx)


def positions(close: pd.DataFrame, L: int, H: int, long_only: bool = False) -> pd.DataFrame:
    """Позиция на закрытии дня t (по данным до t включительно)."""
    lc = np.log(close)
    s = pd.DataFrame(1.0, index=close.index, columns=close.columns) if long_only else np.sign(lc - lc.shift(L))
    s = s.where(lc.shift(L).notna())
    jt = s.rolling(H, min_periods=H).mean()
    sig = lc.diff().rolling(VOL_WIN, min_periods=VOL_WIN).std() * math.sqrt(ANN)
    p = jt * np.minimum(1.0, TARGET / sig)
    age = close.notna().cumsum()
    return p.where(age > WARMUP)


def portfolio_returns(open_: pd.DataFrame, p: pd.DataFrame, fee_bps: float = FEE_BPS) -> pd.Series:
    """Дневная доходность портфеля: позиция с закрытия t−1 держится с открытия t до открытия t+1."""
    r = open_.shift(-1) / open_ - 1                    # open_t → open_{t+1}
    pl = p.shift(1)
    live = pl.notna() & r.notna()
    n = live.sum(axis=1)
    gross = (pl * r).where(live).sum(axis=1)
    turn = (pl.fillna(0) - pl.shift(1).fillna(0)).abs().where(live).sum(axis=1)
    x = (gross - turn * fee_bps / 1e4) / n.replace(0, np.nan)
    return x[n > 0].dropna()


def ann_sharpe(x: pd.Series) -> float:
    return float(x.mean() / x.std() * math.sqrt(ANN)) if x.std() > 0 else 0.0


def nw_tstat_alpha(y: pd.Series, x: pd.Series, lags: int = 10) -> tuple[float, float]:
    """Альфа регрессии y на x и её t-статистика Newey–West."""
    d = pd.concat([y, x], axis=1).dropna()
    Y, X = d.iloc[:, 0].values, np.c_[np.ones(len(d)), d.iloc[:, 1].values]
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    u = Y - X @ b
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for k in range(1, lags + 1):
        G = Xu[k:].T @ Xu[:-k]
        S += (1 - k / (lags + 1)) * (G + G.T)
    XtX = np.linalg.inv(X.T @ X)
    V = XtX @ S @ XtX
    return float(b[0]), float(b[0] / math.sqrt(V[0, 0]))


def sharpe_diff_ci(a: pd.Series, b: pd.Series, n_boot: int = 1000, block: float = 20.0, seed: int = 0):
    """95 % ДИ разницы годовых Sharpe (a − b) стационарным бутстрапом пар дней."""
    d = pd.concat([a, b], axis=1).dropna().values
    n, rng, out = len(d), np.random.default_rng(seed), []
    for _ in range(n_boot):
        idx = np.empty(n, dtype=int)
        i = rng.integers(n)
        for t in range(n):
            idx[t] = i
            i = rng.integers(n) if rng.random() < 1 / block else (i + 1) % n
        s = d[idx]
        sd = s.std(0, ddof=1)
        out.append((s[:, 0].mean() / sd[0] - s[:, 1].mean() / sd[1]) * math.sqrt(ANN))
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def dsr_daily(x: pd.Series, N: int) -> float:
    m = trade_moments(x.values)
    if "sr_trade" not in m:
        return 0.0
    return dsr_from_moments(m["sr_trade"], m["n_trades"], m["skew"], m["kurt"], N)


def evaluate(open_: pd.DataFrame, close: pd.DataFrame, n_prior: int, fee_bps: float = FEE_BPS) -> dict:
    bench = portfolio_returns(open_, positions(close, 7, 7, long_only=True), fee_bps)
    ew = (open_.shift(-1) / open_ - 1).where(close.notna().cumsum() > WARMUP).mean(axis=1)
    specs = []
    for L, H in itertools.product(LOOKBACKS, HOLDS):
        x = portfolio_returns(open_, positions(close, L, H), fee_bps)
        specs.append({"L": L, "H": H, "rets": x})
    common = pd.concat([s["rets"] for s in specs] + [bench], axis=1).dropna().index
    for s in specs:
        x = s["rets"].loc[common]
        s["rets"] = x
        s["sharpe"] = ann_sharpe(x)
        eq = (1 + x).cumprod()
        s["mdd"] = float((1 - eq / eq.cummax()).max())
        s["dsr_round"] = dsr_daily(x, N_ROUND)
        s["dsr_registry"] = dsr_daily(x, n_prior + len(specs))
        s["ci"] = stationary_bootstrap_ci(x.values, block=20, n_boot=1000)
        s["alpha"], s["alpha_t"] = nw_tstat_alpha(x, ew.loc[common])
        s["sharpe_x2"] = ann_sharpe(portfolio_returns(open_, positions(close, s["L"], s["H"]), 2 * fee_bps)
                                    .loc[common])
        yrs = x.groupby(x.index.year).mean()
        s["years_pos"] = (int((yrs > 0).sum()), len(yrs))
    best = max(specs, key=lambda s: s["sharpe"])
    best["diff_ci"] = sharpe_diff_ci(best["rets"], bench.loc[common])
    M = np.column_stack([s["rets"].values for s in specs])
    return {"specs": specs, "best": best, "bench": bench.loc[common], "bench_sharpe": ann_sharpe(bench.loc[common]),
            "ew_sharpe": ann_sharpe(ew.loc[common].dropna()), "pbo": pbo_cscv(M, 10)["pbo"],
            "period": (common[0], common[-1]), "N_registry": n_prior + len(specs), "fee_bps": fee_bps}


def gate(best: dict, pbo: float, dsr_key: str) -> dict:
    checks = {"DSR > 0.95": best[dsr_key] > 0.95, "PBO < 0.2": pbo < 0.2,
              "нижняя граница ДИ средней > 0": best["ci"][0] > 0,
              "альфа к рынку, t_NW > 3": best["alpha_t"] > 3, "Sharpe при издержках ×2 > 0": best["sharpe_x2"] > 0,
              "ДИ разницы Sharpe с vol-long > 0": best["diff_ci"][0] > 0,
              "больше половины лет в плюсе": best["years_pos"][0] > best["years_pos"][1] / 2}
    return {"checks": checks, "ok": all(checks.values())}


def placebo(open_: pd.DataFrame, close: pd.DataFrame, L: int, H: int, n_perm: int = 200, seed: int = 0) -> dict:
    """Сдвиг сигнала каждой монеты по кругу на случайное число дней (≥ 60): автокорреляция сигнала сохраняется,
    связь с будущей доходностью разрушается."""
    real = ann_sharpe(portfolio_returns(open_, positions(close, L, H)))
    rng = np.random.default_rng(seed)
    p0 = positions(close, L, H)
    sh = []
    for _ in range(n_perm):
        p = p0.copy()
        for k in p.columns:
            v = p[k].dropna()
            p.loc[v.index, k] = np.roll(v.values, int(rng.integers(60, max(61, len(v) - 60))))
        sh.append(ann_sharpe(portfolio_returns(open_, p)))
    sh = np.array(sh)
    return {"real": real, "p": float((np.sum(sh >= real) + 1) / (n_perm + 1)), "median": float(np.median(sh)),
            "q99": float(np.quantile(sh, 0.99))}


def synthetic_panel(n_coins: int = 10, days: int = 3200, sharpe: float = 1.5, seed: int = 0):
    """Положительный контроль: монеты с общим фактором и посаженным моментумом 28 дней. Сила эффекта подобрана
    так, чтобы годовой Sharpe правила L=28, H=28 был около `sharpe`."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=days, freq="D", tz="UTC")
    sd, k = 0.04, sharpe / math.sqrt(ANN) * 0.04 * 0.85           # 0.85 — поправка на корреляцию монет и JT
    mkt = rng.standard_t(4, days) * sd * 0.6 / math.sqrt(2)
    R = np.empty((days, n_coins))
    for j in range(n_coins):
        eps = rng.standard_t(4, days) * sd * 0.8 / math.sqrt(2)
        r = np.empty(days)
        for t in range(days):
            past = r[max(0, t - 28):t].sum() if t >= 28 else 0.0
            r[t] = mkt[t] + eps[t] + (k * np.sign(past) if t >= 28 else 0.0)
        R[:, j] = r
    close = pd.DataFrame(100 * np.exp(np.cumsum(R, 0)), index=idx, columns=[f"S{j}" for j in range(n_coins)])
    return close.shift(1).fillna(100.0), close                    # open_t = close_{t−1}


def positive_control(n_sims: int = 40, sharpes=(0.0, 1.0, 1.5, 2.0)) -> list[dict]:
    out = []
    for S in sharpes:
        hits_round = hits_reg = 0
        realized = []
        for i in range(n_sims):
            o, c = synthetic_panel(sharpe=S, seed=1000 + i)
            x = portfolio_returns(o, positions(c, 28, 28))
            realized.append(ann_sharpe(x))
            hits_round += dsr_daily(x, N_ROUND) > 0.95
            hits_reg += dsr_daily(x, 517) > 0.95
        out.append({"target": S, "realized": float(np.median(realized)), "power_round": hits_round / n_sims,
                    "power_registry": hits_reg / n_sims})
    return out


def report(res: dict, half: str, g_round: dict, g_reg: dict) -> str:
    b = res["best"]
    L = [f"# R2-A — TSMOM панелью на дневках · половина {half}", "",
         f"Период {res['period'][0].date()} — {res['period'][1].date()}, монеты: {', '.join(DEV if half == 'dev' else CHECK)}. "
         f"Издержки {res['fee_bps']:g} б.п. на оборот. N для DSR: раунд {N_ROUND}, реестр {res['N_registry']}.", "",
         f"Ориентиры: long-only с тем же размером по волатильности — Sharpe {res['bench_sharpe']:.2f}; "
         f"равновзвешенный buy & hold — Sharpe {res['ew_sharpe']:.2f}.", "",
         "| L, дней | H, дней | Sharpe | MDD | средняя, б.п./день [95 % ДИ] | альфа t_NW | Sharpe при ×2 | лет в плюсе | DSR (N раунда) | DSR (N реестра) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for s in res["specs"]:
        L.append(f"| {s['L']} | {s['H']} | {s['sharpe']:.2f} | {s['mdd']:.1%} | {s['rets'].mean()*1e4:+.1f} "
                 f"[{s['ci'][0]*1e4:+.1f}; {s['ci'][1]*1e4:+.1f}] | {s['alpha_t']:.2f} | {s['sharpe_x2']:.2f} | "
                 f"{s['years_pos'][0]}/{s['years_pos'][1]} | {s['dsr_round']:.2f} | {s['dsr_registry']:.2f} |")
    L += ["", f"PBO по 6 конфигурациям: **{res['pbo']:.2f}**. Лучшая на разработке: L={b['L']}, H={b['H']}; "
          f"ДИ разницы Sharpe с vol-long: [{b['diff_ci'][0]:+.2f}; {b['diff_ci'][1]:+.2f}].", "",
          "## Гейт (лучшая конфигурация)", "", "| условие | N раунда | N реестра |", "|---|---|---|"]
    for k in g_round["checks"]:
        L.append(f"| {k} | {'да' if g_round['checks'][k] else 'нет'} | {'да' if g_reg['checks'][k] else 'нет'} |")
    L += ["", f"Итог: при N раунда — **{'пройден' if g_round['ok'] else 'не пройден'}**, "
          f"при N реестра — **{'пройден' if g_reg['ok'] else 'не пройден'}**."]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--half", choices=["dev", "check"])
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--placebo", type=int, default=0)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args()
    out_dir = Path("reports")
    if a.control:
        rows = positive_control()
        txt = ["# R2-A — положительный контроль на синтетике", "",
               "10 монет × 3200 дней, общий фактор, хвосты t(4), посаженный моментум 28 дней; правило L=28, H=28; "
               "40 симуляций на точку. Мощность = доля симуляций с DSR > 0.95.", "",
               "| целевой Sharpe | реализованный (медиана) | мощность при N=26 | мощность при N=517 |", "|---|---|---|---|"]
        txt += [f"| {r['target']:.1f} | {r['realized']:.2f} | {r['power_round']:.0%} | {r['power_registry']:.0%} |"
                for r in rows]
        rep = "\n".join(txt)
        (out_dir / "R2A_positive_control.md").write_text(rep, encoding="utf-8")
        print(rep)
        return
    coins = DEV if a.half == "dev" else CHECK
    o, c = load_panel([f"{k}" for k in coins], a.data_dir)
    if a.placebo:
        res = evaluate(o, c, n_trials())
        b = res["best"]
        pl = placebo(o, c, b["L"], b["H"], a.placebo)
        rep = (f"# R2-A — плацебо (круговой сдвиг сигнала) · половина {a.half}\n\nЛучшая конфигурация L={b['L']}, "
               f"H={b['H']}: Sharpe **{pl['real']:.2f}**. На {a.placebo} сдвигах: медиана {pl['median']:.2f}, "
               f"99-й перцентиль {pl['q99']:.2f}, p = {pl['p']:.3f}. Порог плана: p < 0.01.")
        if not a.no_register:
            register([{"hypothesis": "R2-A-placebo", "asset": f"panel-{a.half}", "tf": "1d", "model": "tsmom-jt",
                       "rule": f"L={b['L']} H={b['H']} круговой сдвиг ×{a.placebo}", "holdout": False,
                       "notes": f"real {pl['real']:.2f}; median {pl['median']:.2f}; q99 {pl['q99']:.2f}; p={pl['p']:.3f}",
                       "n_trials_at_reg": n_trials() + 1, "verdict": "контроль"}])
        (out_dir / f"R2A_{a.half}_placebo.md").write_text(rep, encoding="utf-8")
        print(rep)
        return
    if a.half == "check":
        raise SystemExit("половина check открывается только после прохода гейта на dev (см. пре-регистрацию)")
    res = evaluate(o, c, n_trials())
    g_round, g_reg = gate(res["best"], res["pbo"], "dsr_round"), gate(res["best"], res["pbo"], "dsr_registry")
    if not a.no_register:
        register([{"hypothesis": "R2-A", "asset": f"panel-{a.half}", "tf": "1d", "model": "tsmom-jt",
                   "features": f"L={s['L']}", "horizon_h": s["H"] * 24,
                   "rule": f"sign(L) JT H={s['H']} × min(1,0.25/σ20) EW long-short {FEE_BPS:g}bps",
                   "period_start": res["period"][0], "period_end": res["period"][1], "holdout": False,
                   "mean_bps": float(s["rets"].mean() * 1e4), "ci_lo_bps": s["ci"][0] * 1e4,
                   "ci_hi_bps": s["ci"][1] * 1e4, **trade_moments(s["rets"].values),
                   "n_trials_at_reg": res["N_registry"], "dsr_at_reg": s["dsr_registry"], "pbo": res["pbo"],
                   "verdict": "прошёл" if (s is res["best"] and g_round["ok"]) else "не прошёл",
                   "notes": f"Sharpe {s['sharpe']:.2f}; DSR(N=26) {s['dsr_round']:.2f}; alpha t {s['alpha_t']:.2f}; "
                            f"vol-long Sharpe {res['bench_sharpe']:.2f}"} for s in res["specs"]])
    rep = report(res, a.half, g_round, g_reg)
    (out_dir / f"R2A_{a.half}.md").write_text(rep, encoding="utf-8")
    print(rep)


if __name__ == "__main__":
    main()
