"""Ablation и фильтры поверх модели v1: H4 (taker imbalance), H10 (HMM), H7 (хвостовой фильтр dOI/DVOL).

    python -m signal_lab.ablation --data data/BTCUSDT_4h.parquet --hours 48 --drop F5_ --hypothesis H4
    python -m signal_lab.ablation --data data/BTCUSDT_4h.parquet --hours 48 --drop X_hmm --hypothesis H10
    python -m signal_lab.ablation --data data/BTCUSDT_4h.parquet --hours 48 --gate --hypothesis H7

Ablation (H4, H10): полная M2 против M2 без группы признаков на одних и тех же тестовых событиях.
Смотрим изменение log-loss (и против константы — доли прибыльных сделок на обучении) и изменение средней сделки
правила «EV > 0»; интервалы — стационарный бутстрап по событиям. Критерий (раздел 4): удаление группы значимо
ухудшает log-loss (p < 0.05 с поправкой на число окон/групп).
H7: сигнал не выдаётся, если dOI за 7 дней или DVOL выше 95-го перцентиля обучающего окна (порог — только по
прошлому). Критерий: просадка ниже на ≥ 20 % при средней сделке не хуже.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .registry import n_trials, register, trade_moments
from .run import load
from .stats import stationary_bootstrap_ci
from .v1 import evaluate_v1


def logloss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Средняя разница a − b по событиям и 95 % ДИ стационарного бутстрапа; p — доля бутстрап-средних ≤ 0."""
    d = a - b
    lo, hi = stationary_bootstrap_ci(d, block=5.0, n_boot=1000)
    rng = np.random.default_rng(1)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    return float(d.mean()), float(lo), float(hi), float(np.mean(boots <= 0))


def max_dd(r: np.ndarray) -> float:
    eq = np.cumsum(r)
    return float(np.max(np.maximum.accumulate(eq) - eq)) if len(r) else 0.0


def ablation(df, tf, hours, drop, n_prior):
    full = evaluate_v1(df, tf, hours, n_trials_prior=n_prior)
    abl = evaluate_v1(df, tf, hours, drop=drop, n_trials_prior=n_prior)
    assert np.array_equal(full["test_idx"], abl["test_idx"]), "разные тестовые события — сравнение некорректно"
    y = full["y"]
    out = {"full": full, "abl": abl, "rows": []}
    for k in ("logit", "lgbm"):
        ll_f, ll_a, ll_c = logloss(full["probs"][k], y), logloss(abl["probs"][k], y), logloss(full["p_base"], y)
        d, lo, hi, p = boot_diff(ll_a, ll_f)                 # > 0 — без группы хуже (группа полезна)
        cost = 2 * full["fee_bps"] / 1e4
        ev = lambda res: res["probs"][k] * res["tp_w"] - (1 - res["probs"][k]) * res["sl_w"] - cost > 0
        tf_, ta_ = ev(full), ev(abl)
        rf, ra = np.where(tf_, full["r_sig"] - cost, 0.0), np.where(ta_, abl["r_sig"] - cost, 0.0)
        dm, mlo, mhi, _ = boot_diff(rf, ra)
        out["rows"].append({"model": k, "ll_full": ll_f.mean(), "ll_abl": ll_a.mean(), "ll_const": ll_c.mean(),
                            "d_ll": d, "d_ll_lo": lo, "d_ll_hi": hi, "p": p,
                            "share_folds_better_than_const": full["fold_ll_better"][k],
                            "trades_full": int(tf_.sum()), "trades_abl": int(ta_.sum()),
                            "d_mean_bps": dm * 1e4, "d_mean_lo": mlo * 1e4, "d_mean_hi": mhi * 1e4,
                            "rets_abl": ra[ta_]})
    return out


def tail_gate(df, tf, hours, n_prior, q: float = 0.95):
    res = evaluate_v1(df, tf, hours, n_trials_prior=n_prior)
    X, idx, fs = res["X"], res["test_idx"], res["fold_start"]
    cols = [c for c in ("F6_doi_7d", "F8_dvol") if c in X]
    block = np.zeros(len(idx), bool)
    for s in np.unique(fs):
        m = fs == s
        train = X.iloc[: max(0, s - res["h"])]
        for c in cols:
            thr = np.nanquantile(train[c].values, q)                # порог только по прошлому
            block[m] |= X[c].values[idx[m]] > thr
    cost = 2 * res["fee_bps"] / 1e4
    rows = []
    for name, take in (("primary", np.ones(len(idx), bool)),
                       ("M2[logit] EV>0", res["probs"]["logit"] * res["tp_w"] - (1 - res["probs"]["logit"]) * res["sl_w"] - cost > 0),
                       ("M2[lgbm] EV>0", res["probs"]["lgbm"] * res["tp_w"] - (1 - res["probs"]["lgbm"]) * res["sl_w"] - cost > 0)):
        r0 = res["r_sig"][take] - cost
        r1 = res["r_sig"][take & ~block] - cost
        rows.append({"rule": name, "trades": len(r0), "trades_gated": len(r1), "mean_bps": r0.mean() * 1e4,
                     "mean_gated_bps": r1.mean() * 1e4 if len(r1) else np.nan, "mdd": max_dd(r0), "mdd_gated": max_dd(r1),
                     "rets_gated": r1})
    return {"res": res, "rows": rows, "blocked_share": float(block.mean()), "cols": cols}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--hours", type=int, default=48)
    ap.add_argument("--drop", default="")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--hypothesis", required=True)
    ap.add_argument("--n-tests", type=int, default=2, help="поправка Бонферрони: сколько ablation-сравнений в семье")
    a = ap.parse_args()
    name = Path(a.data).stem
    asset = name.split("_")[0]
    df = load(a.data)
    L = []
    if a.gate:
        g = tail_gate(df, a.tf, a.hours, n_trials())
        L += [f"# {a.hypothesis} — хвостовой фильтр ({', '.join(g['cols'])} > 95-го перцентиля обучения) · {name} · {a.hours} ч", "",
              f"Заблокировано событий: {g['blocked_share']:.1%}. Период {g['res']['period'][0]} — {g['res']['period'][1]} (до holdout).", "",
              "| правило | сделок | с фильтром | средняя, б.п. | с фильтром | макс. просадка, б.п. | с фильтром |", "|---|---|---|---|---|---|---|"]
        ok_any = False
        for r in g["rows"]:
            L.append(f"| {r['rule']} | {r['trades']} | {r['trades_gated']} | {r['mean_bps']:+.1f} | {r['mean_gated_bps']:+.1f} | "
                     f"{r['mdd'] * 1e4:.0f} | {r['mdd_gated'] * 1e4:.0f} |")
            r["ok"] = r["mdd_gated"] <= 0.8 * r["mdd"] and r["mean_gated_bps"] >= r["mean_bps"]
            ok_any |= r["ok"]
        L += ["", "## Вердикт", "", f"**{a.hypothesis}: {'подтверждена' if ok_any else 'не подтверждена'}** "
              "(критерий: просадка ниже на ≥ 20 % при средней сделке не хуже)."]
        register([{"hypothesis": a.hypothesis, "asset": asset, "tf": a.tf, "model": r["rule"], "horizon_h": a.hours,
                   "rule": "tail gate dOI7d/DVOL >q95(train)", "holdout": False, "mean_bps": r["mean_gated_bps"],
                   **trade_moments(r["rets_gated"]), "n_trials_at_reg": n_trials() + 3,
                   "verdict": "подтверждена" if r["ok"] else "не подтверждена",
                   "notes": f"MDD {r['mdd'] * 1e4:.0f}->{r['mdd_gated'] * 1e4:.0f} bps; mean {r['mean_bps']:+.1f}->{r['mean_gated_bps']:+.1f}"}
                  for r in g["rows"]])
    else:
        drop = tuple(p for p in a.drop.split(",") if p)
        o = ablation(df, a.tf, a.hours, drop, n_trials())
        alpha = 0.05 / a.n_tests
        L += [f"# {a.hypothesis} — ablation: без признаков {', '.join(drop)} · {name} · {a.hours} ч", "",
              f"Тестовые события: {len(o['full']['y'])}, период {o['full']['period'][0]} — {o['full']['period'][1]} (до holdout). "
              f"Порог значимости с поправкой Бонферрони: p < {alpha:.3f}.", "",
              "| модель | log-loss полная | без группы | константа | Δ (без − полная) | 95 % ДИ | p | фолдов лучше константы | Δ средней сделки EV>0, б.п. [95 % ДИ] |",
              "|---|---|---|---|---|---|---|---|---|"]
        ok = False
        for r in o["rows"]:
            L.append(f"| {r['model']} | {r['ll_full']:.4f} | {r['ll_abl']:.4f} | {r['ll_const']:.4f} | {r['d_ll']:+.4f} | "
                     f"[{r['d_ll_lo']:+.4f}; {r['d_ll_hi']:+.4f}] | {r['p']:.3f} | {r['share_folds_better_than_const']:.0%} | "
                     f"{r['d_mean_bps']:+.1f} [{r['d_mean_lo']:+.0f}; {r['d_mean_hi']:+.0f}] |")
            r["ok"] = r["d_ll"] > 0 and r["p"] < alpha
            ok |= r["ok"]
        L += ["", "Δ log-loss > 0 — без группы модель хуже, т.е. группа несёт информацию. Положительная Δ средней сделки — "
              "с группой сделки лучше.", "", "## Вердикт", "",
              f"**{a.hypothesis}: {'подтверждена' if ok else 'не подтверждена'}** (критерий: удаление группы значимо ухудшает log-loss)."]
        register([{"hypothesis": a.hypothesis, "asset": asset, "tf": a.tf, "model": r["model"], "horizon_h": a.hours,
                   "features": "drop=" + ",".join(drop), "rule": "M2 EV>0 без группы", "holdout": False,
                   "mean_bps": float(np.mean(r["rets_abl"]) * 1e4) if len(r["rets_abl"]) else "",
                   **trade_moments(r["rets_abl"]), "n_trials_at_reg": n_trials() + 2,
                   "verdict": "подтверждена" if r["ok"] else "не подтверждена",
                   "notes": f"dLL {r['d_ll']:+.4f} [{r['d_ll_lo']:+.4f};{r['d_ll_hi']:+.4f}] p={r['p']:.3f}"} for r in o["rows"]])
    rep = "\n".join(L)
    tag = "gate" if a.gate else "drop-" + a.drop.replace(",", "-")
    out = Path("reports") / f"{a.hypothesis}_{name}_{a.hours}h_{tag}.md"
    out.write_text(rep, encoding="utf-8")
    print(rep)


if __name__ == "__main__":
    main()
