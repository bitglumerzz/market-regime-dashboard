"""Модель v1 из docs/research/directional-edge-2026-09.md (раздел 3) и проверка гипотез H1–H3.

    python -m signal_lab.v1 --data data/BTCUSDT_4h.parquet --tf 4h            # горизонт по умолчанию 48 ч
    python -m signal_lab.v1 --data data/BTCUSDT_4h.parquet --tf 4h --placebo  # контроль: на перемешанных данных всё обязано провалиться

Устройство (мета-разметка, López de Prado):
  primary  — трендовый сигнал (ансамбль Дончиана + TSMOM 4 нед.) задаёт сторону сделки;
  барьеры  — цель +2σ, стоп −1.5σ (σ = волатильность на горизонт), выход по времени через 48 ч;
  M2       — LightGBM/логит предсказывает P(primary-сделка закроется в плюс) по признакам F1–F12;
  калибровка — изотоническая на последней четверти обучающего окна (с зазором);
  вход     — только если EV = p·цель − (1−p)·стоп − издержки > 0, либо p > τ.
Сравнение (после издержек): primary без M2, «всегда long» с теми же барьерами, buy & hold и long с управлением
размером по волатильности (H1). Гейт — как в отчёте: DSR > 0.95, PBO < 0.2, нижняя граница 95% ДИ средней сделки > 0.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .barrier_eval import Config, score_configs
from .evaluate import Folds, _make_model, nonoverlap
from .features import build_features_v1, primary_side
from .labels import triple_barrier_sided, uniqueness_weights
from .run import BARS_PER_DAY, load


def _fit_calibrated(kind: str, X: np.ndarray, y: np.ndarray, horizon: int, w: np.ndarray | None = None):
    """Модель на первых 75% обучения + изотоническая калибровка на последних 25% (между ними зазор h баров).
    w — веса уникальности меток (пересекающиеся сделки весят меньше)."""
    cut = int(len(X) * 0.75)
    fit_idx, cal_idx = np.arange(0, max(0, cut - horizon)), np.arange(cut, len(X))
    if len(fit_idx) < 150 or len(cal_idx) < 50 or len(np.unique(y[fit_idx])) < 2:
        return None
    sw = None if w is None else w[fit_idx]
    m = _make_model(kind)
    m = m.fit(X[fit_idx], y[fit_idx], **({"logisticregression__sample_weight": sw} if kind == "logit" else {"sample_weight": sw}))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(m.predict_proba(X[cal_idx])[:, 1], y[cal_idx])
    return lambda Z: iso.predict(m.predict_proba(Z)[:, 1])


def vol_managed_long(close: pd.Series, bars_per_day: int, idx: np.ndarray, target_ann: float = 0.25) -> dict:
    """H1: long с весом min(1, σ_цель/σ̂_20д) против buy & hold на тех же тестовых отрезках (барная доходность, без плеча)."""
    r = np.log(close).diff()
    sig = r.ewm(span=20 * bars_per_day, min_periods=10 * bars_per_day).std().shift(1) * math.sqrt(365 * bars_per_day)
    w = np.minimum(1.0, target_ann / sig)
    span = np.zeros(len(close), bool)
    span[idx.min():idx.max() + 1] = True
    out = {}
    for name, series in (("buy & hold", r), ("long, размер по волатильности", w * r)):
        x = series[span].dropna().values
        eq = np.exp(np.cumsum(x))
        mdd = float(np.max(1 - eq / np.maximum.accumulate(eq))) if len(eq) else math.nan
        out[name] = {"sharpe": float(x.mean() / x.std() * math.sqrt(365 * bars_per_day)) if x.std() > 0 else 0.0,
                     "cagr": float(np.exp(x.sum() * 365 * bars_per_day / max(1, len(x))) - 1), "mdd": mdd}
    return out


def evaluate_v1(df: pd.DataFrame, tf: str, horizon_hours: int = 48, fee_bps: float = 5.0, tp: float = 2.0, sl: float = 1.5,
                taus=(0.55, 0.60), test_days: int = 90, min_train_days: int = 365, placebo: bool = False, seed: int = 0) -> dict:
    bpd = BARS_PER_DAY[tf]
    h = max(1, int(round(horizon_hours * bpd / 24)))
    X = build_features_v1(df, bpd)
    side = primary_side(X).values
    tb = triple_barrier_sided(df, h, side, tp, sl, vol_span=7 * bpd)
    tb_long = triple_barrier_sided(df, h, np.ones(len(df)), tp, sl, vol_span=7 * bpd)
    Xs = X.assign(side=side)
    if placebo:
        # плацебо для M2: базовый сигнал настоящий, а признаки мета-модели перемешаны блоками по 5 дней —
        # их связь с исходом сделки разорвана. M2 не должна улучшать primary; если «улучшает» — в конвейере утечка.
        rng = np.random.default_rng(seed)
        blocks = np.array_split(np.arange(len(X)), max(2, len(X) // (5 * bpd)))
        order = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
        Xs = pd.DataFrame(X.values[order], index=X.index, columns=X.columns).assign(side=side)
    cost = 2 * fee_bps / 1e4
    valid = ~Xs.isna().any(axis=1).values & tb["ret"].notna().values & tb_long["ret"].notna().values
    y = tb["label"].values
    folds = Folds(len(df), h, min_train=min_train_days * bpd, test_len=test_days * bpd)

    test_idx = np.concatenate([nonoverlap(te, h) for _, te in folds])
    test_idx = test_idx[valid[test_idx]]
    pos = {t: i for i, t in enumerate(test_idx)}
    probs = {k: np.full(len(test_idx), np.nan) for k in ("logit", "lgbm")}
    for tr, te in folds:
        tr = tr[valid[tr]]
        te = nonoverlap(te, h)
        te = te[valid[te]]
        if not len(te):
            continue
        wts = uniqueness_weights(tr + 1, tb["exit_bar"].values[tr])
        for k in probs:
            f = _fit_calibrated(k, Xs.values[tr], y[tr], h, wts)
            if f is not None:
                probs[k][[pos[t] for t in te]] = f(Xs.values[te])
    have = ~np.isnan(probs["logit"])
    test_idx = test_idx[have]
    s = side[test_idx]
    r_sig = tb["ret"].values[test_idx]                       # результат сделки со стороны primary
    tp_w, sl_w = tb["tp_w"].values[test_idx], tb["sl_w"].values[test_idx]
    r_long = tb_long["ret"].values[test_idx]

    configs = [Config("primary без фильтра", np.ones(len(test_idx)))]
    for k, p in probs.items():
        p = p[have]
        ev = p * tp_w - (1 - p) * sl_w - cost
        configs.append(Config(f"primary + M2[{k}] EV>0", (ev > 0).astype(float), p, tp_w, sl_w))
        for tau in taus:
            configs.append(Config(f"primary + M2[{k}] p>{tau:.2f}", (p > tau).astype(float), p, tp_w, sl_w))
    pbo = score_configs(configs, r_sig, cost, 365 * bpd / h)
    # ориентиры: «всегда long» с теми же барьерами и случайная сторона (результат long-стороны × случайный знак)
    rng = np.random.default_rng(seed + 1)
    base = [Config("ориентир: всегда long (те же барьеры)", np.ones(len(test_idx))),
            Config("ориентир: случайная сторона", rng.choice([-1.0, 1.0], len(test_idx)))]
    score_configs(base, r_long, cost, 365 * bpd / h)

    ranked = sorted([c for c in configs if c.stats.get("trades", 0) >= 10], key=lambda c: -c.stats["dsr"])
    best = ranked[0] if ranked else None
    long_mean = base[0].stats.get("mean_bps", -1e9)
    beats = bool(best and best.stats["mean_bps"] > long_mean)
    gate = bool(best and best.stats["dsr"] > 0.95 and best.stats["ci_lo_bps"] > 0 and best.stats["trades"] >= 100
                and (np.isnan(pbo["pbo"]) or pbo["pbo"] < 0.2) and beats)
    # H3: улучшает ли мета-модель базовый сигнал (по средней сделке и без потери значимости)
    m2 = [c for c in ranked if c is not configs[0]]
    prim = configs[0].stats
    m2_best = max(m2, key=lambda c: c.stats["mean_bps"], default=None)
    m2_improves = bool(m2_best and prim.get("trades", 0) >= 10 and m2_best.stats["dsr"] > 0.95
                       and m2_best.stats["mean_bps"] > prim["mean_bps"] + max(5.0, 0.1 * abs(prim["mean_bps"])))
    signals_per_month = best.stats["trades"] / max(1e-9, len(test_idx) * h / bpd / 30) if best else 0.0
    return {"configs": configs, "baselines": base, "pbo": pbo, "best": best, "gate": gate, "beats": beats,
            "m2_improves": m2_improves, "m2_best": m2_best,
            "h": h, "bpd": bpd, "n_test": len(test_idx), "primary_share": float(np.mean(s != 0)),
            "vol": vol_managed_long(df["close"], bpd, test_idx), "signals_per_month": signals_per_month,
            "fee_bps": fee_bps, "tp": tp, "sl": sl, "placebo": placebo,
            "period": (df.index[test_idx[0]], df.index[test_idx[-1]]) if len(test_idx) else None}


def report_v1(res: dict, name: str) -> str:
    b = res["best"]
    L = [f"# Модель v1 (мета-разметка) · {name}{' · ПЛАЦЕБО' if res['placebo'] else ''}", "",
         f"Горизонт {res['h']} баров ({res['h'] / res['bpd'] * 24:.0f} ч); цель +{res['tp']:g}σ, стоп −{res['sl']:g}σ; "
         f"комиссия {res['fee_bps']:g} б.п. за сторону. Тестовые события: {res['n_test']} (непересекающиеся), "
         f"период {res['period'][0]} — {res['period'][1]}." if res["period"] else "Недостаточно данных.", "",
         f"PBO по всем конфигурациям: **{res['pbo']['pbo']:.2f}**.", "",
         "| конфигурация | сделок | прибыльных | средняя, б.п. | 95% ДИ, б.п. | PSR | DSR | ¼-Келли, б.п./сделку |",
         "|---|---|---|---|---|---|---|---|"]
    for c in [*res["configs"], *res["baselines"]]:
        t = c.stats
        if t.get("trades", 0) < 10:
            continue
        kel = f"{t['kelly_logg_bps']:+.1f}" if "kelly_logg_bps" in t else "—"
        L.append(f"| {c.name} | {t['trades']} | {t['hit']:.1%} | {t['mean_bps']:+.1f} | [{t['ci_lo_bps']:+.0f}; {t['ci_hi_bps']:+.0f}] | "
                 f"{t['psr']:.2f} | {t['dsr']:.2f} | {kel} |")
    L += ["", "**H1 — управление размером по волатильности** (барная доходность на тестовом периоде):", "",
          "| стратегия | Sharpe | CAGR | макс. просадка |", "|---|---|---|---|"]
    for k, v in res["vol"].items():
        L.append(f"| {k} | {v['sharpe']:+.2f} | {v['cagr']:+.1%} | {v['mdd']:.1%} |")
    L += ["", "## Вердикт",
          ("**Гейт пройден**" if res["gate"] else "**Гейт НЕ пройден**") +
          (f": лучшая — {b.name}, DSR {b.stats['dsr']:.2f}, средняя {b.stats['mean_bps']:+.1f} б.п. "
           f"(95% ДИ [{b.stats['ci_lo_bps']:+.0f}; {b.stats['ci_hi_bps']:+.0f}]), {b.stats['trades']} сделок, "
           f"≈{res['signals_per_month']:.1f} сигналов в месяц; "
           f"{'бьёт' if res['beats'] else 'НЕ бьёт'} ориентир «всегда long»." if b else "."),
          "", f"**H3 (мета-модель улучшает базовый сигнал): {'подтверждена' if res['m2_improves'] else 'не подтверждена'}**"
          + (f" — лучшая M2 {res['m2_best'].name}: {res['m2_best'].stats['mean_bps']:+.1f} б.п. против "
             f"{res['configs'][0].stats.get('mean_bps', float('nan')):+.1f} у primary." if res["m2_best"] else "."),
          "", "Требования гейта: DSR > 0.95, PBO < 0.2, нижняя граница 95% ДИ > 0, ≥ 100 сделок, лучше «всегда long». "
          "Меньше ~2 сигналов в месяц — продукт нежизнеспособен, даже если гейт пройден."]
    suspicious = [c for c in [*res["configs"], *res["baselines"]] if c.stats.get("trades", 0) >= 100 and c.stats["hit"] > 0.62]
    if suspicious:
        L += ["", f"⚠️ **Подозрение на утечку данных:** {', '.join(c.name for c in suspicious)} — больше 62 % прибыльных сделок "
              "на ≥ 100 сделках. На реальном рынке такое почти всегда означает, что признак видит будущее. Проверьте время признаков "
              "и меток, прогоните --placebo."]
    if res["placebo"]:
        L += ["", "_Плацебо: базовый сигнал настоящий, признаки мета-модели перемешаны блоками. Если здесь H3 «подтверждается» — "
              "в конвейере утечка, и результатам основного прогона верить нельзя._"]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tf", default="4h", choices=list(BARS_PER_DAY))
    ap.add_argument("--hours", type=int, default=48, help="вертикальный барьер, часов (24/48/72)")
    ap.add_argument("--fee-bps", type=float, default=5.0, help="комиссия за сторону: 5 ≈ taker Binance USDT-M")
    ap.add_argument("--tp", type=float, default=2.0)
    ap.add_argument("--sl", type=float, default=1.5)
    ap.add_argument("--placebo", action="store_true")
    a = ap.parse_args()
    df = load(a.data)
    name = Path(a.data).stem
    res = evaluate_v1(df, a.tf, a.hours, a.fee_bps, a.tp, a.sl, placebo=a.placebo)
    rep = report_v1(res, name)
    out = Path("reports") / f"v1_{name}_{a.hours}h{'_placebo' if a.placebo else ''}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(rep, encoding="utf-8")
    print(rep, f"\n\nСохранено: {out}", sep="")


if __name__ == "__main__":
    main()
