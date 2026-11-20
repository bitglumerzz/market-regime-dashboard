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
from .registry import HOLDOUT_START, before_holdout, holdout_opened, n_trials, open_holdout, register, trade_moments
from .run import BARS_PER_DAY, load


def _fit_calibrated(kind: str, X: np.ndarray, y: np.ndarray, horizon: int, w: np.ndarray | None = None,
                    conformal: float | None = None):
    """Модель на первых 75% обучения + изотоническая калибровка на последних 25% (между ними зазор h баров).
    w — веса уникальности меток (пересекающиеся сделки весят меньше).
    conformal=α (H9): калибровочный хвост делится пополам — изотоническая регрессия на первой половине, split-conformal
    (MAPIE, мера LAC) на второй; возвращается пара функций (p, «множество = {выигрыш}»)."""
    cut = int(len(X) * 0.75)
    fit_idx, cal_idx = np.arange(0, max(0, cut - horizon)), np.arange(cut, len(X))
    if len(fit_idx) < 150 or len(cal_idx) < 50 or len(np.unique(y[fit_idx])) < 2:
        return None
    sw = None if w is None else w[fit_idx]
    m = _make_model(kind)
    m = m.fit(X[fit_idx], y[fit_idx], **({"logisticregression__sample_weight": sw} if kind == "logit" else {"sample_weight": sw}))
    if conformal is None:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(m.predict_proba(X[cal_idx])[:, 1], y[cal_idx])
        return lambda Z: iso.predict(m.predict_proba(Z)[:, 1])
    half = len(cal_idx) // 2
    iso_idx, conf_idx = cal_idx[:half], cal_idx[half:]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(m.predict_proba(X[iso_idx])[:, 1], y[iso_idx])
    from mapie.classification import SplitConformalClassifier
    est = _Calibrated(m, iso)
    mc = SplitConformalClassifier(est, confidence_level=1 - conformal, prefit=True, conformity_score="lac")
    mc.conformalize(X[conf_idx], y[conf_idx].astype(int))
    def win_singleton(Z):
        sets = mc.predict_set(Z)[1][:, :, 0]                 # (n, 2): входит ли класс 0 / 1 в множество
        return sets[:, 1] & ~sets[:, 0]
    return (lambda Z: iso.predict(m.predict_proba(Z)[:, 1])), win_singleton


class _Calibrated:
    """Обёртка «модель + изотоническая калибровка» в интерфейсе sklearn-классификатора для MAPIE (prefit)."""
    _estimator_type = "classifier"

    def __init__(self, m, iso):
        self.m, self.iso, self.classes_ = m, iso, np.array([0, 1])
        self.fitted_ = True

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        p = self.iso.predict(self.m.predict_proba(X)[:, 1])
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def __sklearn_tags__(self):
        from sklearn.utils import Tags, ClassifierTags, TargetTags
        return Tags(estimator_type="classifier", classifier_tags=ClassifierTags(), target_tags=TargetTags(required=True))


def _share_folds_better(p: np.ndarray, y: np.ndarray, p0: np.ndarray, fold: np.ndarray) -> float:
    """Доля тестовых блоков, где log-loss M2 ниже, чем у константы (доли выигрышей на обучении)."""
    ll = lambda q: -(y * np.log(np.clip(q, 1e-4, 1 - 1e-4)) + (1 - y) * np.log(np.clip(1 - q, 1e-4, 1)))
    a, b = ll(p), ll(p0)
    better = [a[fold == f].mean() < b[fold == f].mean() for f in np.unique(fold)]
    return float(np.mean(better)) if better else float("nan")


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
                taus=(0.55, 0.60), test_days: int = 90, min_train_days: int = 365, placebo: bool = False, seed: int = 0,
                drop: tuple[str, ...] = (), n_trials_prior: int = 0, holdout: bool = False,
                cutoff: pd.Timestamp | None = HOLDOUT_START, conformal: float | None = None) -> dict:
    """drop — префиксы признаков, которые убираем (ablation: «F5_», «X_hmm»…); n_trials_prior — сколько испытаний уже
    в реестре (N для DSR = они + конфигурации этого прогона). По умолчанию данные обрезаются до holdout;
    holdout=True — тестовые блоки только внутри holdout (обучение — на всём до него)."""
    is_dt = isinstance(df.index, pd.DatetimeIndex)
    if cutoff is not None and is_dt and not holdout:
        df = before_holdout(df, cutoff)
    bpd = BARS_PER_DAY[tf]
    h = max(1, int(round(horizon_hours * bpd / 24)))
    X = build_features_v1(df, bpd)
    side = primary_side(X).values                           # сторону задаёт тренд — её ablation не трогает
    if drop:
        X = X.drop(columns=[c for c in X.columns if c.startswith(tuple(drop))])
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
    # первый тест — когда набрано min_train_days дней ВАЛИДНЫХ строк (у DVOL/OI история короче свечей)
    start = int(np.searchsorted(np.cumsum(valid), min_train_days * bpd)) + h
    if holdout:
        if cutoff is None or not is_dt:
            raise ValueError("holdout требует DatetimeIndex и дату начала")
        start = max(start, int(np.searchsorted(df.index, cutoff)))
    folds = Folds(len(df), h, min_train=start, test_len=test_days * bpd)

    test_idx = np.concatenate([nonoverlap(te, h) for _, te in folds])
    test_idx = test_idx[valid[test_idx]]
    pos = {t: i for i, t in enumerate(test_idx)}
    probs = {k: np.full(len(test_idx), np.nan) for k in ("logit", "lgbm")}
    single = {k: np.zeros(len(test_idx), bool) for k in probs}          # H9: конформное множество = {выигрыш}
    fold_start, p_base = np.zeros(len(test_idx), int), np.full(len(test_idx), np.nan)
    for tr, te in folds:
        tr = tr[valid[tr]]
        te = nonoverlap(te, h)
        te = te[valid[te]]
        if not len(te):
            continue
        ii = [pos[t] for t in te]
        fold_start[ii], p_base[ii] = te[0], y[tr].mean() if len(tr) else np.nan
        wts = uniqueness_weights(tr + 1, tb["exit_bar"].values[tr])
        for k in probs:
            f = _fit_calibrated(k, Xs.values[tr], y[tr], h, wts, conformal)
            if f is None:
                continue
            if conformal is not None:
                f, g = f
                single[k][ii] = g(Xs.values[te])
            probs[k][ii] = f(Xs.values[te])
    have = ~np.isnan(probs["logit"])
    test_idx, fold_start, p_base = test_idx[have], fold_start[have], p_base[have]
    single = {k: v[have] for k, v in single.items()}
    s = side[test_idx]
    r_sig = tb["ret"].values[test_idx]                       # результат сделки со стороны primary
    tp_w, sl_w = tb["tp_w"].values[test_idx], tb["sl_w"].values[test_idx]
    r_long = tb_long["ret"].values[test_idx]

    configs = [Config("primary без фильтра", np.ones(len(test_idx)))]
    for k, p in probs.items():
        p = p[have]
        ev = p * tp_w - (1 - p) * sl_w - cost
        configs.append(Config(f"primary + M2[{k}] EV>0", (ev > 0).astype(float), p, tp_w, sl_w))
        if conformal is not None:
            configs.append(Config(f"primary + M2[{k}] EV>0 + conformal α={conformal:g}",
                                  ((ev > 0) & single[k]).astype(float), p, tp_w, sl_w))
        for tau in taus:
            configs.append(Config(f"primary + M2[{k}] p>{tau:.2f}", (p > tau).astype(float), p, tp_w, sl_w))
    n_trials = n_trials_prior + len(configs)
    pbo = score_configs(configs, r_sig, cost, 365 * bpd / h, n_trials=n_trials)
    # ориентиры: «всегда long» с теми же барьерами и случайная сторона (результат long-стороны × случайный знак)
    rng = np.random.default_rng(seed + 1)
    base = [Config("ориентир: всегда long (те же барьеры)", np.ones(len(test_idx))),
            Config("ориентир: случайная сторона", rng.choice([-1.0, 1.0], len(test_idx)))]
    score_configs(base, r_long, cost, 365 * bpd / h, n_trials=n_trials)

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
            "fee_bps": fee_bps, "tp": tp, "sl": sl, "placebo": placebo, "n_trials": n_trials, "holdout": holdout,
            "drop": tuple(drop), "features": list(X.columns), "horizon_hours": horizon_hours, "tf": tf,
            # сырьё для ablation/H7/H9: события, вероятности, исходы, фолды
            "test_idx": test_idx, "probs": {k: v[have] for k, v in probs.items()}, "y": y[test_idx], "r_sig": r_sig,
            "tp_w": tp_w, "sl_w": sl_w, "p_base": p_base, "fold_start": fold_start, "X": X,
            "fold_ll_better": {k: _share_folds_better(v[have], y[test_idx], p_base, fold_start) for k, v in probs.items()},
            "conformal": conformal,
            "period": (df.index[test_idx[0]], df.index[test_idx[-1]]) if len(test_idx) else None}


def registry_rows(res: dict, asset: str, hypothesis: str) -> list[dict]:
    """Каждая конфигурация прогона — отдельное испытание в реестре."""
    feats = ";".join(res["features"])
    p0, p1 = res["period"] or (None, None)
    rows = []
    for c in res["configs"]:
        t = c.stats
        model = "primary" if "M2" not in c.name else ("lgbm" if "lgbm" in c.name else "logit")
        rows.append({"hypothesis": hypothesis, "asset": asset, "tf": res["tf"], "model": model, "features": feats,
                     "barriers": f"tp{res['tp']:g}σ/sl{res['sl']:g}σ", "horizon_h": res["horizon_hours"], "rule": c.name,
                     "period_start": p0, "period_end": p1, "holdout": res["holdout"], "hit": t.get("hit"),
                     "mean_bps": t.get("mean_bps"), "ci_lo_bps": t.get("ci_lo_bps"), "ci_hi_bps": t.get("ci_hi_bps"),
                     **trade_moments(c.rets), "n_trials_at_reg": res["n_trials"], "dsr_at_reg": t.get("dsr"),
                     "pbo": res["pbo"]["pbo"], "verdict": "gate" if (c is res["best"] and res["gate"]) else "",
                     "notes": ("placebo; " if res["placebo"] else "") + (f"drop={','.join(res['drop'])}" if res["drop"] else "")})
    return rows


def report_v1(res: dict, name: str) -> str:
    b = res["best"]
    L = [f"# Модель v1 (мета-разметка) · {name}{' · ПЛАЦЕБО' if res['placebo'] else ''}", "",
         f"Горизонт {res['h']} баров ({res['h'] / res['bpd'] * 24:.0f} ч); цель +{res['tp']:g}σ, стоп −{res['sl']:g}σ; "
         f"комиссия {res['fee_bps']:g} б.п. за сторону. Тестовые события: {res['n_test']} (непересекающиеся), "
         f"период {res['period'][0]} — {res['period'][1]}." if res["period"] else "Недостаточно данных.", "",
         f"PBO по всем конфигурациям: **{res['pbo']['pbo']:.2f}**. N испытаний для DSR (реестр + этот прогон): "
         f"**{res['n_trials']}**. " + ("**HOLDOUT** (тест только на отложенном периоде). " if res["holdout"] else
                                       f"Данные обрезаны до {HOLDOUT_START.date()} (holdout не тронут). ")
         + (f"Убраны признаки: {', '.join(res['drop'])}." if res["drop"] else ""), "",
         f"Признаки M2 ({len(res['features'])}): {', '.join(res['features'])}.", "",
         "Log-loss M2 ниже константы (доли выигрышей на обучении) в доле тестовых блоков: "
         + ", ".join(f"{k} {v:.0%}" for k, v in res["fold_ll_better"].items()) + " (критерий H3 — ≥ 80 %).", "",
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
    ap.add_argument("--drop", default="", help="префиксы признаков через запятую, которые убрать (ablation), напр. F5_,X_hmm")
    ap.add_argument("--hypothesis", default="H3", help="метка испытания в reports/trials.csv")
    ap.add_argument("--no-register", action="store_true", help="не писать в реестр (только для отладки кода)")
    ap.add_argument("--holdout", action="store_true", help="ОДНОКРАТНАЯ проверка на holdout замороженной спецификации")
    ap.add_argument("--conformal", type=float, default=None, help="H9: α конформного воздержания (напр. 0.2)")
    a = ap.parse_args()
    df = load(a.data)
    name = Path(a.data).stem
    if a.holdout:
        open_holdout(f"v1 {name} {a.hours}h drop={a.drop}")        # второй раз — исключение
    drop = tuple(p for p in a.drop.split(",") if p)
    res = evaluate_v1(df, a.tf, a.hours, a.fee_bps, a.tp, a.sl, placebo=a.placebo, drop=drop,
                      n_trials_prior=n_trials(), holdout=a.holdout, conformal=a.conformal)
    if not a.no_register:
        register(registry_rows(res, name.split("_")[0], a.hypothesis + ("-placebo" if a.placebo else "")))
    rep = report_v1(res, name)
    tag = (f"_conformal{a.conformal:g}" if a.conformal else "") + (f"_drop-{'-'.join(drop)}" if drop else "") + ("_placebo" if a.placebo else "") + ("_HOLDOUT" if a.holdout else "")
    out = Path("reports") / f"v1_{name}_{a.hours}h{tag}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(rep, encoding="utf-8")
    print(rep, f"\n\nСохранено: {out}", sep="")


if __name__ == "__main__":
    main()
