"""Оценка в формате реальных сигналов: вход → стоп/цель на k·σ·√h → выход по времени.

Для каждой конфигурации (модель × правило входа, плюс простые правила по отдельным признакам — они тоже
«перебирались», поэтому входят в поправки) строится ряд доходностей на одной и той же тестовой шкале времени.
По этой матрице считаем PBO; по каждой конфигурации — PSR, MinTRL, DSR, бутстрап-интервал, рост капитала по Келли
и разбивку по режимам рынка.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as st

from .evaluate import Folds, _make_model, deflated_sharpe, nonoverlap
from .labels import ev_threshold, kelly_binary
from .stats import min_track_record, pbo_cscv, psr, stationary_bootstrap_ci


@dataclass
class Config:
    name: str
    side: np.ndarray                 # позиция на каждом тестовом моменте: +1 / −1 / 0
    p: np.ndarray | None = None      # вероятность выигрыша выбранной стороны (для Келли)
    win: np.ndarray | None = None    # размер выигрыша и проигрыша (для Келли)
    loss: np.ndarray | None = None
    r_own: np.ndarray | None = None  # результат «со стороны сигнала» для КАЖДОГО момента, если у конфигураций разные
                                     # направления (барьеры асимметричны, short ≠ −long); тогда side — маска 0/1
    rets: np.ndarray = field(default_factory=lambda: np.array([]))
    stats: dict = field(default_factory=dict)


def evaluate_barrier(X: pd.DataFrame, tb: pd.DataFrame, horizon: int, folds: Folds, fee_bps: float,
                     periods_per_year: float, kinds=("logit", "lgbm"), thresholds=(0.0, 0.03, 0.06),
                     factor_rules: bool = True) -> dict:
    cost = 2 * fee_bps / 1e4
    valid = ~X.isna().any(axis=1).values & tb["ret"].notna().values
    y = (tb["ret"].values > 0).astype(int)
    Xv = X.values

    # тестовая шкала: непересекающиеся бары всех тестовых блоков
    test_idx = np.concatenate([nonoverlap(te, horizon) for _, te in folds])
    test_idx = test_idx[valid[test_idx]]
    pos = {t: i for i, t in enumerate(test_idx)}
    probs = {k: np.full(len(test_idx), np.nan) for k in kinds}
    fsides = {c: np.zeros(len(test_idx)) for c in X.columns} if factor_rules else {}

    for tr, te in folds:
        tr = tr[valid[tr]]
        te = nonoverlap(te, horizon)
        te = te[valid[te]]
        if len(tr) < 200 or not len(te):
            continue
        ii = [pos[t] for t in te]
        for k in kinds:
            try:
                m = _make_model(k).fit(Xv[tr], y[tr])
            except ImportError:
                continue
            probs[k][ii] = m.predict_proba(Xv[te])[:, 1]
        for j, c in enumerate(X.columns if factor_rules else []):
            x = Xv[:, j]
            rho = st.spearmanr(x[tr], tb["ret"].values[tr])[0]
            orient = np.sign(rho) if np.isfinite(rho) and rho != 0 else 1.0
            fsides[c][ii] = np.sign((x[te] - np.median(x[tr])) * orient)

    covered = ~np.isnan(probs[kinds[0]]) if kinds else np.ones(len(test_idx), bool)
    test_idx = test_idx[covered]
    r_long = tb["ret"].values[test_idx]
    width = tb["width"].values[test_idx]

    configs: list[Config] = []
    for k in kinds:
        p = probs[k][covered]
        if np.isnan(p).all():
            continue
        for th in thresholds:
            side = np.where(np.abs(p - 0.5) > th, np.sign(p - 0.5), 0.0)
            configs.append(Config(f"{k} |p−½|>{th:.2f}", side, p))
        configs.append(Config(f"{k} EV>издержек", ev_threshold(p, width, cost), p))
    for c, s in fsides.items():
        configs.append(Config(f"признак {c}", s[covered]))

    for cfg in configs:
        if cfg.p is not None:                              # симметричные барьеры: выигрыш и проигрыш = ширина
            cfg.p = np.where(cfg.side > 0, cfg.p, 1 - cfg.p)
            cfg.win = cfg.loss = width
    pbo = score_configs(configs, r_long, cost, periods_per_year)
    return {"configs": configs, "pbo": pbo, "test_idx": test_idx, "n_test": len(test_idx)}


def regime_breakdown(cfg: Config, X: pd.DataFrame, test_idx: np.ndarray) -> list[tuple[str, int, float, float]]:
    """Где конфигурация работает: по волатильности (z-оценка) и по тренду (знак моментума за 28 дней) на момент входа."""
    traded = cfg.side != 0
    idx = test_idx[traded]
    r = cfg.rets
    out = []
    if "vol_z_28d" in X:
        vz = X["vol_z_28d"].values[idx]
        for name, m in (("низкая волатильность", vz < -0.5), ("средняя волатильность", (vz >= -0.5) & (vz <= 0.5)),
                        ("высокая волатильность", vz > 0.5)):
            if m.sum() >= 10:
                out.append((name, int(m.sum()), float(np.mean(r[m] > -1e-12)), float(r[m].mean() * 1e4)))
    if "mom_28d" in X:
        mo = X["mom_28d"].values[idx]
        for name, m in (("восходящий тренд 28д", mo > 0), ("нисходящий тренд 28д", mo <= 0)):
            if m.sum() >= 10:
                out.append((name, int(m.sum()), float(np.mean(r[m] > -1e-12)), float(r[m].mean() * 1e4)))
    return out


def score_configs(configs: list[Config], r_side: np.ndarray, cost: float, periods_per_year: float,
                  n_trials: int | None = None) -> dict:
    """Общий подсчёт: для каждой конфигурации — доходности сделок (side·r − издержки), PSR, MinTRL, DSR,
    бутстрап-интервал, рост капитала по ¼-Келли; по матрице всех конфигураций — PBO.
    r_side — результат long-позиции (если side = ±1 означает long/short) или результат «со стороны сигнала».
    n_trials — N для DSR (все испытания из реестра); по умолчанию — только конфигурации этого прогона."""
    n_trials = len(configs) if n_trials is None else max(n_trials, len(configs))
    n_test = len(r_side)
    M = np.zeros((n_test, len(configs)))
    for i, cfg in enumerate(configs):
        traded = cfg.side != 0
        r = cfg.r_own[traded] if cfg.r_own is not None else cfg.side[traded] * r_side[traded]
        M[traded, i] = r - cost
        cfg.rets = M[traded, i]
    pbo = pbo_cscv(M)
    for c in configs:
        r = c.rets
        n = len(r)
        if n < 10:
            c.stats = {"trades": n}
            continue
        hits = int(np.sum(r + cost > 0))
        lo, hi = stationary_bootstrap_ci(r, block=3.0, n_boot=600)
        s = {
            "trades": n, "coverage": n / n_test, "hit": hits / n,
            "p": st.binomtest(hits, n, 0.5, alternative="greater").pvalue,
            "mean_bps": r.mean() * 1e4, "ci_lo_bps": lo * 1e4, "ci_hi_bps": hi * 1e4,
            "sharpe": r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year * n / n_test) if r.std() > 0 else 0.0,
            "psr": psr(r), "min_trl": min_track_record(r), "dsr": deflated_sharpe(r, n_trials), "n_trials": n_trials,
        }
        if c.p is not None and c.win is not None:
            traded = c.side != 0
            f = kelly_binary(c.p[traded], c.win[traded], c.loss[traded])
            s["kelly_logg_bps"] = float(np.mean(np.log1p(f * np.expm1(r)))) * 1e4
        c.stats = s
    return pbo
