"""Честная проверка предсказательной силы: walk-forward с зазором, непересекающиеся сделки, издержки,
поправка на множественное тестирование. Всё считается только на данных «из будущего» относительно обучения."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class Folds:
    """Расширяющееся окно: train = [0, s − h), test = [s, s + L). Последние h баров трейна выброшены (purge),
    потому что их метки заглядывают в тестовый период."""
    n: int
    horizon: int
    min_train: int
    test_len: int

    def __iter__(self):
        s = self.min_train
        while s < self.n:
            e = min(self.n, s + self.test_len)
            yield np.arange(0, max(0, s - self.horizon)), np.arange(s, e)
            s = e


def nonoverlap(idx: np.ndarray, horizon: int) -> np.ndarray:
    """Каждый h-й бар теста — сделки не пересекаются по времени, поэтому p-value честные."""
    return idx[::horizon]


# ---------------------------------------------------------------- однофакторные тесты
@dataclass
class FactorResult:
    name: str
    n: int
    hit: float
    p_value: float
    ic: float
    ic_t: float
    q_value: float = 1.0


def factor_tests(X: pd.DataFrame, fwd: pd.Series, horizon: int, folds: Folds) -> list[FactorResult]:
    """Для каждого признака: знак связи выбирается на трейне, попадание в направление проверяется на тесте."""
    out = []
    y = np.sign(fwd.values)
    for col in X.columns:
        x = X[col].values
        hits, ics = [], []
        for tr, te in folds:
            tr = tr[~np.isnan(x[tr]) & ~np.isnan(fwd.values[tr])]
            te = nonoverlap(te, horizon)
            te = te[~np.isnan(x[te]) & ~np.isnan(fwd.values[te]) & (y[te] != 0)]
            if len(tr) < 50 or len(te) < 5:
                continue
            med = np.median(x[tr])
            orient = np.sign(stats.spearmanr(x[tr], fwd.values[tr])[0] or 1.0)
            pred = np.sign((x[te] - med) * orient)
            hits.extend((pred == y[te]).tolist())
            if len(te) > 8:
                ics.append(stats.spearmanr(x[te] * orient, fwd.values[te])[0])
        n = len(hits)
        if n < 20:
            continue
        k = int(np.sum(hits))
        p = stats.binomtest(k, n, 0.5, alternative="greater").pvalue
        ics = np.array([i for i in ics if not np.isnan(i)])
        ic_t = ics.mean() / (ics.std(ddof=1) / math.sqrt(len(ics))) if len(ics) > 2 and ics.std() > 0 else 0.0
        out.append(FactorResult(col, n, k / n, p, float(ics.mean()) if len(ics) else 0.0, float(ic_t)))
    # поправка Бенджамини–Хохберга на множественное тестирование
    ps = np.array([r.p_value for r in out])
    if len(ps):
        order = np.argsort(ps)
        q = ps[order] * len(ps) / (np.arange(len(ps)) + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
        for i, o in enumerate(order):
            out[o].q_value = float(min(1.0, q[i]))
    return sorted(out, key=lambda r: r.p_value)


# ---------------------------------------------------------------- модель
@dataclass
class ModelResult:
    model: str
    threshold: float
    trades: int
    coverage: float
    hit: float
    p_value: float
    mean_ret_bps: float
    sharpe: float
    trade_rets: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    dsr: float = 0.0


def _make_model(kind: str):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
    if kind == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=100,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1)
    raise ValueError(kind)


def model_tests(X: pd.DataFrame, fwd: pd.Series, horizon: int, folds: Folds, periods_per_year: float,
                fee_bps: float = 10.0, thresholds=(0.0, 0.03, 0.06), kinds=("logit", "lgbm")) -> list[ModelResult]:
    """Классификатор направления. Сделка только если |p − 0.5| > порога (иначе «воздержаться»).
    Доходность сделки = направление × форвард-доходность − комиссия за вход и выход."""
    y = (fwd.values > 0).astype(int)
    valid = ~X.isna().any(axis=1).values & ~np.isnan(fwd.values)
    Xv = X.values
    probs = {k: np.full(len(X), np.nan) for k in kinds}
    for tr, te in folds:
        tr, te = tr[valid[tr]], nonoverlap(te, horizon)
        te = te[valid[te]]
        if len(tr) < 200 or not len(te):
            continue
        for k in kinds:
            try:
                m = _make_model(k).fit(Xv[tr], y[tr])
            except ImportError:
                continue
            probs[k][te] = m.predict_proba(Xv[te])[:, 1]
    out = []
    cost = 2 * fee_bps / 1e4
    for k, p in probs.items():
        idx = np.where(~np.isnan(p))[0]
        if not len(idx):
            continue
        for th in thresholds:
            take = idx[np.abs(p[idx] - 0.5) > th]
            if len(take) < 10:
                continue
            side = np.where(p[take] > 0.5, 1.0, -1.0)
            rets = side * fwd.values[take] - cost
            hits = int(np.sum(side * fwd.values[take] > 0))
            pv = stats.binomtest(hits, len(take), 0.5, alternative="greater").pvalue
            sharpe = rets.mean() / rets.std(ddof=1) * math.sqrt(periods_per_year) if rets.std() > 0 else 0.0
            out.append(ModelResult(k, th, len(take), len(take) / len(idx), hits / len(take), pv, rets.mean() * 1e4, sharpe, rets))
    return out


def deflated_sharpe(rets: np.ndarray, n_trials: int, sr_trials_std: float | None = None) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014): вероятность, что настоящий Sharpe > 0
    с учётом числа испробованных вариантов, асимметрии и толстых хвостов. Считается по Sharpe за сделку.

    Порог SR0 — ожидаемый максимум Sharpe из n_trials попыток, если преимущества нет ни у одной.
    Разброс оценки Sharpe при нулевой гипотезе ≈ 1/√(n−1); его и берём по умолчанию. Разброс Sharpe между
    нашими конфигурациями для этого не годится: когда настоящее преимущество есть у многих коррелированных
    конфигураций сразу, он раздувается и штрафует реальный сигнал как шум."""
    n = len(rets)
    if n < 10 or rets.std() == 0:
        return 0.0
    sr = rets.mean() / rets.std(ddof=1)
    g3, g4 = stats.skew(rets), stats.kurtosis(rets, fisher=False)
    emc = 0.5772156649
    n_trials = max(2, n_trials)
    if sr_trials_std is None:
        sr_trials_std = 1 / math.sqrt(n - 1)
    sr0 = sr_trials_std * ((1 - emc) * stats.norm.ppf(1 - 1 / n_trials) + emc * stats.norm.ppf(1 - 1 / (n_trials * math.e)))
    denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(n - 1) / denom))


def attach_dsr(results: list[ModelResult], extra_trials: int = 0) -> None:
    for r in results:
        r.dsr = deflated_sharpe(r.trade_rets, len(results) + extra_trials)
