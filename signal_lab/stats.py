"""Статистика, которая отличает умение от везения.

PSR / MinTRL — Bailey & López de Prado (2012), «The Sharpe Ratio Efficient Frontier».
PBO (CSCV)  — Bailey, Borwein, López de Prado, Zhu (2017), «The Probability of Backtest Overfitting».
Stationary bootstrap — Politis & Romano (1994).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats


def _moments(r: np.ndarray) -> tuple[float, float, float]:
    sr = r.mean() / r.std(ddof=1)
    return sr, float(stats.skew(r)), float(stats.kurtosis(r, fisher=False))


def psr(r: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio: P(истинный Sharpe за сделку > sr_benchmark) с поправкой на асимметрию и хвосты."""
    r = np.asarray(r, float)
    if len(r) < 10 or r.std() == 0:
        return 0.0
    sr, g3, g4 = _moments(r)
    denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(stats.norm.cdf((sr - sr_benchmark) * math.sqrt(len(r) - 1) / denom))


def min_track_record(r: np.ndarray, sr_benchmark: float = 0.0, alpha: float = 0.05) -> float:
    """Minimum Track Record Length: сколько сделок нужно, чтобы с уверенностью 1−alpha утверждать Sharpe > benchmark."""
    r = np.asarray(r, float)
    if len(r) < 10 or r.std() == 0:
        return math.inf
    sr, g3, g4 = _moments(r)
    if sr <= sr_benchmark:
        return math.inf
    z = stats.norm.ppf(1 - alpha)
    return float(1 + (1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) * (z / (sr - sr_benchmark)) ** 2)


def stationary_bootstrap_ci(x: np.ndarray, stat=np.mean, block: float = 5.0, n_boot: int = 2000,
                            alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Доверительный интервал для статистики ряда с зависимостью: блоки случайной (геометрической) длины."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 10:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    p = 1.0 / block
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        i = rng.integers(n)
        for t in range(n):
            idx[t] = i
            i = rng.integers(n) if rng.random() < p else (i + 1) % n
        out[b] = stat(x[idx])
    return float(np.quantile(out, alpha / 2)), float(np.quantile(out, 1 - alpha / 2))


def pbo_cscv(M: np.ndarray, n_splits: int = 10) -> dict:
    """Probability of Backtest Overfitting через Combinatorially Symmetric Cross-Validation.

    M — матрица T×N: доходности N конфигураций в одни и те же T моментов времени (0, если конфигурация воздержалась).
    Для каждого разбиения блоков пополам: выбираем лучшую конфигурацию на одной половине (IS) и смотрим её ранг
    на другой (OOS). PBO = доля разбиений, где «лучшая на истории» оказалась ниже медианы на новых данных.
    ~0.5 — выбор конфигурации ничем не лучше случайного; < 0.2 — выбор устойчив.
    """
    M = np.asarray(M, float)
    T, N = M.shape
    if N < 2 or T < n_splits * 4:
        return {"pbo": math.nan, "n_combos": 0, "degradation": math.nan}
    blocks = np.array_split(np.arange(T), n_splits)
    sharpe = lambda X: X.mean(0) / np.where(X.std(0, ddof=1) > 0, X.std(0, ddof=1), np.inf)
    logits, deg = [], []
    for combo in itertools.combinations(range(n_splits), n_splits // 2):
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(n_splits) if i not in combo])
        s_is, s_oos = sharpe(M[is_idx]), sharpe(M[oos_idx])
        best = int(np.argmax(s_is))
        rank = stats.rankdata(s_oos)[best] / (N + 1)            # относительный ранг лучшей IS-конфигурации на OOS
        logits.append(math.log(rank / (1 - rank)))
        deg.append(s_oos[best] - s_is[best])
    logits = np.array(logits)
    return {"pbo": float(np.mean(logits <= 0)), "n_combos": len(logits), "degradation": float(np.mean(deg))}
