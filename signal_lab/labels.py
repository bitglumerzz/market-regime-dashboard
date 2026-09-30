"""Метки triple-barrier (López de Prado, AFML гл. 3) и размер позиции по Келли.

Сделка оценивается так, как её ведёт подписчик сигнала: цель и стоп на расстоянии k·σ·√h от входа
(σ — текущая волатильность за бар), и ограничение по времени h баров. Что наступит раньше — то и результат.
Барьеры симметричны, поэтому для шорта исход зеркален: касание верхнего барьера — его стоп.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ewm_vol(close: pd.Series, span: int) -> pd.Series:
    """Волатильность лог-доходности за бар (экспоненциальная, каузальная)."""
    return np.log(close).diff().ewm(span=span, min_periods=span // 2).std()


def triple_barrier(df: pd.DataFrame, horizon: int, k: float = 1.0, vol_span: int = 42) -> pd.DataFrame:
    """Для каждого бара t: какой барьер коснулся первым на (t, t+h] и с каким результатом (лог-доходность long-позиции).

    Возвращает колонки: label (+1 верхний, −1 нижний, 0 по времени), ret (результат long в лог-доходности,
    при касании — ровно ±барьер), width (k·σ·√h), exit_bars.
    Если в одной свече задеты оба барьера, консервативно считаем, что первым сработал стоп (для long — нижний).
    """
    c = df["close"].values
    hi, lo = df["high"].values, df["low"].values
    sig = ewm_vol(df["close"], vol_span).values
    n = len(c)
    label = np.full(n, np.nan)
    ret = np.full(n, np.nan)
    exit_bars = np.full(n, np.nan)
    width = k * sig * np.sqrt(horizon)
    for t in range(n - horizon):
        w = width[t]
        if not np.isfinite(w) or w <= 0:
            continue
        up, dn = c[t] * np.exp(w), c[t] * np.exp(-w)
        lab, r, eb = 0, np.log(c[t + horizon] / c[t]), horizon
        for j in range(1, horizon + 1):
            touch_dn, touch_up = lo[t + j] <= dn, hi[t + j] >= up
            if touch_dn:                                       # стоп проверяем первым — консервативно
                lab, r, eb = -1, -w, j
                break
            if touch_up:
                lab, r, eb = 1, w, j
                break
        label[t], ret[t], exit_bars[t] = lab, r, eb
    return pd.DataFrame({"label": label, "ret": ret, "width": width, "exit_bars": exit_bars}, index=df.index)


def ev_threshold(p_up: np.ndarray, width: np.ndarray, cost: float) -> np.ndarray:
    """Решение по матожиданию: при симметричных барьерах EV(long) ≈ (2p−1)·w − cost, EV(short) ≈ (1−2p)·w − cost.
    Возвращает +1 / −1 / 0 (воздержаться, если ни одна сторона не окупает издержки)."""
    edge = (2 * p_up - 1) * width
    return np.where(edge > cost, 1.0, np.where(-edge > cost, -1.0, 0.0))


def kelly_fraction(p_win: np.ndarray, width: np.ndarray, frac: float = 0.25, cap: float = 1.0) -> np.ndarray:
    """Доля капитала по Келли для бинарного исхода ±w: f* = (2p−1)/w (в долях капитала на единицу лог-движения).
    Используем дробный Келли (по умолчанию ¼) и потолок cap — полный Келли слишком агрессивен при ошибке в p."""
    f = frac * (2 * p_win - 1) / np.where(width > 0, width, np.inf)
    return np.clip(f, 0.0, cap)
