"""Кандидаты в предикторы направления. Все признаки каузальные: значение на баре t использует только данные до t включительно.

Вход — DataFrame с колонками open, high, low, close, volume (обязательно) и по возможности:
taker_buy_volume (агрессивные покупки), funding (ставка финансирования перпетуала), oi (открытый интерес).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _z(x: pd.Series, n: int) -> pd.Series:
    m, s = x.rolling(n, min_periods=n // 2).mean(), x.rolling(n, min_periods=n // 2).std()
    return (x - m) / s.replace(0, np.nan)


def build_features(df: pd.DataFrame, bars_per_day: int) -> pd.DataFrame:
    """Набор признаков. Окна заданы в днях и переводятся в бары, чтобы одинаково работать на 1h/4h/1d."""
    d = lambda days: max(2, int(round(days * bars_per_day)))
    c, v = df["close"], df["volume"]
    r = np.log(c).diff()
    f = pd.DataFrame(index=df.index)

    # --- цена: тренд, моментум, разворот
    for days in (1, 3, 7, 14, 28):
        f[f"mom_{days}d"] = np.log(c / c.shift(d(days)))
    vol7 = r.rolling(d(7)).std()
    f["tsmom_28d_vs"] = f["mom_28d"] / (vol7 * np.sqrt(d(28)))           # моментум, нормированный на волатильность
    f["ma_gap_20d"] = c / c.rolling(d(20)).mean() - 1
    f["ma_cross_7_28"] = c.rolling(d(7)).mean() / c.rolling(d(28)).mean() - 1
    hi, lo = df["high"].rolling(d(20)).max(), df["low"].rolling(d(20)).min()
    f["donchian_pos_20d"] = (c - lo) / (hi - lo).replace(0, np.nan) - 0.5
    f["rev_1bar"] = -r                                                     # краткосрочный разворот
    f["rsi_14"] = _rsi(c, 14) / 100 - 0.5

    # --- волатильность и объём (часто работают как фильтры режима, а не направление)
    f["vol_ratio_1_7"] = r.rolling(d(1)).std() / vol7
    f["vol_z_28d"] = _z(r.rolling(d(1)).std(), d(28))
    f["volume_z_7d"] = _z(np.log(v.replace(0, np.nan)), d(7))

    # --- поток ордеров
    if "taker_buy_volume" in df:
        imb = 2 * df["taker_buy_volume"] / v.replace(0, np.nan) - 1
        f["taker_imb_1bar"] = imb
        f["taker_imb_1d"] = imb.rolling(d(1)).mean()

    # --- позиционирование на деривативах
    if "funding" in df:
        fr = df["funding"].ffill()
        f["funding"] = fr
        f["funding_z_14d"] = _z(fr, d(14))
    if "oi" in df:
        oi = np.log(df["oi"].ffill())
        f["oi_chg_1d"] = oi.diff(d(1))
        f["oi_x_price_1d"] = f["oi_chg_1d"] * np.sign(f["mom_1d"])       # рост OI вместе с движением цены

    return f.replace([np.inf, -np.inf], np.nan)


def _rsi(c: pd.Series, n: int) -> pd.Series:
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """Лог-доходность за следующие horizon баров: (t, t+h]. Это цель, а не признак."""
    return np.log(close.shift(-horizon) / close)
