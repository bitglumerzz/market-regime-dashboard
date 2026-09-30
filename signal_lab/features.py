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


def donchian_ensemble(close: pd.Series, lookbacks_bars: list[int]) -> pd.Series:
    """Ансамбль пробоев канала Дончиана (по мотивам Zarattini et al. 2025), симметричный: для каждого окна L —
    вход в long, когда close ≥ max(close за предыдущие L баров), в short — когда close ≤ min; выход, когда цена
    возвращается за середину канала (max+min)/2. Итог — среднее состояние по окнам, −1…1.
    (У авторов стратегия long-only; шорт оставлен признаком — решать, торговать ли его, будет мета-модель.)"""
    c = close.values
    out = np.zeros(len(c))
    for L in lookbacks_bars:
        hi = close.shift(1).rolling(L).max().values
        lo = close.shift(1).rolling(L).min().values
        pos = 0
        state = np.zeros(len(c))
        for t in range(len(c)):
            if np.isnan(hi[t]):
                continue
            mid = (hi[t] + lo[t]) / 2
            if pos == 1 and c[t] < mid or pos == -1 and c[t] > mid:
                pos = 0
            if pos == 0:
                pos = 1 if c[t] >= hi[t] else -1 if c[t] <= lo[t] else 0
            state[t] = pos
        out += state
    res = pd.Series(out / len(lookbacks_bars), index=close.index)
    res[close.shift(1).rolling(max(lookbacks_bars)).max().isna()] = np.nan
    return res


def build_features_v1(df: pd.DataFrame, bars_per_day: int) -> pd.DataFrame:
    """Признаки модели v1 из docs/research/directional-edge-2026-09.md (раздел 3.2): F1–F7 и F12.
    F8 (DVOL), F9 (бета к BTC), F10 (макро), F11 (HMM) требуют внешних рядов — добавляются колонками в df,
    если они есть: dvol, hmm_p_*, dxy."""
    d = lambda days: max(2, int(round(days * bars_per_day)))
    c = df["close"]
    r = np.log(c).diff()
    f = pd.DataFrame(index=df.index)
    f["F1_donchian"] = donchian_ensemble(c, [d(10), d(20), d(60), d(120)])
    for days in (20, 50, 100):
        f[f"F2_log_p_ma{days}"] = np.log(c / c.rolling(d(days)).mean())
    s30 = r.ewm(span=d(30), min_periods=d(15)).std()
    for w, days in (("1w", 7), ("4w", 28)):
        f[f"F3_tsmom_{w}"] = np.log(c / c.shift(d(days))) / (s30 * np.sqrt(d(days)))
    f["F4_vol_ratio_5_60"] = r.ewm(span=d(5)).std() / r.ewm(span=d(60), min_periods=d(30)).std()
    f["F4_vol_20d"] = r.ewm(span=d(20), min_periods=d(10)).std() * np.sqrt(365 * bars_per_day)
    if "taker_buy_volume" in df:
        tb, v = df["taker_buy_volume"], df["volume"].replace(0, np.nan)
        for w, hours in (("4h", 4), ("24h", 24)):
            n = max(1, int(round(hours * bars_per_day / 24)))
            ti = (2 * tb.rolling(n).sum() - v.rolling(n).sum()) / v.rolling(n).sum()
            f[f"F5_taker_imb_{w}_z"] = _z(ti, d(30))
    if "oi" in df:
        oi = np.log(df["oi"].ffill())
        f["F6_doi_24h"] = oi.diff(d(1))
        f["F6_doi_7d"] = oi.diff(d(7))
    if "funding" in df:
        fr = df["funding"].ffill()
        f["F7_funding_30d"] = fr.rolling(d(30), min_periods=d(10)).mean()
        sign = np.sign(fr)
        run = sign.groupby((sign != sign.shift()).cumsum()).cumcount() + 1
        f["F7_funding_sign_run"] = sign * run / bars_per_day               # сколько дней держится знак, со знаком
    for col in df.columns:
        if col == "dvol" or col.startswith("hmm_p_") or col == "dxy":
            f[f"X_{col}"] = df[col].ffill()
    if isinstance(df.index, pd.DatetimeIndex):
        h = df.index.hour
        f["F12_hour_sin"], f["F12_hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
        f["F12_weekday"] = df.index.dayofweek.astype(float)
        f["F12_h_to_funding"] = ((8 - h % 8) % 8).astype(float)
    return f.replace([np.inf, -np.inf], np.nan)


def primary_side(f: pd.DataFrame, band: float = 0.2) -> pd.Series:
    """Базовый (primary) трендовый сигнал: ансамбль Дончиана (−1…1) + tanh(TSMOM 4 недели), поровну.
    |оценка| < band → нет сигнала (0)."""
    score = 0.5 * f["F1_donchian"] + 0.5 * np.tanh(f["F3_tsmom_4w"])
    return pd.Series(np.where(score > band, 1.0, np.where(score < -band, -1.0, 0.0)), index=f.index).where(score.notna())


def _rsi(c: pd.Series, n: int) -> pd.Series:
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """Лог-доходность за следующие horizon баров: (t, t+h]. Это цель, а не признак."""
    return np.log(close.shift(-horizon) / close)
