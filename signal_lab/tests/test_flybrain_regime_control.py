"""H-FlyBrain-2, positive control для гейта новизны (дизайн A): синтетический 4h-ряд, где первые 2/3 — трендовый
режим (моментум), последняя треть — резко иной режим (возврат к среднему + другой профиль волатильности/объёма).
Требования, записанные до рынка: (1) новизна по фильтру Блума с памятью τ даёт пик в первых барах нового режима;
(2) порог новизны, выбранный ТОЛЬКО на обучающей части (квантиль), в новом режиме блокирует заметную долю баров;
(3) трендовая база (ансамбль Дончиана) с таким гейтом имеет меньшую просадку на тесте, чем без гейта."""
import numpy as np
import pandas as pd
import pytest

from signal_lab import flybrain as fb
from signal_lab.features import build_features_v1, donchian_ensemble
from signal_lab.flybrain_novelty import novelty_series

HAS_DATA = (fb.FLYWIRE / "mb_right_nodes.parquet").exists()
pytestmark = pytest.mark.skipif(not HAS_DATA, reason="нужны data/flywire/*")


def two_regimes(n1=6 * 365 * 2, n2=6 * 180, seed=0):
    """Режим 1: доходность с положительной автокорреляцией (тренды), умеренный объём.
    Режим 2: сильный возврат к среднему, вдвое выше волатильность, всплеск объёма и дисбаланса тейкеров."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n1 + n2)
    for t in range(1, n1):
        r[t] = 0.0004 + 0.35 * np.tanh(r[max(0, t - 42):t].sum() / 0.05) * 0.01 + rng.normal(0, 0.01)
    for t in range(n1, n1 + n2):
        r[t] = -0.6 * r[t - 1] + rng.normal(0, 0.02)
    c = 30000 * np.exp(np.cumsum(r))
    idx = pd.date_range("2021-01-01", periods=n1 + n2, freq="4h", tz="UTC")
    v = rng.lognormal(10, .3, n1 + n2) * np.r_[np.ones(n1), np.full(n2, 3.0)]
    tb = v * np.r_[rng.uniform(.45, .55, n1), rng.uniform(.2, .8, n2)]
    hi, lo = c * (1 + np.abs(rng.normal(0, .003, n1 + n2))), c * (1 - np.abs(rng.normal(0, .003, n1 + n2)))
    df = pd.DataFrame({"open": np.r_[c[0], c[:-1]], "high": hi, "low": lo, "close": c, "volume": v, "taker_buy_volume": tb}, index=idx)
    return df, n1


def max_dd(x: np.ndarray) -> float:
    eq = np.cumsum(x)
    return float(np.max(np.maximum.accumulate(eq) - eq)) if len(eq) else 0.0


@pytest.mark.xfail(strict=True, reason="H-FlyBrain-2, зафиксировано 02.10.2026: новизна по ВСЕМ 25 сырым признакам не "
                                        "различает смену режима (0.18 до/0.18 после при k=5 %, τ=12 по формуле ёмкости); "
                                        "хеш хранит общее сходство вектора, где 20 измерений — шум бар-к-бару")
def test_novelty_gate_on_all_raw_features_cuts_drawdown_after_regime_shift():
    c = fb.load_circuit()
    df, n1 = two_regimes()
    X = build_features_v1(df, 6)
    valid = ~X.isna().any(axis=1).values
    first = int(np.argmax(valid))
    tr_end = int(n1 * 0.8)                                               # обучение заканчивается ДО смены режима
    mu, sd = X.values[first:tr_end].mean(0), X.values[first:tr_end].std(0) + 1e-9
    Z = (X.values - mu) / sd
    nov = novelty_series(c, Z, tau_bars=12.0, warmup=first + 6 * 60)          # τ ≈ 0.69·m/k ≈ 14 баров (ёмкость фильтра)
    # (1) пик новизны в первых барах нового режима
    assert np.nanmean(nov[n1:n1 + 24]) > np.nanmean(nov[n1 - 240:n1]) + 0.2
    # (2) порог — 95-й перцентиль новизны на обучении; в новом режиме блокируется заметная доля
    thr = np.nanquantile(nov[first:tr_end], 0.95)
    blocked_new = np.nanmean(nov[n1:] > thr)
    blocked_old = np.nanmean(nov[tr_end:n1] > thr)
    assert blocked_new > 0.3 > blocked_old
    # (3) база: симметричный ансамбль Дончиана (состояние на закрытии t применяется к доходности t+1)
    state = donchian_ensemble(df["close"], [60, 120, 360, 720]).shift(1).fillna(0).values
    r = np.log(df["close"]).diff().fillna(0).values
    base = state * r
    gated = np.where(np.nan_to_num(nov, nan=0.0) > thr, 0.0, state) * r
    te = np.arange(tr_end, len(df))
    dd_base, dd_gated = max_dd(base[te]), max_dd(gated[te])
    assert dd_gated < 0.8 * dd_base, f"gate did not cut drawdown: {dd_gated:.3f} vs {dd_base:.3f}"
    assert gated[te].sum() > base[te].sum() - 0.02                           # и не хуже по сумме (сдвиг режима ломает базу)


def discrete_vocab(X, Z, valid):
    """Маленький дискретный словарь состояний (5 бит, ≤ 32 комбинаций — в пределах ёмкости фильтра ≈ m/k ≈ 20
    различимых паттернов); пороги — знак относительно среднего ОБУЧЕНИЯ (Z уже стандартизован по обучению)."""
    ci = {n: i for i, n in enumerate(X.columns)}
    S = np.c_[(X["F1_donchian"].values > 0), (Z[:, ci["F4_vol_ratio_5_60"]] > 0), (Z[:, ci["F4_vol_20d"]] > 0),
              (Z[:, ci["F5_taker_imb_24h_z"]] > 0), (Z[:, ci["F2_log_p_ma20"]] > 0)].astype(float)
    S[~valid] = np.nan
    return S


@pytest.mark.xfail(strict=True, reason="H-FlyBrain-2, зафиксировано 02.10.2026: гейт новизны на дискретном словаре "
                                        "(5 бит, τ=180) не воспроизводится по seed'ам — отношение просадок 0.65/0.99/0.91, "
                                        "новизна в новом режиме растёт лишь на 2 из 3 seed'ов; линия A′ закрыта по правилу остановки")
def test_novelty_gate_on_discrete_vocabulary_cuts_drawdown_after_regime_shift():
    """Positive control дизайна A′ (словарь + фильтр Блума, τ = 180 баров ≈ 30 дней): на 3 seed'ах синтетики со сменой
    режима гейт по порогу, выбранному на обучении, обязан снижать просадку базы (ансамбль Дончиана) и не ухудшать сумму."""
    from signal_lab import flybrain_novelty as fn
    c = fb.load_circuit()
    m = len(c.kc_ids)
    dd_ratio, nov_up, sum_ok = [], [], []
    for seed in (0, 1, 2):
        df, n1 = two_regimes(seed=seed)
        X = build_features_v1(df, 6)
        valid = ~X.isna().any(axis=1).values
        first, tr_end = int(np.argmax(valid)), int(n1 * 0.8)
        mu, sd = X.values[first:tr_end].mean(0), X.values[first:tr_end].std(0) + 1e-9
        S = discrete_vocab(X, (X.values - mu) / sd, valid)
        fbf = fn.FlyBloom(m, 1.0, 180.0)
        nov = np.full(len(S), np.nan)
        for t in range(len(S)):
            fbf.tick()
            if np.isnan(S[t]).any():
                continue
            tag = fn.flyhash_rates(c, fn.encode_setups(S[t], len(c.pn_ids)))
            if t >= first + 360:
                nov[t] = fbf.novelty(tag)
            fbf.insert(tag)
        thr = np.nanquantile(nov[first:tr_end], 0.95)
        state = donchian_ensemble(df["close"], [60, 120, 360, 720]).shift(1).fillna(0).values
        r = np.log(df["close"]).diff().fillna(0).values
        te = np.arange(tr_end, len(df))
        base, gated = (state * r)[te], (np.where(np.nan_to_num(nov, nan=0.0) > thr, 0.0, state) * r)[te]
        dd_ratio.append(max_dd(gated) / max(1e-9, max_dd(base)))
        nov_up.append(np.nanmean(nov[n1:]) > np.nanmean(nov[tr_end:n1]))
        sum_ok.append(gated.sum() > base.sum() - 0.05)
    assert np.median(dd_ratio) < 0.8, f"DD ratios {np.round(dd_ratio, 2)}"
    assert sum(nov_up) >= 2 and sum(sum_ok) >= 2, f"nov_up {nov_up}, sum_ok {sum_ok}"
