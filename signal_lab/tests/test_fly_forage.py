"""H-FlyForage: механика (каузальность, зеркальность) на синтетике + positive controls PC1-PC3 (§4 пре-регистрации).
Параметры TAU_BARS/FLOOR_BUSY/K_SENS зафиксированы калибровкой на шуме в самом модуле — здесь не подбираются."""
import numpy as np
import pandas as pd
import pytest

from signal_lab import fly_forage as ff
from signal_lab.h2 import donchian_state


def _gbm(n, seed, drift=0.0, sigma=0.01):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, sigma, n)
    c = 100 * np.exp(np.cumsum(r))
    idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"open": np.r_[c[0], c[:-1]], "high": c, "low": c, "close": c}, index=idx), r


def test_short_state_mirrors_long_state_on_inverted_price():
    df, _ = _gbm(1500, seed=1)
    long_on_inverted = donchian_state(1 / df["close"], 20, "mid_ratchet", "v1")
    short_on_original = ff.short_state(df["close"], 20)
    # 1/price инвертирует направление движения (не точная арифметика high/low, но каузальный тест на знак событий)
    valid = long_on_inverted.notna() & short_on_original.notna()
    agree = (long_on_inverted[valid] == short_on_original[valid]).mean()
    assert agree > 0.85


def test_encounter_counts_are_causal():
    df, _ = _gbm(2000, seed=2)
    cl1, cs1, v1 = ff.encounter_counts(df["close"], 6)
    df2 = df.copy()
    df2.iloc[1500:, df2.columns.get_indexer(["open", "high", "low", "close"])] *= 1.4
    cl2, cs2, v2 = ff.encounter_counts(df2["close"], 6)
    np.testing.assert_array_equal(cl1[:1350], cl2[:1350])     # запас на самое длинное окно (360д ≈ 2160 баров)
    np.testing.assert_array_equal(cs1[:1350], cs2[:1350])


def test_memory_traces_decay_and_are_causal():
    cl = np.zeros(300); cl[50] = 3.0
    cs = np.zeros(300)
    el, es = ff.memory_traces(cl, cs, tau=50.0)
    assert el[49] == 0.0 and el[50] == 3.0
    assert el[50] > el[100] > el[200] > 0            # экспоненциальное затухание, не доходит до 0 мгновенно
    assert np.all(es == 0.0)


def test_decide_search_state_when_no_recent_encounters():
    n = 200
    cl = np.zeros(n); cs = np.zeros(n)
    el, es = ff.memory_traces(cl, cs)
    side = ff.decide(el, es, cl, cs, "stoch", seed=0)
    assert np.all(side == 0.0)                       # нет встреч вообще → всегда «в поиске»


def test_decide_holds_position_between_encounters_stoch_and_det():
    n = 100
    cl = np.zeros(n); cs = np.zeros(n)
    cl[10] = 9.0                                      # одна мощная встреча long, достаточно чтобы busy >= floor
    el, es = ff.memory_traces(cl, cs)
    for config in ("stoch", "det"):
        side = ff.decide(el, es, cl, cs, config, seed=0)
        assert side[9] == 0.0                         # до встречи — в поиске
        nonzero = side[10:]
        first = nonzero[np.argmax(nonzero != 0)]
        hold_len = np.argmax(nonzero != first) if np.any(nonzero != first) else len(nonzero)
        assert hold_len > 5                           # позиция держится, не дёргается каждый бар


def test_bias_sign_only_changes_at_encounters():
    """Находка, не баг (объяснение, почему в decide() нет отдельного config 'cont'): между встречами e_long и
    e_short затухают с ОДНИМ decay, их разность (bias) тоже затухает к нулю, но не меняет знак — знак bias(t)
    может перевернуться только в момент новой встречи (приход count>0 на другую сторону). Поэтому «решать каждый
    бар» и «решать только в момент встречи» при ДЕТЕРМИНИРОВАННОМ правиле sign(bias) дали бы один и тот же
    результат — различие момента решения осмысленно только вместе со стохастикой. Зафиксировано в H-FlyForage.md."""
    cl = np.zeros(80); cs = np.zeros(80)
    cl[10] = 9.0; cs[40] = 9.0                          # сперва long-встреча, затем мощная short-встреча
    el, es = ff.memory_traces(cl, cs, tau=120.0)
    bias = el - es
    sign_changes = np.where(np.diff(np.sign(bias)) != 0)[0] + 1
    is_encounter = (cl + cs) > 0
    assert len(sign_changes) > 0
    assert np.all(is_encounter[sign_changes])           # каждая смена знака bias — ровно на баре встречи
    side_d = ff.decide(el, es, cl, cs, "det", seed=0)
    assert side_d[39] == 1.0 and side_d[40] == -1.0     # det переключается ровно в момент новой встречи, не раньше


def test_trade_returns_next_bar_entry_convention():
    df, _ = _gbm(60, seed=3)
    side = np.zeros(60); side[10] = 1.0
    r = ff.strat_returns(df["close"], side, fee_bps=0.0)
    expected = df["close"].pct_change().iloc[11]
    assert np.isclose(r.loc[df.index[11]], expected)


# --------------------------------------------------------------------------- positive controls (§4 H-FlyForage.md)
def regime_switch_synthetic(n, seed, switch_min=500, switch_max=1500, drift=0.0012):
    rng = np.random.default_rng(seed)
    regime = np.zeros(n)
    t, s = 0, 1
    while t < n:
        length = rng.integers(switch_min, switch_max)
        regime[t:t + length] = s
        t += length
        s = -s
    r = regime * drift + rng.normal(0, 0.01, n)
    c = 100 * np.exp(np.cumsum(r))
    idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
    df = pd.DataFrame({"open": np.r_[c[0], c[:-1]], "high": c * 1.002, "low": c * 0.998, "close": c}, index=idx)
    return df, regime


def test_PC1_signal_exists_stoch_median_above_threshold():
    agree = []
    for seed in range(20):
        df, regime = regime_switch_synthetic(4000, seed)
        cl, cs, valid = ff.encounter_counts(df["close"], 6)
        el, es = ff.memory_traces(cl, cs)
        side = ff.decide(el, es, cl, cs, "stoch", seed=seed)
        m = valid & (side != 0)
        if m.sum() < 50:
            continue
        agree.append(float(np.mean(np.sign(side[m]) == np.sign(regime[m]))))
    assert len(agree) >= 15, "мало валидных seed'ов — контроль неинформативен"
    assert np.median(agree) > 0.55, f"median agreement {np.median(agree):.3f}"


def test_PC2_no_signal_no_false_profit_on_pure_noise():
    n_ok = 0
    for seed in range(20):
        df, _ = _gbm(4000, seed=1000 + seed)
        res = ff.run(df, "4h", "stoch", fee_bps=5.0, seed=seed, cutoff=None)
        if len(res["trades"]) < 10:
            n_ok += 1          # недостаточно сделок на чистом шуме — тоже не ложная прибыль
            continue
        lo, hi = res["stats"]["ci_lo_bps"], res["stats"]["ci_hi_bps"]
        n_ok += int(lo <= 0 <= hi)
    assert n_ok >= 18, f"only {n_ok}/20 seeds show no false profit"


def test_PC3_stochastic_vs_deterministic_on_regime_signal():
    """§2.4/§4 PC3: единственная содержательная ось сравнения — стохастика против детерминизма (момент решения
    не варьируется отдельно, см. test_decide_cont_equals_det_...). Пред-заявлено: F-stoch не обязана побеждать —
    фиксируем оба результата как есть."""
    results = {c: [] for c in ("stoch", "det")}
    for seed in range(10):
        df, regime = regime_switch_synthetic(4000, seed)
        cl, cs, valid = ff.encounter_counts(df["close"], 6)
        el, es = ff.memory_traces(cl, cs)
        for config in results:
            side = ff.decide(el, es, cl, cs, config, seed=seed)
            m = valid & (side != 0)
            if m.sum() >= 50:
                results[config].append(float(np.mean(np.sign(side[m]) == np.sign(regime[m]))))
    medians = {c: np.median(v) for c, v in results.items() if v}
    assert len(medians) == 2
    print(f"\n[PC3] медианное совпадение с режимом: {medians}")
    assert all(0.4 < v < 0.95 for v in medians.values())
