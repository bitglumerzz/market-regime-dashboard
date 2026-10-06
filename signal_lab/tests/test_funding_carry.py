import numpy as np
import pandas as pd

from signal_lab import funding_carry as fc

H8 = 28_800_000


def _setup(n_bars=600, rate=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    ts = np.arange(n_bars, dtype="int64") * fc.BAR_MS
    p = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_bars))), index=ts)
    ev = np.arange(H8, ts[-1] + 1, H8, dtype="int64")
    return p, p.copy(), pd.DataFrame({"ts": ev, "rate": np.full(len(ev), rate)})


def test_each_funding_event_counted_once():
    perp, spot, f = _setup()
    df = fc.align(perp, spot, f)
    assert np.isclose(df["fund"].sum(), f["rate"].sum())       # не размножается на 4h-бары
    assert (df["fund"] > 0).sum() == len(f)


def test_static_pnl_equals_funding_minus_fees_when_legs_identical():
    perp, spot, f = _setup()
    df = fc.align(perp, spot, f)
    ret = fc.simulate(df, "static") * fc.CAPITAL_PER_NOTIONAL
    assert np.isclose(ret.sum(), f["rate"].sum() - 2 * fc.FEE_SIDE)


def test_filter_signal_is_causal():
    perp, spot, f = _setup(rate=1e-4)
    f.loc[f.index >= 200, "rate"] = -1e-4                      # ставка становится отрицательной с 200-го события
    df = fc.align(perp, spot, f)
    flip_bar = ((f["ts"].iloc[200] - 1) // fc.BAR_MS) * fc.BAR_MS
    pos = (df["sig"] > 0)
    assert pos.loc[:flip_bar].iloc[-1]                          # на баре первой отрицательной выплаты ещё в позиции
    assert not pos.iloc[-1]                                     # позже фильтр выходит


def test_basis_change_hits_pnl():
    perp, spot, f = _setup(rate=0.0)
    perp = perp * np.linspace(1.0, 1.01, len(perp))             # перп дорожает относительно спота на 1 %
    df = fc.align(perp, spot, f)
    ret = fc.simulate(df, "static") * fc.CAPITAL_PER_NOTIONAL
    assert ret.sum() < -0.009                                   # шорт перпа теряет на расширении базиса
