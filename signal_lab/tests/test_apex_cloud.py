"""Apex Cloud: механика stop-and-reverse на синтетике — отладка без просмотра реальных данных индикатора."""
import math

import numpy as np
import pandas as pd

from signal_lab.apex_cloud import infer_periods_per_year, perf, placebo, stop_and_reverse, strat_returns, trade_returns


def _df(n=50, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    up, dn = np.zeros(n), np.zeros(n)
    up[5], dn[20], up[35] = 1, 1, 1               # long с бара 5, short с 20, long с 35
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
                         "Trend UP": up, "Trend DOWN": dn}, index=idx)


def test_position_holds_until_opposite_signal():
    df = _df()
    pos = stop_and_reverse(df, "Trend UP", "Trend DOWN")
    assert (pos.iloc[:5] == 0).all()
    assert (pos.iloc[5:20] == 1).all()
    assert (pos.iloc[20:35] == -1).all()
    assert (pos.iloc[35:] == 1).all()


def test_entry_uses_next_bar_not_signal_bar():
    df = _df()
    pos = stop_and_reverse(df, "Trend UP", "Trend DOWN")
    rets = strat_returns(df, pos, fee_bps=0.0)
    # сигнал на баре 5 -> позиция решена на закрытии 5 -> реализуется в доходности бара 6 (close5->close6)
    expected = df["close"].pct_change().iloc[6]
    assert np.isclose(rets.loc[df.index[6]], expected)
    assert df.index[5] not in rets.index or np.isclose(rets.loc[df.index[5]], 0.0)


def test_no_lookahead_future_bars_do_not_change_past_returns():
    df = _df(n=60)
    pos = stop_and_reverse(df, "Trend UP", "Trend DOWN")
    r1 = strat_returns(df, pos, fee_bps=5.0)
    df2 = df.copy()
    df2.iloc[40:, df2.columns.get_indexer(["open", "high", "low", "close"])] *= 1.5
    pos2 = stop_and_reverse(df2, "Trend UP", "Trend DOWN")
    r2 = strat_returns(df2, pos2, fee_bps=5.0)
    common = r1.index.intersection(r2.index)
    common = common[common < df.index[38]]
    pd.testing.assert_series_equal(r1.loc[common], r2.loc[common])


def test_fees_reduce_return_on_flip():
    df = _df()
    pos = stop_and_reverse(df, "Trend UP", "Trend DOWN")
    free = strat_returns(df, pos, fee_bps=0.0)
    costly = strat_returns(df, pos, fee_bps=20.0)
    flip_bar = df.index[21]                        # сигнал на 20 -> позиция p меняется на баре 21 (shift(1))
    assert costly.loc[flip_bar] < free.loc[flip_bar]


def test_trade_returns_one_row_per_holding_period():
    df = _df()
    pos = stop_and_reverse(df, "Trend UP", "Trend DOWN")
    trades = trade_returns(df, pos, fee_bps=0.0)
    assert len(trades) == 3                         # long[5:20), short[20:35), long[35:]


def test_placebo_preserves_event_count():
    df = _df(n=200, seed=1)
    df["Trend UP"] = 0.0
    df["Trend DOWN"] = 0.0
    df.loc[df.index[[10, 50, 90, 130]], "Trend UP"] = 1
    df.loc[df.index[[30, 70, 110, 150]], "Trend DOWN"] = 1
    pl = placebo(df, "Trend UP", "Trend DOWN", fee_bps=5.0, periods_per_year=365 * 6, n_perm=20, seed=2)
    assert np.isfinite(pl["real"]) and len(pl["perm"]) == 20


def test_infer_periods_per_year_detects_bar_step():
    idx4h = pd.date_range("2024-01-01", periods=100, freq="4h", tz="UTC")
    idx1d = pd.date_range("2024-01-01", periods=100, freq="1D", tz="UTC")
    assert math.isclose(infer_periods_per_year(idx4h), 365 * 6, rel_tol=1e-9)
    assert math.isclose(infer_periods_per_year(idx1d), 365, rel_tol=1e-9)


def test_infer_periods_per_year_robust_to_rare_gaps():
    # DOGE-style случай: почти все шаги 4h, но пара однократных дыр не должна сбивать медиану
    idx = pd.date_range("2024-01-01", periods=200, freq="4h", tz="UTC")
    idx = idx.delete([50, 120])                    # две дырки среди 200 — медиана шага всё равно 4h
    assert math.isclose(infer_periods_per_year(idx), 365 * 6, rel_tol=1e-9)


def test_perf_annualization_scales_with_periods_per_year():
    # одни и те же по-бару доходности, разная частота баров -> Sharpe/vol отличаются ровно на sqrt(6),
    # CAGR — на корректную степень (если это не учесть, 4H-ряд, посчитанный как дневной, занижает оба показателя)
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0.0005, 0.01, 500))
    p_daily = perf(x, periods_per_year=365)
    p_4h = perf(x, periods_per_year=365 * 6)
    assert math.isclose(p_4h["sharpe"], p_daily["sharpe"] * math.sqrt(6), rel_tol=1e-9)
    assert math.isclose(p_4h["vol"], p_daily["vol"] * math.sqrt(6), rel_tol=1e-9)
    assert p_4h["mdd"] == p_daily["mdd"]            # просадка не зависит от аннуализации
    assert p_4h["cagr"] != p_daily["cagr"]
