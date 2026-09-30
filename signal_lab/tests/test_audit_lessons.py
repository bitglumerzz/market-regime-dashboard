"""Уроки аудита открытых проектов (docs/research/oss-trading-ml-audit-2026-09.md), закреплённые тестами."""
import numpy as np
import pandas as pd
import pytest

from signal_lab.evaluate import deflated_sharpe
from signal_lab.labels import triple_barrier_sided, uniqueness_weights
from signal_lab.run import analyze
from signal_lab.stats import min_track_record, pbo_cscv, psr
from signal_lab.tests.test_lab import synth


def test_leak_canary_is_flagged():
    """Подсаженная утечка (признак = знак будущей доходности) должна давать предупреждение в отчёте."""
    df = synth(momentum=0.0, seed=11)
    fwd = np.log(df["close"].shift(-6) / df["close"])
    df["taker_buy_volume"] = df["volume"] * (0.5 + 0.4 * np.sign(fwd.fillna(0)))    # «поток ордеров» знает будущее
    report, _ = analyze(df, "4h", 6)
    assert "Подозрение на утечку" in report


def test_entry_next_open_uses_open_price():
    idx = pd.date_range("2024-01-01", periods=80, freq="4h", tz="UTC")
    c = 100 * np.exp(np.cumsum(np.r_[0, np.tile([0.01, -0.01], 39), 0.0]))
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c}, index=idx)
    df.iloc[70, df.columns.get_loc("open")] = c[69] * 1.05          # гэп на открытии следующего бара
    tb_open = triple_barrier_sided(df, 5, np.ones(80), vol_span=20, entry="next_open")
    tb_close = triple_barrier_sided(df, 5, np.ones(80), vol_span=20, entry="close")
    assert tb_open["ret"].iloc[69] < tb_close["ret"].iloc[69]          # вход после гэпа хуже, чем «по закрытию»


def test_uniqueness_weights():
    w = uniqueness_weights(np.array([0, 0, 10]), np.array([4, 4, 14]))   # две сделки на одних барах и одна отдельная
    assert np.isclose(w[0], w[1]) and w[2] > w[0] and np.isclose(w[2] / w[0], 2.0)


def test_stats_match_purgedcv():
    """Независимая сверка математики с purgedcv (MIT): PSR, DSR, MinTRL, PBO."""
    p = pytest.importorskip("purgedcv")
    from scipy import stats
    rng = np.random.default_rng(3)
    r = rng.standard_t(5, 800) * 0.01 + 0.0015
    sr, g3, g4 = r.mean() / r.std(ddof=1), stats.skew(r), stats.kurtosis(r, fisher=False)
    assert abs(psr(r) - p.probabilistic_sharpe_ratio(r, 0.0)) < 1e-3
    assert abs(deflated_sharpe(r, 20) - p.deflated_sharpe_ratio(r, 20, 1 / (len(r) - 1))) < 5e-3
    assert abs(min_track_record(r) - p.min_track_record_length(sr, 0.0, 0.05, g3, g4)) / min_track_record(r) < 0.02
    M = rng.normal(0, 1, (480, 12))
    M[:, 4] += 0.3
    for X in (rng.normal(0, 1, (480, 12)), M):
        assert abs(pbo_cscv(X, 10)["pbo"] - p.probability_of_backtest_overfitting(X.T, n_splits=10).pbo) < 1e-9
