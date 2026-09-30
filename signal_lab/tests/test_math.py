"""Проверки математики: PBO, PSR/MinTRL, бутстрап, triple-barrier, решение по матожиданию."""
import math

import numpy as np
import pandas as pd

from signal_lab.labels import ev_threshold, kelly_fraction, triple_barrier
from signal_lab.stats import min_track_record, pbo_cscv, psr, stationary_bootstrap_ci


def test_pbo_noise_vs_real_edge():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, (400, 20))
    assert 0.25 < pbo_cscv(noise)["pbo"] < 0.75                  # выбор лучшей на шуме ≈ случаен
    edge = noise.copy()
    edge[:, 3] += 0.35                                           # одна конфигурация действительно лучше
    assert pbo_cscv(edge)["pbo"] < 0.1


def test_psr_and_min_track_record():
    rng = np.random.default_rng(1)
    good = rng.normal(0.2, 1, 500)
    bad = rng.normal(-0.05, 1, 500)
    assert psr(good) > 0.99 and psr(bad) < 0.5
    assert math.isinf(min_track_record(bad))
    trl = min_track_record(good)
    assert 30 < trl < 200                                        # ≈ 1 + (z/SR)² ≈ 1 + (1.645/0.2)² ≈ 69


def test_stationary_bootstrap_ci_covers_mean():
    rng = np.random.default_rng(2)
    x = rng.normal(0.5, 1, 300)
    lo, hi = stationary_bootstrap_ci(x, n_boot=500)
    assert lo < 0.5 < hi and hi - lo < 0.5


def _path(closes, highs=None, lows=None):
    c = np.array(closes, float)
    h = np.array(highs if highs is not None else c, float)
    l = np.array(lows if lows is not None else c, float)
    return pd.DataFrame({"close": c, "high": h, "low": l})


def test_triple_barrier_first_touch():
    # 60 спокойных баров для оценки σ, затем явный рост → верхний барьер
    base = list(100 * np.exp(np.cumsum(np.r_[0, np.tile([0.01, -0.01], 30)])))
    up = base + [base[-1] * 1.05] * 6
    tb = triple_barrier(_path(up), horizon=5, k=1.0, vol_span=20)
    t = len(base) - 1
    assert tb["label"].iloc[t] == 1 and np.isclose(tb["ret"].iloc[t], tb["width"].iloc[t])
    dn = base + [base[-1] * 0.95] * 6
    tb = triple_barrier(_path(dn), horizon=5, k=1.0, vol_span=20)
    assert tb["label"].iloc[t] == -1 and np.isclose(tb["ret"].iloc[t], -tb["width"].iloc[t])
    # в одной свече задеты оба барьера → консервативно считаем стоп
    both_h = base + [base[-1] * 1.05] + [base[-1]] * 5
    both_l = base + [base[-1] * 0.95] + [base[-1]] * 5
    tb = triple_barrier(_path(base + [base[-1]] * 6, both_h, both_l), horizon=5, k=1.0, vol_span=20)
    assert tb["label"].iloc[t] == -1
    # тихо до конца → выход по времени
    flat = base + [base[-1]] * 6
    tb = triple_barrier(_path(flat), horizon=5, k=1.0, vol_span=20)
    assert tb["label"].iloc[t] == 0 and np.isclose(tb["ret"].iloc[t], 0)


def test_ev_threshold_and_kelly():
    w = np.array([0.02, 0.02, 0.02, 0.002])
    p = np.array([0.60, 0.40, 0.52, 0.60])
    # cost 0.2%: edge = (2p−1)·w → 0.4%, −0.4%, 0.08% (<cost), 0.04% (<cost)
    assert list(ev_threshold(p, w, 0.002)) == [1, -1, 0, 0]
    f = kelly_fraction(np.array([0.5, 0.55, 0.9]), np.array([0.05, 0.05, 0.05]), frac=0.25, cap=1.0)
    assert f[0] == 0 and np.isclose(f[1], 0.25 * 0.1 / 0.05) and f[2] == 1.0    # 0.5 и упор в потолок
