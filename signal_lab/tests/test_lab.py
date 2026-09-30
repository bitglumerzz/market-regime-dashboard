"""Проверка лаборатории на синтетике, где ответ известен заранее."""
import numpy as np
import pandas as pd

from signal_lab.features import build_features, forward_return
from signal_lab.run import analyze


def synth(n=6 * 365 * 4, momentum=0.0, seed=0):
    """4 года 4h-баров. momentum > 0 — в доходности встроена зависимость от прошлой недели (предсказуемо)."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n) * 0.01
    r = np.zeros(n)
    for t in range(n):
        past = r[max(0, t - 42):t].sum()
        r[t] = eps[t] + momentum * np.tanh(past / 0.05) * 0.01
    c = 30000 * np.exp(np.cumsum(r))
    idx = pd.date_range("2021-01-01", periods=n, freq="4h", tz="UTC")
    hi, lo = c * (1 + np.abs(rng.normal(0, .003, n))), c * (1 - np.abs(rng.normal(0, .003, n)))
    v = rng.lognormal(10, .3, n)
    return pd.DataFrame({"open": np.r_[c[0], c[:-1]], "high": hi, "low": lo, "close": c, "volume": v,
                         "taker_buy_volume": v * rng.uniform(.4, .6, n)}, index=idx)


def test_random_walk_finds_nothing():
    """Ложные открытия под контролем: на случайном блуждании гейт не проходит, значимых признаков почти нет."""
    report, res = analyze(synth(momentum=0.0, seed=1), "4h", 6)
    assert not res["gate"]
    assert sum(r.q_value < 0.05 for r in res["factors"]) <= 1
    for m in res["models"]:
        assert m.hit < 0.56


def test_planted_momentum_is_detected():
    report, res = analyze(synth(momentum=0.35, seed=2), "4h", 6)
    top = {r.name for r in res["factors"][:4]}
    assert top & {"mom_7d", "mom_3d", "tsmom_28d_vs", "ma_cross_7_28", "mom_14d", "ma_gap_20d", "donchian_pos_20d"}
    assert any(r.q_value < 0.05 for r in res["factors"])
    best = max(res["models"], key=lambda m: m.hit)
    assert best.hit > 0.53 and best.mean_ret_bps > 0


def test_features_do_not_look_ahead():
    df = synth(n=2000, seed=3)
    f1 = build_features(df, 6)
    df2 = df.copy()
    df2.iloc[1500:, :] *= 1.7                                   # меняем «будущее»
    f2 = build_features(df2, 6)
    pd.testing.assert_frame_equal(f1.iloc[:1500], f2.iloc[:1500])
    fwd = forward_return(df["close"], 6)
    assert np.isnan(fwd.iloc[-1]) and not np.isnan(fwd.iloc[0])
