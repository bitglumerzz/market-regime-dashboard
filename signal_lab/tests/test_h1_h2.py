"""H1 (vol targeting) и H2 (Donchian-ансамбль): причинность и механика стопа."""
import numpy as np
import pandas as pd

from signal_lab.h1 import strategy_returns, vol_target_weights
from signal_lab.h2 import apply_band, donchian_state, model_weights


def _daily(n=900, seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(drift, 0.03, n)))
    return pd.Series(c, index=pd.date_range("2018-01-01", periods=n, freq="D", tz="UTC"))


def test_h1_h2_weights_do_not_look_ahead():
    c = _daily()
    c2 = c.copy()
    c2.iloc[700:] *= 1.5
    pd.testing.assert_series_equal(vol_target_weights(c)[:700], vol_target_weights(c2)[:700])
    pd.testing.assert_series_equal(model_weights(c)[:700], model_weights(c2)[:700])


def test_ratchet_stop_exits_below_previous_stop():
    # рост → вход → стоп подтягивается к середине канала → падение ниже стопа → выход
    up = np.linspace(100, 200, 30)
    down = np.linspace(199, 150, 10)
    c = pd.Series(np.r_[np.linspace(110, 100, 10), up, down],
                  index=pd.date_range("2020-01-01", periods=50, freq="D", tz="UTC"))
    s = donchian_state(c, 10, "mid_ratchet").fillna(0).values
    first_in = int(np.argmax(s == 1))
    assert c.iloc[first_in] >= c.iloc[first_in - 9:first_in + 1].max()  # вход — закрытие равно максимуму n дней
    assert c.iloc[first_in] > c.iloc[first_in - 1]
    assert s[-1] == 0                                                   # упали ниже подтянутого стопа — вне позиции


def test_weights_applied_next_day():
    c = _daily(n=100, seed=2)
    w = pd.Series(1.0, index=c.index)
    x = strategy_returns(c, w, fee_bps=0.0)
    np.testing.assert_allclose(x.values, c.pct_change().loc[x.index].values)


def test_band_keeps_weight_within_threshold():
    out = apply_band(np.array([0.5, 0.55, 0.62, 0.0, 0.0, 0.3]), band=0.2)
    assert out.tolist() == [0.5, 0.5, 0.62, 0.0, 0.0, 0.3]
