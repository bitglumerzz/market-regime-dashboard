import numpy as np
import pandas as pd

from signal_lab.r2a_tsmom import portfolio_returns, positions, synthetic_panel


def test_positions_do_not_look_ahead():
    o, c = synthetic_panel(n_coins=3, days=400, sharpe=1.0, seed=1)
    p = positions(c, 14, 7)
    c2 = c.copy()
    c2.iloc[300:] *= 3.0
    pd.testing.assert_frame_equal(p.iloc[:300], positions(c2, 14, 7).iloc[:300])


def test_return_uses_next_open():
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    o = pd.DataFrame({"A": [100, 100, 110, 121, 121.0]}, index=idx)
    p = pd.DataFrame({"A": [np.nan, 1.0, 1.0, 1.0, 1.0]}, index=idx)
    x = portfolio_returns(o, p, fee_bps=0.0)
    # позиция с закрытия 2-го дня держится с open 3-го (110) до open 4-го (121)
    assert abs(x.iloc[0] - 0.10) < 1e-12


def test_random_walk_gives_no_edge():
    o, c = synthetic_panel(n_coins=10, days=3200, sharpe=0.0, seed=3)
    x = portfolio_returns(o, positions(c, 28, 28))
    assert abs(x.mean() / x.std() * np.sqrt(365)) < 1.0
