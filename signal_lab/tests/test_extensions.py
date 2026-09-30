"""H6 (события фандинга) и H9 (конформное воздержание): механика и отсутствие ложных «открытий» на шуме."""
import numpy as np
import pandas as pd

from signal_lab.h6 import funding_events
from signal_lab.tests.test_lab import synth
from signal_lab.v1 import evaluate_v1


def test_h6_events_do_not_overlap_and_need_a_long_negative_run():
    df = synth(n=6 * 400, seed=4)
    days = (df.index - df.index[0]).days
    df["funding"] = np.where((days > 40) & (days < 200), -1e-4, 1e-4)      # один длинный отрицательный период
    ev, unc = funding_events(df)
    assert len(ev) >= 1
    assert (ev.date.diff().dt.days.dropna() >= 30).all()
    assert (ev.run_days >= 14).all() and (ev.mean_funding_30d < 0).all()
    first_neg = df.index[0] + pd.Timedelta(days=41)
    # при ставках ±1e-4 среднее за 30 дн. уходит в минус через 15 дней, затем нужны ещё 14 дней подряд
    assert first_neg + pd.Timedelta(days=15 + 13) <= ev.date.iloc[0] <= first_neg + pd.Timedelta(days=15 + 15)


def test_conformal_filter_is_subset_and_no_gate_on_random_walk():
    res = evaluate_v1(synth(momentum=0.0, seed=11), "4h", 48, conformal=0.2)
    names = {c.name: c for c in res["configs"]}
    for k in ("logit", "lgbm"):
        ev, cf = names[f"primary + M2[{k}] EV>0"].side, names[f"primary + M2[{k}] EV>0 + conformal α=0.2"].side
        assert np.all(cf <= ev)                                            # воздержание только убирает сделки
    assert not res["gate"]
