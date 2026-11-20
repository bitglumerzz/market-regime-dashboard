"""Новые ряды (OI, DVOL, DXY, HMM) не заглядывают вперёд и учитывают задержку публикации; реестр и holdout работают."""
import numpy as np
import pandas as pd
import pytest

from signal_lab.evaluate import deflated_sharpe
from signal_lab.features import build_features_v1
from signal_lab.fetch import TF_MS, to_ms, attach_asof, dxy_available_ms, hmm_filtered_probs, parse_metrics_csv
from signal_lab.registry import (HOLDOUT_START, before_holdout, dsr_from_moments, n_trials, open_holdout, register,
                                 trade_moments)
from signal_lab.tests.test_lab import synth


def _bars(n=10, tf="4h", start="2024-01-01"):
    ts = to_ms(pd.Series(pd.date_range(start, periods=n, freq=tf, tz="UTC")))
    return pd.DataFrame({"ts": ts, "close": np.arange(n, dtype=float)})


def test_attach_asof_uses_only_values_known_at_bar_close():
    b = _bars()
    close0 = int(b["ts"][0]) + TF_MS["4h"]
    s = pd.DataFrame({"available_ms": [close0, close0 + 1], "oi": [1.0, 2.0]})
    out = attach_asof(b, "4h", s, ["oi"])
    assert out["oi"][0] == 1.0                    # известно ровно к закрытию бара 0
    assert out["oi"][1] == 2.0                    # появилось на 1 мс позже — только со следующего бара
    assert np.isnan(attach_asof(b, "4h", s.assign(available_ms=s["available_ms"] + 1), ["oi"])["oi"][0])


def test_dxy_publication_lag():
    # H.10 за неделю пн 01.01.2024 – пт 05.01.2024 выходит в пн 08.01 → считаем доступным в ср 10.01 00:00 UTC
    got = dxy_available_ms(pd.Series(pd.to_datetime(["2024-01-01", "2024-01-05"])))
    want = 1704844800000
    assert list(got) == [want, want]


def test_parse_metrics_csv():
    raw = (b"create_time,symbol,sum_open_interest,sum_open_interest_value\n"
           b"2021-01-01 00:00:00,BTCUSDT,100.5,1\n2021-01-01 00:00:00,BTCUSDT,100.5,1\n2021-01-01 00:05:00,BTCUSDT,101,1\n")
    m = parse_metrics_csv(raw)
    assert len(m) == 2 and m["oi"].tolist() == [100.5, 101.0]
    assert m["available_ms"].iloc[0] == 1609459200000


def test_hmm_probs_are_causal():
    d = synth(n=6 * 700, seed=5).resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                                      "volume": "sum"}).dropna()
    d["ts"] = to_ms(pd.Series(d.index)).values
    d = d.reset_index(drop=True)
    p1 = hmm_filtered_probs(d, min_train_days=365, refit_days=60)
    d2 = d.copy()
    d2.loc[600:, ["high", "low", "close", "volume"]] *= 1.8                # меняем «будущее»
    p2 = hmm_filtered_probs(d2, min_train_days=365, refit_days=60)
    cols = [c for c in p1 if c.startswith("hmm_p_")]
    assert p1[cols].iloc[:385].isna().all().all()                        # до конца первого окна обучения — пусто
    np.testing.assert_allclose(p1[cols].iloc[:600].fillna(-1), p2[cols].iloc[:600].fillna(-1))
    assert np.allclose(p1[cols].iloc[400:].sum(axis=1), 1.0)


def test_v1_features_with_external_series_do_not_look_ahead():
    df = synth(n=3000, seed=3)
    rng = np.random.default_rng(0)
    df = df.assign(funding=rng.normal(0, 1e-4, len(df)), oi=np.exp(rng.normal(10, .1, len(df))),
                   dvol=60 + rng.normal(0, 5, len(df)), dxy_chg_5d=rng.normal(0, 1, len(df)),
                   hmm_p_0=0.5, hmm_p_1=0.5)
    f1 = build_features_v1(df, 6)
    assert {"F6_doi_7d", "F7_funding_30d", "F8_vrp", "F10_dxy_chg_5d", "X_hmm_p_0"} <= set(f1.columns)
    df2 = df.copy()
    df2.iloc[2500:, :] *= 1.3
    f2 = build_features_v1(df2, 6)
    pd.testing.assert_frame_equal(f1.iloc[:2500], f2.iloc[:2500])


def test_registry_append_only_and_holdout_once(tmp_path):
    p = tmp_path / "trials.csv"
    assert n_trials(p) == 0
    ids = register([{"hypothesis": "H0", "rule": "a"}, {"hypothesis": "H0", "rule": "b"}], p)
    assert ids == [1, 2] and n_trials(p) == 2
    assert register([{"hypothesis": "H0", "rule": "c"}], p) == [3]
    open_holdout("финальная проверка", p)
    with pytest.raises(RuntimeError):
        open_holdout("вторая попытка", p)
    open_holdout("альты, один раз", p, group="alts-2026-10")          # другая группа — своё одноразовое открытие
    with pytest.raises(RuntimeError):
        open_holdout("альты, второй раз", p, group="alts-2026-10")
    from signal_lab.registry import holdout_opened
    assert holdout_opened(p) and holdout_opened(p, "alts-2026-10") and not holdout_opened(p, "sol-only")


def test_before_holdout_cuts_everything_after():
    df = synth(n=6 * 365 * 6, seed=1)                                   # 2021-01 … 2026-12
    cut = before_holdout(df)
    assert cut.index.max() < HOLDOUT_START and len(cut) < len(df)


def test_dsr_from_moments_matches_deflated_sharpe():
    r = np.random.default_rng(3).standard_normal(300) * 0.01 + 0.002
    m = trade_moments(r)
    for n in (1, 7, 50):
        assert dsr_from_moments(m["sr_trade"], m["n_trades"], m["skew"], m["kurt"], n) == pytest.approx(deflated_sharpe(r, n))
