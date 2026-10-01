"""H-FlyBrain, positive control (§7 пре-регистрации): на синтетике с ПОСАЖЕННЫМ предсказателем цепь обязана
выучить его. Если не выучивает — конвейер сломан, и нулевой результат на рынке ничего не значит.
Пропускается, пока валентность MBON не заморожена (MBON_VALENCE пуст) или нет данных коннектома."""
import numpy as np
import pandas as pd
import pytest

from signal_lab import flybrain as fb

HAS_DATA = (fb.FLYWIRE / "mb_right_nodes.parquet").exists()
pytestmark = [pytest.mark.skipif(not HAS_DATA, reason="нужны data/flywire/*"), pytest.mark.slow]


def planted(n=6 * 365 * 2, seed=0, strength=0.35):
    """4h-свечи, где скрытый медленный фактор d_t сдвигает доходность следующих баров: sign(d) предсказывает
    направление ~70 % времени. d подмешивается в df как колонка 'planted', которую build_features_v1 не трогает —
    поэтому в тесте мы подменяем ею один из признаков (см. _inject)."""
    rng = np.random.default_rng(seed)
    d = np.zeros(n)
    for t in range(1, n):
        d[t] = 0.98 * d[t - 1] + rng.normal(0, 0.2)
    eps = rng.standard_normal(n) * 0.01
    r = eps + strength * 0.01 * np.tanh(d)
    c = 30000 * np.exp(np.cumsum(r))
    idx = pd.date_range("2021-01-01", periods=n, freq="4h", tz="UTC")
    hi, lo = c * (1 + np.abs(rng.normal(0, .003, n))), c * (1 - np.abs(rng.normal(0, .003, n)))
    v = rng.lognormal(10, .3, n)
    df = pd.DataFrame({"open": np.r_[c[0], c[:-1]], "high": hi, "low": lo, "close": c, "volume": v,
                       "taker_buy_volume": v * rng.uniform(.4, .6, n)}, index=idx)
    return df, d


def _inject(monkeypatch, d):
    """Подменяем F12_weekday (неинформативный) на посаженный фактор — остальной конвейер без изменений."""
    orig = fb.build_features_v1

    def patched(df, bpd):
        X = orig(df, bpd)
        X["F12_weekday"] = d[: len(X)]
        return X
    monkeypatch.setattr(fb, "build_features_v1", patched)


def test_fly_learns_planted_signal(monkeypatch):
    monkeypatch.setattr(fb, "T_PRESENT_MS", 50)             # только ради скорости теста; прогон на рынке — 100 мс
    df, d = planted()
    _inject(monkeypatch, d)
    res = fb.run_fly(df, "4h", seed=0, cutoff=None, test_days=60, min_train_days=240)
    d1 = next(c for c in res["configs"] if c.name.startswith("fly D1"))
    prim = res["baselines"][0].stats
    assert d1.stats.get("trades", 0) >= 50, "мало сделок — контроль не информативен"
    assert d1.stats["hit"] > 0.60, f"цепь не выучила посаженный сигнал: hit {d1.stats['hit']:.2f}"
    assert d1.stats["mean_bps"] > prim["mean_bps"], "не бьёт «всегда long» на данных с посаженным сигналом"


def test_fly_finds_nothing_on_pure_noise(monkeypatch):
    monkeypatch.setattr(fb, "T_PRESENT_MS", 50)
    df, d = planted(strength=0.0)
    _inject(monkeypatch, np.zeros_like(d))
    res = fb.run_fly(df, "4h", seed=0, cutoff=None, test_days=60, min_train_days=240)
    for c in res["configs"]:
        if c.stats.get("trades", 0) >= 50:
            assert c.stats["hit"] < 0.58, f"ложное открытие на шуме: {c.name} hit {c.stats['hit']:.2f}"
    assert not res["gate"]
