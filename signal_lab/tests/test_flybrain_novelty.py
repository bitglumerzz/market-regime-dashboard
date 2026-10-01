"""H-FlyBrain-2: свойства FlyHash (LSH) и фильтра Блума мухи — на синтетике и на реальной проводке, без рынка."""
import numpy as np
import pytest

from signal_lab import flybrain as fb
from signal_lab.flybrain_novelty import FlyBloom, flyhash, novelty_series, tags_matrix, uniform_circuit

HAS_DATA = (fb.FLYWIRE / "mb_right_nodes.parquet").exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="нет data/flywire/mb_right_*.parquet")


def _jaccard(a, b):
    return (a & b).sum() / max(1, (a | b).sum())


@needs_data
def test_flyhash_is_sparse_and_locality_sensitive():
    c = fb.load_circuit()
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, 25)
    tag = flyhash(c, z)
    assert abs(tag.mean() - 0.05) < 0.002                               # 5 % активных KC
    near = [_jaccard(tag, flyhash(c, z + rng.normal(0, 0.1, 25))) for _ in range(20)]
    far = [_jaccard(tag, flyhash(c, rng.normal(0, 1, 25))) for _ in range(20)]
    assert np.mean(near) > 0.5 > np.mean(far)                           # похожие входы → похожие теги
    assert np.mean(far) < 0.15


@needs_data
def test_flyhash_deterministic_and_causal():
    c = fb.load_circuit()
    z = np.linspace(-2, 2, 25)
    np.testing.assert_array_equal(flyhash(c, z), flyhash(c, z))


def test_bloom_filter_novelty_semantics():
    m = 100
    fbf = FlyBloom(m, eps=1.0)
    a = np.zeros(m, bool); a[:5] = True
    b = np.zeros(m, bool); b[5:10] = True
    ab = np.zeros(m, bool); ab[3:8] = True                               # пересекается с a на 2 из 5
    assert fbf.novelty(a) == 1.0                                         # ничего не видел
    fbf.insert(a)
    assert fbf.novelty(a) == 0.0 and fbf.novelty(b) == 1.0               # Блум: виденное — 0, чужое — 1
    assert abs(fbf.novelty(ab) - 0.6) < 1e-9                             # чувствительность к сходству: 3 из 5 новых


def test_bloom_filter_time_decay_restores_novelty():
    m = 50
    fbf = FlyBloom(m, eps=1.0, tau_bars=10.0)
    a = np.zeros(m, bool); a[:5] = True
    fbf.insert(a)
    n0 = fbf.novelty(a)
    for _ in range(30):
        fbf.tick()
    assert n0 == 0.0 and 0.9 < fbf.novelty(a) < 1.0                    # давно не виденное снова становится новым


def test_bloom_filter_soft_eps():
    m = 50
    fbf = FlyBloom(m, eps=0.5)
    a = np.zeros(m, bool); a[:5] = True
    fbf.insert(a); fbf.insert(a)
    assert abs(fbf.novelty(a) - 0.25) < 1e-9                             # (1−ε)^c


@needs_data
def test_binary_filter_saturates_on_a_stream():
    """Ёмкость фильтра Блума ≈ m/k ≈ 2597/130 ≈ 20 состояний: без затухания на потоке новизна уходит в 0 —
    поэтому для рыночного потока τ обязателен (PNAS 2018: чувствительность ко времени)."""
    c = fb.load_circuit()
    rng = np.random.default_rng(2)
    Z = rng.normal(0, 1, (300, 25))
    nov = novelty_series(c, Z, warmup=0)
    assert nov[0] == 1.0 and np.nanmean(nov[200:]) < 0.05


@needs_data
def test_real_wiring_has_hub_kcs_below_uniform_capacity():
    """Для равномерного хеша доля «виденных» бит после n вставок ≈ 1 − exp(−k·n/m). У РЕАЛЬНОЙ проводки есть
    хабовые KC (сильный uPN-вход), выигрывающие WTA слишком часто → покрытие ниже формулы, эффективная ёмкость меньше.
    Документируем как свойство коннектома (и как предсказание для плацебо с равномерной проекцией)."""
    c = fb.load_circuit()
    rng = np.random.default_rng(3)
    m, k = len(c.kc_ids), int(round(0.05 * len(c.kc_ids)))
    fbf = FlyBloom(m, eps=1.0)
    counts = np.zeros(m)
    for t in range(200):
        tag = flyhash(c, rng.normal(0, 1, 25))
        counts += tag
        if t < 20:
            fbf.insert(tag)
    covered20 = 1 - fbf.w.mean()
    uniform20 = 1 - np.exp(-k * 20 / m)
    ever_used = (counts > 0).mean()
    assert covered20 < uniform20                                          # ниже равномерной формулы
    assert 0.2 < covered20 < 0.75 and 0.3 < ever_used < 1.0
    print(f"\n[hub metric] covered after 20 inserts {covered20:.2f} (uniform {uniform20:.2f}); "
          f"KCs ever active in 200 random tags: {ever_used:.2f}; top-5% KCs take {np.sort(counts)[::-1][:k].sum() / counts.sum():.2f} of activations")


@needs_data
def test_novelty_series_is_causal_and_flags_regime_shift():
    """Реалистичная постановка: недавний режим — кластер состояний (σ = 0.3 вокруг центра), затем скачок центра."""
    c = fb.load_circuit()
    rng = np.random.default_rng(1)
    center = rng.normal(0, 1, 25)
    Z = np.r_[center + rng.normal(0, 0.3, (300, 25)), (center + 3.0) + rng.normal(0, 0.3, (100, 25))]
    nov = novelty_series(c, Z, tau_bars=24.0, warmup=50)                         # память — ~4 дня 4h-баров
    assert np.isnan(nov[:50]).all()
    assert nov[300] > np.nanmax(nov[250:300])                                    # первый бар нового режима — пик
    assert nov[300] > 0.5 > np.nanmean(nov[250:300])
    assert np.nanmean(nov[320:340]) < nov[300]                                   # новый режим становится знакомым
    Z2 = Z.copy(); Z2[350:] += 10.0                                              # меняем будущее
    nov2 = novelty_series(c, Z2, tau_bars=24.0, warmup=50)
    np.testing.assert_array_equal(nov[:350], nov2[:350])                        # прошлое не меняется


@needs_data
def test_uniform_placebo_circuit_hashes_with_fewer_hubs():
    """Равномерная проекция (6 uPN на KC, равные веса) — плацебо для вопроса «важна ли схема мухи».
    Ожидание из замера хабов: покрытие ближе к формуле, задействовано больше KC."""
    c = fb.load_circuit()
    u = uniform_circuit(c, np.random.default_rng(0), s=6)
    assert u.W_pn_kc.shape == c.W_pn_kc.shape and np.all(np.asarray((u.W_pn_kc > 0).sum(1)).ravel() == 6)
    rng = np.random.default_rng(3)
    cu = np.zeros(len(c.kc_ids)); cr = np.zeros(len(c.kc_ids))
    for _ in range(200):
        z = rng.normal(0, 1, 25)
        cu += flyhash(u, z); cr += flyhash(c, z)
    assert abs(flyhash(u, rng.normal(0, 1, 25)).mean() - 0.05) < 0.002
    assert (cu > 0).mean() > (cr > 0).mean()                              # меньше хабов → больше KC в деле


def _planted_readout_data(seed=4, n=2000):
    rng = np.random.default_rng(seed)
    d = np.zeros(n)
    for t in range(1, n):
        d[t] = 0.98 * d[t - 1] + rng.normal(0, 0.2)
    Z = rng.normal(0, 1, (n, 25)); Z[:, 7] = d                            # один информативный признак из 25
    y = (0.35 * np.tanh(d) + rng.normal(0, 0.6, n) > 0).astype(int)         # знак предсказуем ~65–70 %
    return Z, y, np.arange(0, 1400), np.arange(1400, n)


@needs_data
@pytest.mark.xfail(strict=True, reason="H-FlyBrain-2, зафиксированный отрицательный результат 01.10.2026: логит на FlyHash "
                                        "всех 25 признаков (0.60) хуже логита на сырых признаках — LSH хранит ОБЩЕЕ сходство "
                                        "векторов, где 24 измерения шум; вариант «хеш как предиктор направления» отброшен")
def test_linear_readout_on_kc_tags_matches_raw_logit():
    from sklearn.linear_model import LogisticRegression
    c = fb.load_circuit()
    Z, y, tr, te = _planted_readout_data()
    T = tags_matrix(c, Z)
    acc_tags = (LogisticRegression(C=0.1, max_iter=3000).fit(T[tr], y[tr]).predict(T[te]) == y[te]).mean()
    acc_raw = (LogisticRegression(C=0.1, max_iter=3000).fit(Z[tr], y[tr]).predict(Z[te]) == y[te]).mean()
    assert acc_tags >= acc_raw, f"tags {acc_tags:.3f} vs raw {acc_raw:.3f}"     # измерено: 0.600 против 0.618


@needs_data
def test_linear_readout_on_kc_tags_is_above_base_rate():
    """Считывание на коде KC всё же обучаемо (информация не уничтожена полностью) и каузально: тег бара t
    не зависит от будущих строк."""
    from sklearn.linear_model import LogisticRegression
    c = fb.load_circuit()
    Z, y, tr, te = _planted_readout_data()
    T = tags_matrix(c, Z)
    acc = (LogisticRegression(C=0.1, max_iter=3000).fit(T[tr], y[tr]).predict(T[te]) == y[te]).mean()
    base = max(y[te].mean(), 1 - y[te].mean())
    assert acc > base + 0.02, f"readout acc {acc:.2f}, base {base:.2f}"
    Z2 = Z.copy(); Z2[1500:] += 5.0
    assert (tags_matrix(c, Z2)[:1500] != T[:1500]).nnz == 0
