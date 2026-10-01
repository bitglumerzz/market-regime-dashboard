"""H-FlyBrain: цепь, кодирование, перепроводка, каузальность — на синтетике и на самом коннектоме (без рынка).
Positive control (посаженный сигнал) добавляется после заморозки валентности MBON — см. test_flybrain_control.py."""
import numpy as np
import pandas as pd
import pytest

from signal_lab import flybrain as fb

HAS_DATA = (fb.FLYWIRE / "mb_right_nodes.parquet").exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="нет data/flywire/mb_right_*.parquet")


def test_encode_rates_bounds_and_on_off():
    z = np.array([3.0, -3.0, 0.0])
    r = fb.encode_rates(z, 139)
    assert r.min() >= fb.R_BASE_HZ and r.max() <= fb.R_BASE_HZ + fb.R_GAIN_HZ
    assert r[0] > r[3]                                   # +z: ON высоко, OFF низко
    assert r[5] < r[8]                                   # −z: ON низко, OFF высоко
    assert np.allclose(r[10:15], fb.R_BASE_HZ + fb.R_GAIN_HZ * 0.5)   # z=0 → середина
    assert np.all(r[15:] == fb.R_BASE_HZ)                # лишние PN — базовая частота


def test_double_edge_swap_preserves_degrees():
    rng = np.random.default_rng(0)
    pairs = rng.choice(50 * 100, 400, replace=False)                   # 400 УНИКАЛЬНЫХ пар (без дубликатов)
    pre, post = pairs // 100, 1000 + pairs % 100
    pre2, post2 = fb._double_edge_swap(pre, post, rng)
    assert np.array_equal(post, post2)                                 # пост-концы (и их синапсы) на месте
    assert np.array_equal(np.sort(pre), np.sort(pre2))                 # out-степени (мультимножество) те же
    assert (pre != pre2).mean() > 0.3                                  # но партнёры реально перемешаны
    assert len(set(zip(pre2.tolist(), post2.tolist()))) == len(set(zip(pre.tolist(), post.tolist())))   # дубликатов не прибавилось


@needs_data
def test_circuit_counts_match_extraction():
    c = fb.load_circuit()
    assert len(c.pn_ids) == 139 and len(c.kc_ids) == 2597 and len(c.mbon_ids) == 48
    assert c.W_pn_kc.shape == (2597, 139) and c.W_kc_mbon0.shape == (48, 2597)
    # дубликаты пар (pre, post) свёрнуты при извлечении (сумма синапсов): 12 729 рёбер uPN→KC, 23 574 KC→MBON
    assert c.W_pn_kc.nnz == 12729 and int((c.W_kc_mbon0 > 0).sum()) == 23574
    assert abs(c.W_pn_kc.sum() - 178221) < 1e-6 and abs(c.W_kc_mbon0.sum() - 111810) < 1e-3   # синапсы — ровно как в данных
    assert (c.apl_kc > 0).sum() == 2597
    assert c.comp_pam.sum() > 0 and c.comp_ppl1.sum() > 0


@needs_data
def test_rewired_circuit_same_size_different_wiring():
    rng = np.random.default_rng(1)
    a, b = fb.load_circuit(), fb.load_circuit(rng=rng, rewire=True)
    assert a.W_pn_kc.nnz == b.W_pn_kc.nnz and int((b.W_kc_mbon0 > 0).sum()) == int((a.W_kc_mbon0 > 0).sum())
    assert abs(a.W_pn_kc.sum() - b.W_pn_kc.sum()) < 1e-6                 # сумма синапсов сохранена
    assert abs(a.W_kc_mbon0.sum() - b.W_kc_mbon0.sum()) < 1e-3
    # взвешенный вход на КАЖДУЮ KC и КАЖДЫЙ MBON сохраняется ровно (возбудимость та же, партнёры другие)
    np.testing.assert_allclose(np.asarray(a.W_pn_kc.sum(1)).ravel(), np.asarray(b.W_pn_kc.sum(1)).ravel(), atol=1e-6)
    np.testing.assert_allclose(a.W_kc_mbon0.sum(1), b.W_kc_mbon0.sum(1), atol=1e-3)
    # число исходящих рёбер каждого uPN сохранено
    np.testing.assert_array_equal(np.sort(np.asarray((a.W_pn_kc > 0).sum(0)).ravel()), np.sort(np.asarray((b.W_pn_kc > 0).sum(0)).ravel()))
    assert (a.W_pn_kc != b.W_pn_kc).nnz > 0.3 * a.W_pn_kc.nnz
    np.testing.assert_array_equal(a.comp_pam, b.comp_pam)              # компартменты DAN — как в данных


@needs_data
def test_lif_bar_runs_and_is_reproducible():
    c = fb.load_circuit()
    fly = fb.Fly(c, w_unit=0.2, apl_gain=1.0, seed=3)
    kc1, mb1 = fly.present(np.zeros(25))
    fly.n_bar -= 1
    kc2, mb2 = fly.present(np.zeros(25))
    np.testing.assert_array_equal(kc1, kc2); np.testing.assert_array_equal(mb1, mb2)
    assert kc1.shape == (2597,) and mb1.shape == (48,) and kc1.sum() >= 0


def test_reinforce_depresses_only_active_compartment():
    n_kc, n_mb = 10, 4
    c = fb.Circuit(np.arange(3), np.arange(n_kc), np.arange(n_mb), np.array(["a"] * n_mb), None,
                   np.ones((n_mb, n_kc), np.float32), np.ones(n_kc), np.array([1, 1, 0, 0], bool), np.array([0, 0, 1, 1], bool))
    fly = fb.Fly(c, 1.0, 0.0, 0)
    elig = np.zeros(n_kc); elig[:5] = 3.0
    fly.reinforce(elig, +0.01)                       # награда → PAM-компартменты (MBON 0,1) депрессируют активные KC
    assert (fly.W[:2, :5] < 1).all() and (fly.W[:2, 5:] == 1).all() and (fly.W[2:] == 1).all()
    fly.reinforce(elig, -0.05)                       # наказание → PPL1-компартменты (MBON 2,3)
    assert (fly.W[2:, :5] < 1).all() and (fly.W[2:, 5:] == 1).all()
    assert (fly.W >= 0).all()
