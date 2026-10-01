"""H-FlyBrain — грибовидное тело дрозофилы (коннектом FlyWire v783, правое полушарие) как обучаемый сигнал.

Спецификация зафиксирована до прогона: docs/research/experiments/H-FlyBrain.md. Ничего здесь не менять после
просмотра результатов на рынке — любое изменение это новое испытание в реестре.

    python -m signal_lab.flybrain --data data/BTCUSDT_4h.parquet --seed 0
    python -m signal_lab.flybrain --data data/BTCUSDT_4h.parquet --seed 0 --placebo features    # плацебо A
    python -m signal_lab.flybrain --data data/BTCUSDT_4h.parquet --seed 0 --placebo topology    # плацебо B

Цепь: uPN (139, Пуассон по признакам) → KC (2597, LIF, торможение APL) → MBON (48, LIF). Решение — разница частот
approach- и avoidance-MBON. Обучение — депрессия KC→MBON-синапсов в компартментах дофаминового кластера (PAM —
награда, PPL1 — наказание), исход сделки применяется на баре, когда он становится известен. Оценка — тот же стек,
что у v1: triple-barrier 2σ/1.5σ/48 ч, Folds, score_configs, N из реестра.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy import sparse

from .barrier_eval import Config, score_configs
from .evaluate import Folds, nonoverlap
from .features import build_features_v1, primary_side
from .labels import triple_barrier_sided
from .registry import HOLDOUT_START, before_holdout, n_trials, register, trade_moments
from .run import BARS_PER_DAY, load

FLYWIRE = Path("data") / "flywire"

# --- параметры, зафиксированные в пре-регистрации (§3, §4, §5, §6). [ЛИТ]-блок LIF уточняется до заморозки.
DT_MS = 1.0
T_PRESENT_MS = 100
R_BASE_HZ, R_GAIN_HZ = 5.0, 95.0
PN_PER_FEATURE, PN_ON_PER_FEATURE = 5, 3
DEAD_ZONE = 0.1
ETA, TAU_REC_BARS, EMA_TRADES = 0.05, 2000.0, 20
KC_TARGET_SPARSITY = (0.05, 0.10)
LIF = dict(tau_m=20.0, v_rest=-52.0, v_th=-45.0, v_reset=-52.0, refrac=2.2, tau_syn=5.0)   # [ЛИТ] Shiu 2024, проверить

# Валентность MBON (approach / avoid) — [ЛИТ] заполняется по Aso 2014; до заполнения модуль отказывается запускаться.
MBON_VALENCE: dict[str, str] = {}


# ============================================================ коннектом
@dataclass
class Circuit:
    pn_ids: np.ndarray
    kc_ids: np.ndarray
    mbon_ids: np.ndarray
    mbon_types: np.ndarray
    W_pn_kc: sparse.csr_matrix          # KC × PN, число синапсов
    W_kc_mbon0: np.ndarray              # MBON × KC, число синапсов (начальные пластичные веса)
    apl_kc: np.ndarray                  # KC, число синапсов APL→KC
    comp_pam: np.ndarray                # MBON bool: получает PAM-вход
    comp_ppl1: np.ndarray               # MBON bool: получает PPL1-вход
    approach: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    avoid: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))


def load_circuit(side: str = "right", rng: np.random.Generator | None = None, rewire: bool = False) -> Circuit:
    """Подцепь грибовидного тела из data/flywire/mb_{side}_{nodes,edges}.parquet.
    rewire=True — плацебо B: uPN→KC и KC→MBON перепроведены двойными перестановками рёбер (степени и числа
    синапсов сохранены), компартменты DAN и APL — как в данных."""
    nodes = pd.read_parquet(FLYWIRE / f"mb_{side}_nodes.parquet")
    edges = pd.read_parquet(FLYWIRE / f"mb_{side}_edges.parquet")
    pop = lambda p: nodes.loc[nodes["pop"] == p, "id"].to_numpy(np.int64)
    pn, kc, mbon = np.sort(pop("uPN")), np.sort(pop("KC")), np.sort(pop("MBON"))
    # uPN упорядочены по имени гломерулы — детерминированное назначение групп признакам (§3)
    pn_names = nodes.set_index("id").loc[pn, "cell_type"].astype(str)
    pn = pn[np.argsort(pn_names.values, kind="stable")]
    ix = {p: {i: k for k, i in enumerate(arr)} for p, arr in (("pn", pn), ("kc", kc), ("mbon", mbon))}

    def sub(pre, post):
        e = edges[(edges["pre_pop"] == pre) & (edges["post_pop"] == post)]
        return e["pre"].to_numpy(np.int64), e["post"].to_numpy(np.int64), e["syn"].to_numpy(np.float64)

    p_pre, p_post, p_syn = sub("uPN", "KC")
    k_pre, k_post, k_syn = sub("KC", "MBON")
    if rewire:
        assert rng is not None
        p_pre, p_post = _double_edge_swap(p_pre, p_post, rng)
        k_pre, k_post = _double_edge_swap(k_pre, k_post, rng)
    W_pn_kc = sparse.csr_matrix((p_syn, ([ix["kc"][i] for i in p_post], [ix["pn"][i] for i in p_pre])),
                                shape=(len(kc), len(pn)))
    W_kc_mbon0 = np.zeros((len(mbon), len(kc)), np.float32)
    np.add.at(W_kc_mbon0, ([ix["mbon"][i] for i in k_post], [ix["kc"][i] for i in k_pre]), k_syn)
    a_pre, a_post, a_syn = sub("APL", "KC")
    apl_kc = np.zeros(len(kc))
    np.add.at(apl_kc, [ix["kc"][i] for i in a_post], a_syn)
    comp = {c: np.zeros(len(mbon), bool) for c in ("PAM", "PPL1")}
    for c in comp:
        _, d_post, _ = sub(c, "MBON")
        comp[c][[ix["mbon"][i] for i in d_post]] = True
    types = nodes.set_index("id").loc[mbon, "cell_type"].astype(str).to_numpy()
    circ = Circuit(pn, kc, mbon, types, W_pn_kc, W_kc_mbon0, apl_kc, comp["PAM"], comp["PPL1"])
    val = np.array([MBON_VALENCE.get(t.split(",")[0].replace("-like", ""), "unknown") for t in types])
    circ.approach, circ.avoid = val == "approach", val == "avoid"
    return circ


def _double_edge_swap(pre: np.ndarray, post: np.ndarray, rng: np.random.Generator, n_swaps_per_edge: int = 10):
    """Перепроводка двудольного списка рёбер (плацебо B): (a→b, c→d) → (c→b, a→d), если новых дубликатов нет.
    Переставляются ПРЕсинаптические концы, число синапсов остаётся при постсинаптической стороне: каждая KC/MBON
    сохраняет ровно свой набор входных весов (возбудимость не меняется), каждый uPN/KC — число исходящих рёбер;
    меняется только то, КТО с кем соединён."""
    from collections import Counter
    pre, post = pre.copy(), post.copy()
    existing = Counter(zip(pre.tolist(), post.tolist()))     # мультимножество: дубликаты не теряются
    n = len(pre)
    for _ in range(n_swaps_per_edge * n):
        i, j = rng.integers(0, n, 2)
        a, b, c, d = pre[i], post[i], pre[j], post[j]
        if a == c or existing[(c, b)] or existing[(a, d)]:
            continue
        existing[(a, b)] -= 1; existing[(c, d)] -= 1
        existing[(c, b)] += 1; existing[(a, d)] += 1
        pre[i], pre[j] = c, a
    return pre, post


# ============================================================ симуляция одного бара
@njit(cache=True)
def _lif_bar(pn_rates_hz, W_kc_pn_indptr, W_kc_pn_indices, W_kc_pn_data, apl_w, W_mbon_kc, w_unit, apl_gain,
             T, dt, tau_m, v_rest, v_th, v_reset, refrac, tau_syn, seed):
    """LIF-слои KC и MBON на окне T шагов при пуассоновском входе uPN. Возвращает число спайков KC и MBON."""
    np.random.seed(seed)
    n_kc, n_mbon, n_pn = W_mbon_kc.shape[1], W_mbon_kc.shape[0], pn_rates_hz.shape[0]
    v_kc = np.full(n_kc, v_rest); v_mb = np.full(n_mbon, v_rest)
    g_kc = np.zeros(n_kc); g_mb = np.zeros(n_mbon)
    ref_kc = np.zeros(n_kc); ref_mb = np.zeros(n_mbon)
    cnt_kc = np.zeros(n_kc); cnt_mb = np.zeros(n_mbon)
    p_pn = pn_rates_hz * dt / 1000.0
    dec_syn = math.exp(-dt / tau_syn); dec_m = dt / tau_m
    apl = 0.0
    for t in range(T):
        # входные спайки uPN
        pn_spk = np.zeros(n_pn)
        for j in range(n_pn):
            if np.random.random() < p_pn[j]:
                pn_spk[j] = 1.0
        # ток на KC от uPN (CSR: строка = KC)
        for i in range(n_kc):
            s = 0.0
            for q in range(W_kc_pn_indptr[i], W_kc_pn_indptr[i + 1]):
                s += W_kc_pn_data[q] * pn_spk[W_kc_pn_indices[q]]
            g_kc[i] = g_kc[i] * dec_syn + w_unit * s - apl_gain * apl * apl_w[i]
        kc_spk = np.zeros(n_kc)
        for i in range(n_kc):
            if ref_kc[i] > 0:
                ref_kc[i] -= dt
                continue
            v_kc[i] += dec_m * (v_rest - v_kc[i]) + g_kc[i]
            if v_kc[i] >= v_th:
                v_kc[i] = v_reset; ref_kc[i] = refrac; kc_spk[i] = 1.0; cnt_kc[i] += 1.0
        apl = apl * dec_syn + kc_spk.sum() / n_kc            # APL: однородный вход от всех KC (KC→APL нет в данных)
        for m in range(n_mbon):
            s = 0.0
            for i in range(n_kc):
                if kc_spk[i] > 0:
                    s += W_mbon_kc[m, i]
            g_mb[m] = g_mb[m] * dec_syn + w_unit * s
            if ref_mb[m] > 0:
                ref_mb[m] -= dt
                continue
            v_mb[m] += dec_m * (v_rest - v_mb[m]) + g_mb[m]
            if v_mb[m] >= v_th:
                v_mb[m] = v_reset; ref_mb[m] = refrac; cnt_mb[m] += 1.0
    return cnt_kc, cnt_mb


class Fly:
    """Состояние одной «мухи»: цепь, калиброванные w_unit/apl_gain, пластичные веса, трек исходов для дофамина."""

    def __init__(self, circ: Circuit, w_unit: float, apl_gain: float, seed: int):
        self.c, self.w_unit, self.apl_gain, self.seed = circ, w_unit, apl_gain, seed
        self.W = circ.W_kc_mbon0.astype(np.float64).copy()
        self.W0 = self.W.copy()
        self.n_bar = 0
        self.trade_log: list[float] = []

    def reset(self):
        self.W = self.W0.copy(); self.n_bar = 0; self.trade_log = []

    def present(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """z — стандартизованные признаки (NaN недопустимы). Возвращает счётчики спайков KC и MBON."""
        rates = encode_rates(z, len(self.c.pn_ids))
        self.n_bar += 1
        A = self.c.W_pn_kc
        return _lif_bar(rates, A.indptr, A.indices, A.data, self.c.apl_kc, self.W, self.w_unit, self.apl_gain,
                        int(T_PRESENT_MS / DT_MS), DT_MS, LIF["tau_m"], LIF["v_rest"], LIF["v_th"], LIF["v_reset"],
                        LIF["refrac"], LIF["tau_syn"], self.seed * 1_000_003 + self.n_bar)

    def score(self, mb_counts: np.ndarray) -> float:
        ra, rv = mb_counts[self.c.approach].mean(), mb_counts[self.c.avoid].mean()
        return float((ra - rv) / (ra + rv + 1e-9))

    def reinforce(self, elig_kc: np.ndarray, r_trade: float):
        """Дофамин по RPE (§6): δ > 0 → PAM депрессирует KC→MBON в PAM-компартментах; δ ≤ 0 → PPL1 в своих."""
        hist = np.array(self.trade_log[-EMA_TRADES:]) if self.trade_log else np.array([0.0])
        delta = r_trade - hist.mean()
        sd = hist.std() if len(hist) > 2 and hist.std() > 0 else max(abs(delta), 1e-6)
        a = min(2.0, abs(delta) / sd)
        comp = self.c.comp_pam if delta > 0 else self.c.comp_ppl1
        e = elig_kc / max(1.0, elig_kc.max())
        self.W[comp] = np.maximum(0.0, self.W[comp] - ETA * a * e[None, :] * self.W[comp])
        self.trade_log.append(r_trade)

    def recover(self):
        self.W += (self.W0 - self.W) / TAU_REC_BARS


def encode_rates(z: np.ndarray, n_pn: int) -> np.ndarray:
    """§3: признак f → 5 uPN (3 ON кодируют +z, 2 OFF кодируют −z), rate = base + gain·logistic(±z). Лишние PN — base."""
    rates = np.full(n_pn, R_BASE_HZ)
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))
    for f, zf in enumerate(z):
        g0 = f * PN_PER_FEATURE
        if g0 + PN_PER_FEATURE > n_pn:
            break
        rates[g0:g0 + PN_ON_PER_FEATURE] = R_BASE_HZ + R_GAIN_HZ * sig(zf)
        rates[g0 + PN_ON_PER_FEATURE:g0 + PN_PER_FEATURE] = R_BASE_HZ + R_GAIN_HZ * sig(-zf)
    return rates


# ============================================================ калибровка на шуме (§4), без ценовых данных
def calibrate(circ: Circuit, seed: int = 0) -> tuple[float, float]:
    """Подбираем w_unit и apl_gain так, чтобы при всех uPN на 50 Гц доля KC со спайком за окно была в
    KC_TARGET_SPARSITY. Сетка фиксирована; выбирается первая точка, попавшая в диапазон (детерминированно)."""
    n = len(circ.pn_ids)
    for apl_gain in (0.0, 0.5, 1.0, 2.0, 4.0):
        for w_unit in (0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6):
            fly = Fly(circ, w_unit, apl_gain, seed)
            fracs = []
            for k in range(3):
                z = np.zeros(25)
                A = circ.W_pn_kc
                kc, _ = _lif_bar(np.full(n, 50.0), A.indptr, A.indices, A.data, circ.apl_kc, fly.W, w_unit, apl_gain,
                                 int(T_PRESENT_MS / DT_MS), DT_MS, LIF["tau_m"], LIF["v_rest"], LIF["v_th"],
                                 LIF["v_reset"], LIF["refrac"], LIF["tau_syn"], seed * 7 + k)
                fracs.append(float((kc > 0).mean()))
            f = float(np.mean(fracs))
            if KC_TARGET_SPARSITY[0] <= f <= KC_TARGET_SPARSITY[1]:
                return w_unit, apl_gain
    raise RuntimeError("калибровка не нашла точку сетки с нужной разрежённостью KC — проверить LIF/веса")


# ============================================================ walk-forward
def run_fly(df: pd.DataFrame, tf: str, seed: int, placebo: str | None = None, horizon_hours: int = 48,
            fee_bps: float = 5.0, tp: float = 2.0, sl: float = 1.5, test_days: int = 90, min_train_days: int = 365,
            n_trials_prior: int = 0, cutoff=HOLDOUT_START) -> dict:
    if not MBON_VALENCE:
        raise RuntimeError("MBON_VALENCE пуст: спецификация не заморожена (см. H-FlyBrain.md [ЛИТ])")
    rng = np.random.default_rng(seed)
    df = before_holdout(df, cutoff)
    bpd = BARS_PER_DAY[tf]
    h = max(1, int(round(horizon_hours * bpd / 24)))
    X = build_features_v1(df, bpd)
    prim = primary_side(X).values
    if placebo == "features":                       # плацебо A: признаки перемешаны блоками по 5 дней
        blocks = np.array_split(np.arange(len(X)), max(2, len(X) // (5 * bpd)))
        order = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
        X = pd.DataFrame(X.values[order], index=X.index, columns=X.columns)
    circ = load_circuit(rng=rng, rewire=(placebo == "topology"))
    w_unit, apl_gain = calibrate(circ, seed)
    fly = Fly(circ, w_unit, apl_gain, seed)
    tb_long = triple_barrier_sided(df, h, np.ones(len(df)), tp, sl, vol_span=7 * bpd)
    tb_short = triple_barrier_sided(df, h, -np.ones(len(df)), tp, sl, vol_span=7 * bpd)
    cost = 2 * fee_bps / 1e4
    valid = ~X.isna().any(axis=1).values & tb_long["ret"].notna().values & tb_short["ret"].notna().values
    start = int(np.searchsorted(np.cumsum(valid), min_train_days * bpd)) + h
    folds = Folds(len(df), h, min_train=start, test_len=test_days * bpd)
    test_idx = np.concatenate([nonoverlap(te, h) for _, te in folds])
    test_idx = test_idx[valid[test_idx]]
    pos = {t: i for i, t in enumerate(test_idx)}
    s_out = np.full(len(test_idx), np.nan)
    Xv = X.values
    for tr, te in folds:
        tr = tr[valid[tr]]
        te = nonoverlap(te, h)
        te = te[valid[te]]
        if len(tr) < 200 or not len(te):
            continue
        mu, sd = Xv[tr].mean(0), Xv[tr].std(0) + 1e-9     # стандартизация только по обучению
        fly.reset()
        pending: list[tuple[int, np.ndarray, float, int]] = []      # (exit_bar, elig, side, t)
        for t in tr:                                               # обучение онлайн, по времени
            due = [p for p in pending if p[0] <= t]
            for exit_bar, elig, side, t0 in due:
                tb = tb_long if side > 0 else tb_short
                fly.reinforce(elig, float(tb["ret"].values[t0]) - cost)
            pending = [p for p in pending if p[0] > t]
            fly.recover()
            kc, mb = fly.present((Xv[t] - mu) / sd)
            s = fly.score(mb)
            side = 1.0 if s > DEAD_ZONE else (-1.0 if s < -DEAD_ZONE else 0.0)
            if side != 0:
                tb = tb_long if side > 0 else tb_short
                pending.append((int(tb["exit_bar"].values[t]), kc, side, int(t)))
        for t in te:                                               # тест: веса заморожены
            _, mb = fly.present((Xv[t] - mu) / sd)
            s_out[pos[t]] = fly.score(mb)
    have = ~np.isnan(s_out)
    test_idx, s_out = test_idx[have], s_out[have]
    r_long, r_short = tb_long["ret"].values[test_idx], tb_short["ret"].values[test_idx]
    tpw_l, slw_l = tb_long["tp_w"].values[test_idx], tb_long["sl_w"].values[test_idx]

    def sided(side):
        return np.where(side > 0, r_long, np.where(side < 0, r_short, 0.0))

    side_d0 = np.sign(s_out)
    side_d1 = np.where(s_out > DEAD_ZONE, 1.0, np.where(s_out < -DEAD_ZONE, -1.0, 0.0))
    pside = prim[test_idx]
    side_m = np.where((pside != 0) & (s_out * pside > 0), pside, 0.0)
    # у конфигураций разные направления, барьеры асимметричны → каждая несёт свой результат (r_own), side — маска
    configs = [Config(name, (d != 0).astype(float), r_own=sided(d))
               for name, d in (("fly D0: sign(s)", side_d0), ("fly D1: мёртвая зона 0.1", side_d1),
                               ("fly M: primary, если s согласен", side_m))]
    n_trials = n_trials_prior + len(configs)
    pbo = score_configs(configs, np.zeros(len(test_idx)), cost, 365 * bpd / h, n_trials=n_trials)
    base = [Config("ориентир: всегда long (те же барьеры)", np.ones(len(test_idx))),
            Config("ориентир: случайная сторона", rng.choice([-1.0, 1.0], len(test_idx)), r_own=None)]
    base[1].r_own = sided(base[1].side); base[1].side = np.ones(len(test_idx))
    score_configs(base, r_long, cost, 365 * bpd / h, n_trials=n_trials)
    long_mean = base[0].stats.get("mean_bps", -1e9)
    ranked = sorted([c for c in configs if c.stats.get("trades", 0) >= 10], key=lambda c: -c.stats["dsr"])
    best = ranked[0] if ranked else None
    gate = bool(best and best.stats["dsr"] > 0.95 and best.stats["ci_lo_bps"] > 0 and best.stats["trades"] >= 100
                and (np.isnan(pbo["pbo"]) or pbo["pbo"] < 0.2) and best.stats["mean_bps"] > long_mean)
    return {"configs": configs, "baselines": base, "pbo": pbo, "test_idx": test_idx, "s": s_out, "n_trials": n_trials,
            "best": best, "gate": gate, "w_unit": w_unit, "apl_gain": apl_gain, "placebo": placebo, "seed": seed,
            "h": h, "bpd": bpd, "n_test": len(test_idx), "fee_bps": fee_bps, "tf": tf, "horizon_hours": horizon_hours,
            "period": (df.index[test_idx[0]], df.index[test_idx[-1]]) if len(test_idx) else None}


def report_fly(res: dict, name: str) -> str:
    L = [f"# H-FlyBrain · {name} · seed {res['seed']}" + (f" · ПЛАЦЕБО {res['placebo']}" if res["placebo"] else ""), "",
         f"Горизонт {res['h']} баров ({res['horizon_hours']} ч); цель +2σ, стоп −1.5σ; комиссия {res['fee_bps']:g} б.п. за сторону. "
         f"Тестовые события: {res['n_test']}, период {res['period'][0]} — {res['period'][1]} (до holdout {HOLDOUT_START.date()}). "
         f"Калибровка на шуме: w_unit = {res['w_unit']:g}, apl_gain = {res['apl_gain']:g}. "
         f"PBO: **{res['pbo']['pbo']:.2f}**. N для DSR: **{res['n_trials']}**.", "",
         "| конфигурация | сделок | прибыльных | средняя, б.п. | 95% ДИ, б.п. | PSR | DSR |", "|---|---|---|---|---|---|---|"]
    for c in [*res["configs"], *res["baselines"]]:
        t = c.stats
        if t.get("trades", 0) < 10:
            L.append(f"| {c.name} | {t.get('trades', 0)} | — | — | — | — | — |")
            continue
        L.append(f"| {c.name} | {t['trades']} | {t['hit']:.1%} | {t['mean_bps']:+.1f} | [{t['ci_lo_bps']:+.0f}; {t['ci_hi_bps']:+.0f}] | "
                 f"{t['psr']:.2f} | {t['dsr']:.2f} |")
    b = res["best"]
    L += ["", "## Вердикт", "", ("**Гейт пройден**" if res["gate"] else "**Гейт НЕ пройден**")
          + (f": лучшая — {b.name}, DSR {b.stats['dsr']:.2f}, средняя {b.stats['mean_bps']:+.1f} б.п., {b.stats['trades']} сделок." if b else "."),
          "", "Требования гейта: DSR > 0.95, PBO < 0.2, нижняя граница 95% ДИ > 0, ≥ 100 сделок, лучше «всегда long»."]
    sus = [c for c in res["configs"] if c.stats.get("trades", 0) >= 100 and c.stats["hit"] > 0.62]
    if sus:
        L += ["", f"⚠️ **Подозрение на утечку:** {', '.join(c.name for c in sus)} — > 62 % прибыльных на ≥ 100 сделках."]
    return "\n".join(L)


def registry_rows(res: dict, asset: str) -> list[dict]:
    hyp = "H-FlyBrain" + (f"-placebo" if res["placebo"] else "")
    rows = []
    for c in res["configs"]:
        t = c.stats
        rows.append({"hypothesis": hyp, "asset": asset, "tf": res["tf"], "model": "MB-connectome-LIF",
                     "features": "F1-F12 → uPN Poisson", "barriers": "tp2σ/sl1.5σ", "horizon_h": res["horizon_hours"],
                     "rule": c.name + f" (seed {res['seed']}" + (f", placebo {res['placebo']})" if res["placebo"] else ")"),
                     "period_start": res["period"][0] if res["period"] else "", "period_end": res["period"][1] if res["period"] else "",
                     "holdout": False, "hit": t.get("hit"), "mean_bps": t.get("mean_bps"), "ci_lo_bps": t.get("ci_lo_bps"),
                     "ci_hi_bps": t.get("ci_hi_bps"), **trade_moments(c.rets), "n_trials_at_reg": res["n_trials"],
                     "dsr_at_reg": t.get("dsr"), "pbo": res["pbo"]["pbo"],
                     "verdict": "gate" if (c is res["best"] and res["gate"]) else "",
                     "notes": f"w_unit={res['w_unit']:g} apl_gain={res['apl_gain']:g}"})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tf", default="4h", choices=list(BARS_PER_DAY))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--placebo", choices=["features", "topology"], default=None)
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args()
    name = Path(a.data).stem
    res = run_fly(load(a.data), a.tf, a.seed, a.placebo, n_trials_prior=n_trials())
    if not a.no_register:
        register(registry_rows(res, name.split("_")[0]))
    rep = report_fly(res, name)
    out = Path("reports") / f"FlyBrain_{name}_seed{a.seed}{'_placebo-' + a.placebo if a.placebo else ''}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(rep, encoding="utf-8")
    print(rep, f"\n\nСохранено: {out}", sep="")


if __name__ == "__main__":
    main()
