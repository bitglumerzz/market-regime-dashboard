"""H-FlyBrain-2, ядро: код клеток Кеньона как locality-sensitive hashing и фильтр Блума мухи (детектор новизны).

Первоисточники (сверены, см. docs/research/experiments/H-FlyBrain-2.md):
  Dasgupta, Stevens, Navlakha, Science 2017 — FlyHash: нормализация → разрежённая бинарная проекция (≈6 PN на KC,
  расширение 50→2000) → winner-take-all 5 % (APL). Dasgupta, Sheehan, Stevens, Navlakha, PNAS 2018 — фильтр Блума:
  память w = 1; при предъявлении активные биты тега обнуляются (w ← w ∧ ¬h(x)); новизна = доля активных KC запроса,
  которых память не видела; небинарный вариант w = (1−ε)^c; чувствительность к сходству и ко времени.

Здесь проекция — НАСТОЯЩАЯ матрица uPN→KC правого полушария FlyWire v783 (139 → 2597, веса = число синапсов), а не
случайная: это и есть проверяемый вопрос «важна ли именно проводка мухи» (плацебо — перепроводка с сохранением степеней).
Никакого обучения с подкреплением; никаких ценовых данных в этом модуле.

Ёмкость (важно для потоков): тег занимает k = 5 % · m ≈ 130 бит из m = 2597; после n вставок доля «виденных» бит
≈ 1 − exp(−k·n/m) = 1 − exp(−n/20). Без затухания фильтр насыщается за ~20–60 состояний, и новизна ≡ 0 — поэтому на
рыночном потоке память обязана затухать (tau_bars) и означает «непохоже ни на что за последние ~τ баров».
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .flybrain import Circuit, encode_rates

KC_ACTIVE_FRACTION = 0.05          # Dasgupta 2017: «all but the highest firing 5% of Kenyon cells are silenced»


PN_PER_SETUP = 4                     # группа uPN на один элемент словаря сетапов (139 uPN → до 34 сетапов)
SETUP_ON_HZ, SETUP_OFF_HZ = 100.0, 5.0


def encode_setups(states: np.ndarray, n_pn: int, pn_per_setup: int = PN_PER_SETUP) -> np.ndarray:
    """«Запах» из словаря сетапов: states — вектор в [0,1] (0/1 для бинарных сетапов, доли — для категориальных
    с несколькими уровнями, заранее приведённых к [0,1]). Сетап i занимает свою группу из pn_per_setup uPN:
    rate = OFF + (ON − OFF)·state_i. Группы не пересекаются (стимул-специфичный код, как у запахов); лишние uPN — OFF.
    Это вход для flyhash: там он центрируется и проецируется реальной проводкой uPN→KC."""
    states = np.asarray(states, float)
    if len(states) * pn_per_setup > n_pn:
        raise ValueError(f"словарь из {len(states)} сетапов не помещается в {n_pn} uPN по {pn_per_setup}")
    rates = np.full(n_pn, SETUP_OFF_HZ)
    for i, s in enumerate(states):
        if np.isnan(s):
            continue
        rates[i * pn_per_setup:(i + 1) * pn_per_setup] = SETUP_OFF_HZ + (SETUP_ON_HZ - SETUP_OFF_HZ) * float(np.clip(s, 0, 1))
    return rates


def flyhash_rates(circ: Circuit, rates: np.ndarray, k_frac: float = KC_ACTIVE_FRACTION) -> np.ndarray:
    """FlyHash от готовых частот uPN (для категориальных «запахов»): центрирование → проекция → WTA."""
    x = rates - rates.mean()
    kc = circ.W_pn_kc @ x
    k = max(1, int(round(k_frac * len(kc))))
    tag = np.zeros(len(kc), bool)
    tag[np.argpartition(-kc, k - 1)[:k]] = True
    return tag


def flyhash(circ: Circuit, z: np.ndarray, k_frac: float = KC_ACTIVE_FRACTION) -> np.ndarray:
    """Бинарный тег KC для стандартизованного вектора признаков z (25 → 139 uPN → 2597 KC → top-k).
    Детерминированный (без Пуассона): вход PN = та же кодировка частот §3, что у flybrain (ON/OFF-логистика),
    центрированная по среднему (divisive/центрирование из Dasgupta 2017, шаг 1); KC = сумма синаптических входов."""
    rates = encode_rates(z, len(circ.pn_ids))
    x = rates - rates.mean()                                   # шаг 1: центрирование
    kc = circ.W_pn_kc @ x                                      # шаг 2: разрежённая проекция реальной проводкой
    k = max(1, int(round(k_frac * len(kc))))
    tag = np.zeros(len(kc), bool)
    tag[np.argpartition(-kc, k - 1)[:k]] = True                # шаг 3: winner-take-all (роль APL)
    return tag


@dataclass
class FlyBloom:
    """Фильтр Блума мухи с затуханием по времени: память w ∈ [0,1]^m, 1 = «никогда не видел».
    insert: активные биты тега уходят к 0 с силой ε (ε = 1 — бинарный фильтр Dasgupta 2018);
    tick: восстановление к 1 с постоянной tau_bars (время, прошедшее с последнего предъявления — PNAS 2018, форма наша);
    novelty: средняя «невиданность» активных KC запроса ∈ [0,1]."""
    m: int
    eps: float = 1.0
    tau_bars: float = np.inf

    def __post_init__(self):
        self.w = np.ones(self.m)

    def novelty(self, tag: np.ndarray) -> float:
        return float(self.w[tag].mean()) if tag.any() else 1.0

    def insert(self, tag: np.ndarray) -> None:
        self.w[tag] *= (1.0 - self.eps)

    def tick(self) -> None:
        if np.isfinite(self.tau_bars):
            self.w += (1.0 - self.w) / self.tau_bars


def uniform_circuit(like: Circuit, rng: np.random.Generator, s: int = 6) -> Circuit:
    """Плацебо «равномерная проекция» (Dasgupta 2017): каждая KC получает s случайных uPN с равными весами,
    средний суммарный вход на KC — как у реальной проводки. Проверяет, важна ли именно схема мухи (хабы и т.п.),
    или любая разрежённая случайная проекция хеширует не хуже."""
    n_kc, n_pn = like.W_pn_kc.shape
    w_mean = like.W_pn_kc.sum() / (n_kc * s)
    rows = np.repeat(np.arange(n_kc), s)
    cols = np.concatenate([rng.choice(n_pn, s, replace=False) for _ in range(n_kc)])
    W = sparse.csr_matrix((np.full(len(rows), w_mean), (rows, cols)), shape=(n_kc, n_pn))
    c = Circuit(like.pn_ids, like.kc_ids, like.mbon_ids, like.mbon_types, W, like.W_kc_mbon0, like.apl_kc,
                like.comp_pam, like.comp_ppl1, like.approach, like.avoid)
    return c


def tags_matrix(circ: Circuit, Z: np.ndarray, k_frac: float = KC_ACTIVE_FRACTION) -> sparse.csr_matrix:
    """Разрежённая матрица тегов (строки = бары, столбцы = KC) для линейного считывания; NaN-строки → пустые теги."""
    rows, cols = [], []
    for t in range(len(Z)):
        if np.isnan(Z[t]).any():
            continue
        idx = np.where(flyhash(circ, Z[t], k_frac))[0]
        rows.extend([t] * len(idx)); cols.extend(idx.tolist())
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(Z), len(circ.kc_ids)))


def novelty_series(circ: Circuit, Z: np.ndarray, eps: float = 1.0, tau_bars: float = np.inf,
                   warmup: int = 0) -> np.ndarray:
    """Каузальная новизна каждой строки Z относительно всех ПРЕДЫДУЩИХ строк: nov[t] считается до insert(t).
    Первые warmup строк только заполняют память (nov = NaN)."""
    fb = FlyBloom(len(circ.kc_ids), eps, tau_bars)
    out = np.full(len(Z), np.nan)
    for t in range(len(Z)):
        fb.tick()
        if np.isnan(Z[t]).any():
            continue
        tag = flyhash(circ, Z[t])
        if t >= warmup:
            out[t] = fb.novelty(tag)
        fb.insert(tag)
    return out
