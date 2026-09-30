"""Реестр испытаний и защита отложенной выборки (holdout).

Каждая проверенная спецификация — строка в reports/trials.csv (набор признаков, барьеры, правило входа, горизонт,
актив, модель + результат). Строки не удаляются никогда: их общее число — N для Deflated Sharpe Ratio. Если после
просмотра результата что-то поменяли — это новое испытание, новая строка.

Holdout — период с HOLDOUT_START по сегодня. До заморозки спецификации оценка обязана видеть только данные раньше
этой даты (`before_holdout`). Открыть holdout можно один раз (`open_holdout`), факт открытия пишется в реестр.
"""
from __future__ import annotations

import csv
import datetime as dt
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HOLDOUT_START = pd.Timestamp("2025-04-01", tz="UTC")
TRIALS_PATH = Path("reports") / "trials.csv"
FIELDS = ["trial_id", "date", "hypothesis", "asset", "tf", "model", "features", "barriers", "horizon_h", "rule",
          "period_start", "period_end", "holdout", "n_trades", "hit", "mean_bps", "ci_lo_bps", "ci_hi_bps",
          "sr_trade", "skew", "kurt", "n_trials_at_reg", "dsr_at_reg", "pbo", "verdict", "notes"]


def before_holdout(df: pd.DataFrame, start: pd.Timestamp = HOLDOUT_START) -> pd.DataFrame:
    """Обрезка до holdout. Обрезаем исходные бары до построения признаков и меток: тогда ни признак, ни барьер
    сделки физически не могут увидеть цену из отложенного периода."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("нужен DatetimeIndex, чтобы отрезать holdout")
    return df[df.index < start]


def read_trials(path: Path = TRIALS_PATH) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def n_trials(path: Path = TRIALS_PATH) -> int:
    """Сколько спецификаций уже проверено (N для DSR)."""
    return len(read_trials(path))


def register(rows: list[dict], path: Path = TRIALS_PATH) -> list[int]:
    """Дописать испытания в реестр. Возвращает присвоенные trial_id. Ничего не перезаписывает."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_trials(path)
    next_id = 1 + max((int(r["trial_id"]) for r in existing), default=0)
    new = not path.exists()
    ids = []
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            r = {k: _fmt(v) for k, v in r.items()}
            r.setdefault("date", dt.date.today().isoformat())
            r["trial_id"] = next_id
            w.writerow(r)
            ids.append(next_id)
            next_id += 1
    return ids


def holdout_opened(path: Path = TRIALS_PATH) -> bool:
    return any(str(r.get("holdout", "")).lower() in ("true", "1", "yes") for r in read_trials(path))


def open_holdout(reason: str, path: Path = TRIALS_PATH) -> None:
    """Однократное открытие holdout: вторая попытка — исключение. Пишет строку-отметку в реестр."""
    if holdout_opened(path):
        raise RuntimeError("holdout уже открыт — повторная проверка на нём запрещена")
    register([{"hypothesis": "HOLDOUT", "rule": "открытие holdout", "holdout": True,
               "period_start": HOLDOUT_START.date().isoformat(), "notes": reason, "verdict": "открыт"}], path)


def trade_moments(r: np.ndarray) -> dict:
    """Достаточная статистика сделки для пересчёта DSR при любом N позже: n, Sharpe за сделку, асимметрия, эксцесс."""
    r = np.asarray(r, float)
    if len(r) < 10 or r.std() == 0:
        return {"n_trades": len(r)}
    return {"n_trades": len(r), "sr_trade": r.mean() / r.std(ddof=1), "skew": float(stats.skew(r)),
            "kurt": float(stats.kurtosis(r, fisher=False))}


def dsr_from_moments(sr: float, n: int, skew: float, kurt: float, n_trials: int) -> float:
    """DSR (Bailey & López de Prado 2014) по моментам — тот же расчёт, что evaluate.deflated_sharpe."""
    if not n or n < 10 or not np.isfinite(sr):
        return 0.0
    emc = 0.5772156649
    n_trials = max(2, int(n_trials))
    s0 = 1 / math.sqrt(n - 1)
    sr0 = s0 * ((1 - emc) * stats.norm.ppf(1 - 1 / n_trials) + emc * stats.norm.ppf(1 - 1 / (n_trials * math.e)))
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(n - 1) / denom))


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{float(v):.6g}"
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return v
