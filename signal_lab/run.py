"""Запуск лаборатории: python -m signal_lab.run --data data/BTCUSDT_4h.parquet --tf 4h --horizon 6

Печатает и сохраняет отчёт reports/signal_lab_<актив>_<tf>_h<h>.md:
  1) какие признаки по отдельности предсказывают направление (с поправкой на множественное тестирование),
  2) как работает модель на всех признаках с правилом «входить только при уверенности»,
  3) вердикт гейта качества.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import Folds, attach_dsr, factor_tests, model_tests
from .features import build_features, forward_return

BARS_PER_DAY = {"15m": 96, "30m": 48, "1h": 24, "2h": 12, "4h": 6, "8h": 3, "12h": 2, "1d": 1}


def load(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    if "ts" in df:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, unit="ms" if np.issubdtype(df["ts"].dtype, np.number) else None)
        df = df.set_index("ts")
    return df.sort_index()


def analyze(df: pd.DataFrame, tf: str, horizon: int, fee_bps: float = 10.0, test_days: int = 90, min_train_days: int = 365,
            name: str = "asset") -> tuple[str, dict]:
    bpd = BARS_PER_DAY[tf]
    X = build_features(df, bpd)
    fwd = forward_return(df["close"], horizon)
    folds = Folds(len(df), horizon, min_train=min_train_days * bpd, test_len=test_days * bpd)
    periods = 365 * bpd / horizon

    factors = factor_tests(X, fwd, horizon, folds)
    models = model_tests(X, fwd, horizon, folds, periods, fee_bps)
    attach_dsr(models, extra_trials=len(factors))
    base_up = float((fwd.dropna() > 0).mean())

    best = max(models, key=lambda m: m.dsr, default=None)
    gate = bool(best and best.dsr > 0.95 and best.mean_ret_bps > 0 and best.trades >= 100)
    lines = [
        f"# Лаборатория сигналов: {name} · {tf} · горизонт {horizon} баров ({horizon / bpd:.2f} дн.)",
        "",
        f"Данные: {df.index[0]} — {df.index[-1]}, баров: {len(df)}. Walk-forward: старт после {min_train_days} дн., "
        f"тестовые блоки по {test_days} дн., purge {horizon} баров, сделки не пересекаются. Комиссия {fee_bps} б.п. за сторону.",
        f"Доля растущих периодов (базовая линия «всегда long»): **{base_up:.1%}**.",
        "",
        "## 1. Отдельные признаки",
        "Попадание в направление на тестовых данных. q — p-value с поправкой Бенджамини–Хохберга: значимым считаем q < 0.05.",
        "",
        "| признак | сделок | попадание | p | q | IC | t(IC) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in factors:
        mark = " ✅" if r.q_value < 0.05 else ""
        lines.append(f"| {r.name}{mark} | {r.n} | {r.hit:.1%} | {r.p_value:.3f} | {r.q_value:.3f} | {r.ic:+.3f} | {r.ic_t:+.2f} |")
    lines += ["", "## 2. Модель на всех признаках (с воздержанием при низкой уверенности)", "",
              "| модель | порог |p−0.5| | сделок | доля баров | попадание | p | средняя сделка, б.п. | Sharpe/год | DSR |",
              "|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        lines.append(f"| {m.model} | {m.threshold:.2f} | {m.trades} | {m.coverage:.0%} | {m.hit:.1%} | {m.p_value:.3f} | "
                     f"{m.mean_ret_bps:+.1f} | {m.sharpe:+.2f} | {m.dsr:.2f} |")
    lines += ["", "## 3. Вердикт",
              ("**Гейт пройден**: " if gate else "**Гейт НЕ пройден**: ") +
              (f"лучшая конфигурация {best.model}/{best.threshold:.2f} — DSR {best.dsr:.2f}, средняя сделка {best.mean_ret_bps:+.1f} б.п., "
               f"{best.trades} сделок." if best else "недостаточно данных.") +
              " Требование: DSR > 0.95, средняя сделка после комиссий > 0, ≥ 100 непересекающихся сделок. "
              "Даже при прохождении — сначала бесплатная бета и live-статистика.",
              "",
              "_Попадание 52–56% на непересекающихся сделках — это уже сильный результат для рынка; всё, что выше 60%, "
              "почти наверняка означает утечку данных или слишком короткую выборку._"]
    return "\n".join(lines), {"factors": factors, "models": models, "gate": gate}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="parquet/csv: ts, open, high, low, close, volume [, taker_buy_volume, funding, oi]")
    ap.add_argument("--tf", default="4h", choices=list(BARS_PER_DAY))
    ap.add_argument("--horizon", type=int, default=6, help="горизонт в барах")
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    df = load(a.data)
    name = a.name or Path(a.data).stem
    report, _ = analyze(df, a.tf, a.horizon, a.fee_bps, name=name)
    out = Path("reports") / f"signal_lab_{name}_h{a.horizon}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()
