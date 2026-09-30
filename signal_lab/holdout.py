"""Однократная проверка замороженных спецификаций на holdout (01.04.2025 – последняя закрытая свеча).

    python -m signal_lab.holdout --dry-run                          # проверка кода: фиктивный holdout с 01.04.2024
    python -m signal_lab.holdout --i-understand-this-runs-once      # НАСТОЯЩЕЕ открытие — один раз

Спецификации заморожены по результатам периода разработки (reports/trials.csv, docs/research/experiments/):
  • H1: long, вес min(1, 0.25/σ̂_20д), полосы 15 %, 5 б.п. — spot BTC/ETH, дневные бары;
  • H2: ансамбль Дончиана Zarattini v1 (9 окон, стоп-храповик по середине, 0.25/σ90 кап 2, порог 20 % по σ, 10 б.п.)
    — spot BTC/ETH; критерий: Sharpe ≥ 0.8 и DSR > 0.95;
  • v1 (H3): мета-разметка, 48 ч, цель +2σ / стоп −1.5σ, все признаки F1–F12, 5 б.п. за сторону — перпетуалы BTC/ETH 4h.
Никакие параметры после открытия не меняются. Повторное открытие запрещено registry.open_holdout.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd

from . import registry
from .evaluate import deflated_sharpe
from .h1 import perf, strategy_returns, vol_target_weights
from .h2 import WINDOWS, model_weights, strat_returns
from .run import load
from .v1 import evaluate_v1, registry_rows, report_v1


def run(start: pd.Timestamp, end: pd.Timestamp | None, trials: Path) -> str:
    """start — начало проверяемого периода; end — обрезка данных (для dry-run); trials — файл реестра."""
    cut = (lambda d: d[d.index < end]) if end is not None else (lambda d: d)
    L = ["# Holdout: однократная проверка замороженных спецификаций", "",
         f"Период: с {start.date()} до {'последней закрытой свечи' if end is None else end.date()}.", ""]
    rows = []
    for asset in ("BTCUSDT", "ETHUSDT"):
        c = cut(load(f"data/{asset}_spot_1d.parquet"))["close"]
        vm = strategy_returns(c, vol_target_weights(c), 5.0)
        vm = vm[vm.index >= start]
        bh = c.pct_change()[vm.index]
        pv, pb = perf(vm), perf(bh)
        x = strat_returns(c, model_weights(c, WINDOWS, "mid_ratchet", version="v1"), 10.0)
        x = x[x.index >= start]
        p2 = perf(x)
        N = registry.n_trials(trials) + 1
        dsr2 = deflated_sharpe(x.values, N)
        ok2 = p2["sharpe"] >= 0.8 and dsr2 > 0.95
        L += [f"## {asset} — дневные бары, {vm.index[0].date()} — {vm.index[-1].date()} ({len(vm)} дней)", "",
              "| стратегия | Sharpe | CAGR | вол. | MDD | DSR |", "|---|---|---|---|---|---|",
              f"| buy & hold | {pb['sharpe']:.2f} | {pb['cagr']:+.1%} | {pb['vol']:.1%} | {pb['mdd']:.1%} | — |",
              f"| H1: размер по волатильности | {pv['sharpe']:.2f} | {pv['cagr']:+.1%} | {pv['vol']:.1%} | {pv['mdd']:.1%} | — |",
              f"| H2: ансамбль Дончиана (v1) | {p2['sharpe']:.2f} | {p2['cagr']:+.1%} | {p2['vol']:.1%} | {p2['mdd']:.1%} | "
              f"{dsr2:.2f} (N={N}) |", "",
              f"H1: Sharpe {'выше' if pv['sharpe'] > pb['sharpe'] else 'не выше'} B&H, просадка "
              f"{'ниже' if pv['mdd'] < pb['mdd'] else 'не ниже'}. H2 (Sharpe ≥ 0.8 и DSR > 0.95): "
              f"**{'выполнен' if ok2 else 'не выполнен'}**.", ""]
        rows += [{"hypothesis": "H1", "asset": asset, "tf": "1d", "model": "vol-target", "rule": "w=min(1,0.25/σ20) band15%",
                  "period_start": vm.index[0], "period_end": vm.index[-1], "holdout": True,
                  "mean_bps": float(vm.mean() * 1e4), **registry.trade_moments(vm.values), "n_trials_at_reg": N,
                  "verdict": "holdout", "notes": f"Sharpe {pv['sharpe']:.2f} vs B&H {pb['sharpe']:.2f}; MDD {pv['mdd']:.1%} vs {pb['mdd']:.1%}"},
                 {"hypothesis": "H2", "asset": asset, "tf": "1d", "model": "donchian", "features": ",".join(map(str, WINDOWS)),
                  "barriers": "mid_ratchet", "rule": "ансамбль · mid_ratchet; v1: 0.25/σ90 cap2 band20%(σ) 10bps",
                  "period_start": x.index[0], "period_end": x.index[-1], "holdout": True, "mean_bps": float(x.mean() * 1e4),
                  **registry.trade_moments(x.values), "n_trials_at_reg": N, "dsr_at_reg": dsr2,
                  "verdict": "выполнен" if ok2 else "не выполнен",
                  "notes": f"Sharpe {p2['sharpe']:.2f} CAGR {p2['cagr']:+.1%} MDD {p2['mdd']:.1%}"}]
    registry.register(rows, trials)
    for asset in ("BTCUSDT", "ETHUSDT"):
        df = cut(load(f"data/{asset}_4h.parquet"))
        res = evaluate_v1(df, "4h", 48, n_trials_prior=registry.n_trials(trials), holdout=True, cutoff=start)
        registry.register(registry_rows(res, asset, "H3"), trials)
        L += [f"## v1 (H3) · {asset} · 48 ч", "", report_v1(res, f"{asset}_4h").split("\n", 2)[2], ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--i-understand-this-runs-once", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        tmp = Path(tempfile.mkdtemp()) / "trials.csv"
        rep = run(pd.Timestamp("2024-04-01", tz="UTC"), registry.HOLDOUT_START, tmp)
        print(rep, f"\n\n[dry-run] реестр: {tmp}; настоящий holdout не открыт", sep="")
        return
    registry.open_holdout("финальная проверка: H1, H2 (Zarattini v1 ансамбль), v1 48h — BTC/ETH")
    rep = run(registry.HOLDOUT_START, None, registry.TRIALS_PATH)
    out = Path("reports") / "H_holdout.md"
    out.write_text(rep, encoding="utf-8")
    print(rep, f"\n\nСохранено: {out}", sep="")


if __name__ == "__main__":
    main()
