"""H6 — режим финансирования: предсказывает ли долгий отрицательный фандинг рост за 30 дней.

    python -m signal_lab.h6 --data data/BTCUSDT_4h.parquet

Событие (зафиксировано до прогона, раздел 4 отчёта directional-edge): на закрытии дня t средняя ставка финансирования
за 30 дней < 0 и держится ниже нуля не меньше 14 дней подряд. Эпизоды не пересекаются: следующее событие —
не раньше чем через 30 дней после предыдущего. Исход — доходность close(t+30д)/close(t) − 1; окно должно закончиться
до holdout. Сравнение — с безусловной 30-дневной доходностью («всегда long») на том же периоде.
Ставка известна в момент начисления (00/08/16 UTC), поэтому среднее на закрытии дня использует только прошлые начисления.
Ожидание: эпизодов мало (<10) → вердикт «нет ответа», а не «работает».
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .registry import before_holdout, n_trials, register
from .run import load


def funding_events(df4h: pd.DataFrame, mean_days: int = 30, min_run_days: int = 14, horizon_days: int = 30) -> pd.DataFrame:
    d = df4h[["close", "funding"]].resample("1D", label="left", closed="left").last()   # значение на конец дня
    f8 = df4h["funding"].resample("8h").first()                                         # одно значение на начисление
    m = f8.rolling(3 * mean_days, min_periods=3 * mean_days).mean().resample("1D").last().reindex(d.index)
    neg = (m < 0).astype(int)
    run = neg.groupby((neg != neg.shift()).cumsum()).cumsum()
    cond = (m < 0) & (run >= min_run_days)
    fwd = d["close"].shift(-horizon_days) / d["close"] - 1
    events, last = [], None
    for t in d.index[cond.fillna(False).values]:
        if last is not None and (t - last).days < horizon_days:
            continue
        if np.isnan(fwd.get(t, np.nan)):
            continue
        events.append({"date": t, "mean_funding_30d": float(m[t]), "run_days": int(run[t]), "fwd_30d": float(fwd[t])})
        last = t
    uncond = fwd.dropna()
    return pd.DataFrame(events), uncond


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args()
    name = Path(a.data).stem
    df = before_holdout(load(a.data))
    ev, unc = funding_events(df)
    n = len(ev)
    lines = [f"# H6 — долгий отрицательный фандинг → доходность за 30 дней · {name}", "",
             f"Период {df.index[0].date()} — {df.index[-1].date()} (до holdout). Событие: 30-дневное среднее фандинга < 0 "
             "не меньше 14 дней подряд; эпизоды не пересекаются.", "",
             f"Безусловная 30-дневная доходность (все дни): среднее {unc.mean():+.1%}, медиана {unc.median():+.1%}, "
             f"доля роста {np.mean(unc > 0):.0%}, дней {len(unc)}.", ""]
    if n:
        lines += ["| дата события | фандинг 30д, % за 8 ч | дней подряд < 0 | доходность 30д |", "|---|---|---|---|"]
        lines += [f"| {r.date.date()} | {r.mean_funding_30d * 100:+.4f} | {r.run_days} | {r.fwd_30d:+.1%} |" for r in ev.itertuples()]
        t = stats.ttest_1samp(ev.fwd_30d, unc.mean()) if n > 1 else None
        lines += ["", f"Эпизодов: **{n}**. Средняя доходность после события {ev.fwd_30d.mean():+.1%} против безусловной "
                  f"{unc.mean():+.1%}; рост в {int((ev.fwd_30d > 0).sum())} из {n}"
                  + (f"; t-тест разницы средних: t = {t.statistic:+.2f}, p = {t.pvalue:.2f}." if t is not None else ".")]
    verdict = "нет ответа" if n < 10 else ("подтверждена" if n and ev.fwd_30d.mean() > unc.mean() and t.pvalue < 0.05 else "отклонена")
    lines += ["", "## Вердикт", "", f"**H6: {verdict}.** "
              + ("Меньше 10 независимых эпизодов: статистической мощности нет, результат нельзя считать ни подтверждением, "
                 "ни опровержением." if n < 10 else "")]
    rep = "\n".join(lines)
    if not a.no_register:
        register([{"hypothesis": "H6", "asset": name.split("_")[0], "tf": "1d", "model": "event-study",
                   "features": "funding_30d_mean", "horizon_h": 720, "rule": "mean30d<0 & run>=14d, non-overlap",
                   "period_start": df.index[0], "period_end": df.index[-1], "holdout": False, "n_trades": n,
                   "hit": float((ev.fwd_30d > 0).mean()) if n else "", "mean_bps": float(ev.fwd_30d.mean() * 1e4) if n else "",
                   "n_trials_at_reg": n_trials() + 1, "verdict": verdict}])
    out = Path("reports") / f"H6_{name}.md"
    out.write_text(rep, encoding="utf-8")
    print(rep)


if __name__ == "__main__":
    main()
