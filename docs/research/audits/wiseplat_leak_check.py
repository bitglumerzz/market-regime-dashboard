"""Воспроизводимая проверка утечки меток в github.com/WISEPLAT/Hackathon-Finam-NN-Trade-Robot (2023).

    git clone --depth 1 https://github.com/WISEPLAT/hackathon-finam-nn-trade-robot /tmp/wiseplat
    python docs/research/audits/wiseplat_leak_check.py /tmp/wiseplat

1) Доля меток, которые вычисляются по самой картинке (утечка).
2) Честная постановка на тех же данных через signal_lab: предсказать следующие 10 и 60 минут, с комиссией 5 б.п.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from signal_lab import run  # noqa: E402

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/wiseplat") / "csv"
run.BARS_PER_DAY["10m"] = 84          # MOEX: ~14 часов торгов в день

for tk in ("SBER", "VTBR"):
    m1 = pd.read_csv(root / f"{tk}_M1.csv", parse_dates=["datetime"]).set_index("datetime")["close"]
    m10 = pd.read_csv(root / f"{tk}_M10.csv", parse_dates=["datetime"]).set_index("datetime")
    c10 = m10["close"]
    common = c10.index.intersection(m1.index)
    label = (c10 > c10.shift(1)).reindex(common).values                 # их метка (2_prepare_dataset…py)
    last_point = m1.reindex(common).values                               # последняя точка картинки
    prev10 = m1.values[np.clip(m1.index.get_indexer(common) - 10, 0, None)]
    print(f"{tk}: закрытие M10 = последней точке картинки в {(last_point == c10.reindex(common).values).mean():.1%}; "
          f"метка = «последняя точка > точки 10 мин назад» в {((last_point > prev10) == label).mean():.1%} случаев")
    for h in (1, 6):
        _, res = run.analyze(m10, "10m", h, fee_bps=5.0, test_days=60, min_train_days=365, name=tk)
        best = max(res["models"], key=lambda m: m.hit)
        print(f"   честно, горизонт {h * 10} мин: попадание {best.hit:.1%} на {best.trades} сделках, "
              f"средняя сделка {best.mean_ret_bps:+.1f} б.п. после комиссий, гейт {'пройден' if res['gate'] else 'не пройден'}")
