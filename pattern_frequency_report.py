"""Pattern frequency report — run Phase 1+2 classifier on TV ground-truth data
and measure which patterns actually win in real market history.

This tells us whether the new patterns (flat, triangle, diagonal, WXY,
truncation) are being detected at meaningful rates, or whether the scoring is
preventing them from ever winning vs simpler impulse/zigzag candidates.

Usage:
    python pattern_frequency_report.py             # all files
    python pattern_frequency_report.py --tf 1d     # only 1d
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from calibrate_from_tv import load_all_ground_truth, GroundTruth
from elliott import classify
from zigzag import detect_zigzag


# Per-TF threshold to use (from current calibration)
TF_THRESHOLDS = {
    "1d":  0.0571,    # 5.71%
    "4h":  0.0423,    # 4.23%
    "15m": 0.0100,    # 1.00%
    "5m":  0.0050,    # 0.5% rough
}


def run_one(gt: GroundTruth, window_size: int = 12) -> dict:
    """Classify rolling windows of swings and count pattern wins.

    Walking window approach: at each point in history, take the last
    `window_size` swings, run classify(), record top-1 pattern.
    """
    threshold = TF_THRESHOLDS.get(gt.timeframe, 0.05)
    swings = detect_zigzag(gt.prices, threshold_pct=threshold)
    if len(swings) < window_size:
        return {
            "ticker": gt.ticker, "timeframe": gt.timeframe,
            "n_swings": len(swings), "n_classifications": 0,
            "pattern_counts": {}, "skipped": True,
        }

    pattern_counter: Counter[str] = Counter()
    extension_count = 0
    truncation_count = 0
    n_classified = 0

    # Walk windows from the start; step by 1 swing
    for end in range(window_size, len(swings) + 1):
        window = swings[end - window_size:end]
        results = classify(window, top_k=1)
        if not results or results[0].score <= 0:
            continue
        top = results[0]
        pattern_counter[top.pattern] += 1
        if getattr(top, "is_extended", False):
            extension_count += 1
        if getattr(top, "is_truncated", False):
            truncation_count += 1
        n_classified += 1

    return {
        "ticker": gt.ticker,
        "timeframe": gt.timeframe,
        "n_swings": len(swings),
        "n_classifications": n_classified,
        "pattern_counts": dict(pattern_counter),
        "extension_count": extension_count,
        "truncation_count": truncation_count,
        "skipped": False,
    }


def aggregate(results: list[dict]) -> dict:
    """Sum pattern counts across all files."""
    total_classifications = 0
    total_extensions = 0
    total_truncations = 0
    patterns: Counter[str] = Counter()
    by_tf: dict[str, Counter] = defaultdict(Counter)

    for r in results:
        if r.get("skipped"):
            continue
        total_classifications += r["n_classifications"]
        total_extensions += r["extension_count"]
        total_truncations += r["truncation_count"]
        for p, c in r["pattern_counts"].items():
            patterns[p] += c
            by_tf[r["timeframe"]][p] += c

    return {
        "total_classifications": total_classifications,
        "total_extensions": total_extensions,
        "total_truncations": total_truncations,
        "global_pattern_counts": dict(patterns),
        "by_timeframe": {tf: dict(c) for tf, c in by_tf.items()},
    }


def format_report(agg: dict, results: list[dict]) -> str:
    """Pretty-print the aggregated report."""
    out = []
    out.append("=" * 70)
    out.append("PATTERN FREQUENCY REPORT — Phase 1+2 detector on TV ground-truth")
    out.append("=" * 70)

    out.append(f"\nFiles processed:       {len(results)}")
    out.append(f"Files with results:    {sum(1 for r in results if not r.get('skipped'))}")
    out.append(f"Files skipped:         {sum(1 for r in results if r.get('skipped'))}")
    out.append(f"Total classifications: {agg['total_classifications']:,}")
    out.append(f"Extended impulses:     {agg['total_extensions']:,}")
    out.append(f"Truncated impulses:    {agg['total_truncations']:,}")

    out.append(f"\n{'─' * 70}")
    out.append("GLOBAL PATTERN FREQUENCY")
    out.append(f"{'─' * 70}")
    total = agg["total_classifications"]
    if total > 0:
        sorted_patterns = sorted(agg["global_pattern_counts"].items(),
                                  key=lambda x: -x[1])
        out.append(f"{'Pattern':<28} {'Count':>8} {'%':>7}  {'bar':<25}")
        for pat, cnt in sorted_patterns:
            pct = 100 * cnt / total
            bar = "█" * int(pct / 2)
            out.append(f"{pat:<28} {cnt:>8} {pct:>6.1f}%  {bar}")

    out.append(f"\n{'─' * 70}")
    out.append("BY TIMEFRAME")
    out.append(f"{'─' * 70}")
    for tf in sorted(agg["by_timeframe"].keys()):
        counts = agg["by_timeframe"][tf]
        tf_total = sum(counts.values())
        if tf_total == 0:
            continue
        out.append(f"\n  {tf} ({tf_total:,} classifications):")
        sorted_pat = sorted(counts.items(), key=lambda x: -x[1])
        for pat, cnt in sorted_pat[:10]:
            pct = 100 * cnt / tf_total
            out.append(f"    {pat:<26} {cnt:>5} ({pct:>5.1f}%)")

    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tf", help="filter by timeframe (e.g. 1d)")
    parser.add_argument("--gt-dir", default="tv_ground_truth",
                        help="ground truth directory")
    parser.add_argument("--output-json",
                        help="optional path to save full report as JSON")
    parser.add_argument("--window-size", type=int, default=12,
                        help="rolling window of swings to classify")
    args = parser.parse_args()

    gts = load_all_ground_truth(args.gt_dir)
    if args.tf:
        gts = [g for g in gts if g.timeframe == args.tf]
    print(f"Loaded {len(gts)} ground-truth files (after TF filter={args.tf})")

    results = []
    for i, gt in enumerate(gts, 1):
        print(f"  [{i}/{len(gts)}] {gt.ticker} {gt.timeframe} ", end="", flush=True)
        try:
            r = run_one(gt, window_size=args.window_size)
            results.append(r)
            if r.get("skipped"):
                print(f"skipped (only {r['n_swings']} swings)")
            else:
                print(f"→ {r['n_classifications']} classifications, "
                      f"top: {Counter(r['pattern_counts']).most_common(1)}")
        except Exception as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}")

    agg = aggregate(results)
    report = format_report(agg, results)
    print()
    print(report)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"summary": agg, "per_file": results}, f,
                       indent=2, ensure_ascii=False)
        print(f"\nSaved JSON to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
