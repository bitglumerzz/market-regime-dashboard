"""Harvest 4h ground-truth wave labels from TradingView.

Currently tv_ground_truth/ has 76 files but almost all are 1d. The 4h
detector is calibrated against weak data. This script systematically pulls
4h Pine-indicator labels for our 3 core coins + 12 majors so 4h calibration
gets the same density as 1d.

Strategy — auto-discovery of available backends:

  1. Local tv-mcp helper (if user has tradingview-mcp wired in via Cowork):
     CLI command `tv-harvest --symbol BTCUSDT --interval 240` returns CSV
     of `date,close,wave_label` rows.

  2. Manual mode: prints TV URLs to open in the user's logged-in browser,
     guides them to copy Pine output and save into tv_ground_truth/ — fallback
     when no MCP available. Slower but always works.

Output: `tv_ground_truth/{SYMBOL}_4h.csv` matching the existing 1d schema.

Usage:
    python harvest_4h_ground_truth.py --auto         # try MCP, exit if absent
    python harvest_4h_ground_truth.py --manual       # print instructions
    python harvest_4h_ground_truth.py --symbols BTCUSDT ETHUSDT --auto
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Core 3 + 12 majors with high 4h liquidity
DEFAULT_SYMBOLS = [
    "BTCUSDT",  "ETHUSDT",  "HYPEUSDT",
    "SOLUSDT",  "ADAUSDT",  "AVAXUSDT", "DOTUSDT",
    "LINKUSDT", "DOGEUSDT", "ATOMUSDT", "NEARUSDT", "XRPUSDT",
    "LTCUSDT",  "BCHUSDT",  "MATICUSDT",
]

GT_DIR = "tv_ground_truth"


def _command_exists(cmd: str) -> bool:
    from shutil import which
    return which(cmd) is not None


def _try_mcp_harvest(symbol: str, interval: str = "240") -> Path | None:
    """Try the local tv-mcp CLI helper. Returns saved path or None on failure."""
    helpers = ["tv-harvest", "tradingview-harvest"]
    out_path = Path(GT_DIR) / f"{symbol}_4h.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for helper in helpers:
        if not _command_exists(helper):
            continue
        try:
            result = subprocess.run(
                [helper, "--symbol", symbol, "--interval", interval,
                 "--out", str(out_path)],
                capture_output=True, timeout=120, text=True,
            )
            if result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
                return out_path
            print(f"  ! {helper} non-zero exit: {result.stderr[:120]}")
        except subprocess.TimeoutExpired:
            print(f"  ! {helper} timed out for {symbol}")
        except Exception as exc:
            print(f"  ! {helper} failed: {exc}")
    return None


def _print_manual_instructions(symbols: list[str]) -> None:
    print("=" * 60)
    print("MANUAL 4h HARVEST — TV MCP not detected locally")
    print("=" * 60)
    print("Open each URL in your browser, ensure your Pine wave indicator")
    print("is applied, then use TV's Export Chart Data to download CSV.")
    print(f"Save into {GT_DIR}/ with the filename pattern: SYMBOL_4h.csv")
    print(f"Required CSV columns: date,close,wave_label")
    print()
    for sym in symbols:
        tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{sym}&interval=240"
        print(f"  {sym:<12} → {tv_url}")
    print()
    print("Once all files are saved, run the TV Calibration page to refit.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--auto",   action="store_true",
                    help="auto-harvest via TV MCP (fails if MCP absent)")
    p.add_argument("--manual", action="store_true",
                    help="print manual instructions (fallback)")
    args = p.parse_args()

    if not (args.auto or args.manual):
        # If neither flag — try auto first, fall through to manual on failure
        args.auto = True

    if args.auto:
        print(f"Attempting auto-harvest for {len(args.symbols)} symbols at 4h…")
        saved = 0
        failed: list[str] = []
        for sym in args.symbols:
            print(f"  • {sym} ", end="", flush=True)
            path = _try_mcp_harvest(sym, "240")
            if path is not None:
                print(f"→ {path}")
                saved += 1
            else:
                print("FAILED")
                failed.append(sym)
        print(f"\nSaved {saved}/{len(args.symbols)}")
        if failed and not args.manual:
            print(f"\nFailed: {failed}")
            print("Re-run with --manual for instructions on remaining symbols.")
            return 1 if saved == 0 else 0
        if args.manual and failed:
            _print_manual_instructions(failed)
        return 0

    if args.manual:
        _print_manual_instructions(args.symbols)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
